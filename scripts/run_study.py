"""Resumable overnight optimizer study. Re-running the same command skips every finished run.

Phases (all decisions use validation data only; the test split is scored but never used to choose):
  0. references on the test split: pretrained model, real recordings, copy-synthesis
  1. tuning: 3 half-decade learning rates per optimizer, 150 steps, seed 100; the grid is extended
     (at most twice) when the best rate is at an edge; selection by best validation mel MSE
  2. final: seeds 0-4, cut to 4 or at least 3 if the projected job exceeds 10 h; optimizers interleaved by seed
     with the order rotated each seed; each run trains until it has done 400 steps AND used the
     time budget T (= 400 x Adam's median tuned step time); the best-validation checkpoint inside
     each budget is scored on the test split
"""
import argparse
import datetime as dt
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from speechopt import train as T  # noqa: E402
from speechopt.common import RUNS, device, load_manifest, load_tacotron2, load_vocoder, write_json  # noqa: E402
from speechopt.evaluate import ASR, evaluate_checkpoint, evaluate_references  # noqa: E402
from speechopt.runtime import ActiveClock, lr_id  # noqa: E402

OPTS = ["sgd", "adam", "adamw", "rmsprop", "lbfgs"]
GRIDS = {"sgd": [3e-3, 1e-2, 3e-2], "adam": [3e-5, 1e-4, 3e-4], "adamw": [3e-5, 1e-4, 3e-4],
         "rmsprop": [1e-5, 3e-5, 1e-4], "lbfgs": [0.1, 0.3, 1.0]}
HALF_DECADE = math.sqrt(10)
TUNE_STEPS, TUNE_SEED, MAX_EXTENSIONS = 150, 100, 2
FINAL_STEPS, SEEDS, MAX_HOURS = 400, [0, 1, 2, 3, 4], 10.0

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default=str(RUNS))
ap.add_argument("--smoke", action="store_true", help="tiny end-to-end check on validation data")
args = ap.parse_args()
OUT = Path(args.runs)
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "study.log"
if args.smoke:
    GRIDS = {k: v[1:2] for k, v in GRIDS.items()}
    TUNE_STEPS, FINAL_STEPS, SEEDS, MAX_EXTENSIONS = 4, 4, [0], 0
    T.VAL_EVERY = 2


clock = ActiveClock(OUT / "sessions.json")  # running time only; idle gaps between sessions excluded


def log(msg):
    clock.touch()
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


log(f"=== start (pid {os.getpid()}) load {os.getloadavg()}")
manifest = load_manifest()
train, val, test = (T.Clips(manifest, s) for s in ("train", "val", "test"))
if args.smoke:
    train.items, train.text, train.mel = train.items[:40], train.text[:40], train.mel[:40]
    test = T.Clips(manifest, "val")
    test.items, test.text, test.mel = test.items[:3], test.text[:3], test.mel[:3]
log(f"speaker {manifest['speaker']}: train {len(train)} / val {len(val)} / test {len(test)} clips")
dev = device()
model = load_tacotron2()
pristine = {k: v.detach().clone() for k, v in model.state_dict().items()}
cpu_model = load_tacotron2()
vocoder = load_vocoder()
asr = ASR(dev)

plan_path = OUT / "plan.json"
plan = json.loads(plan_path.read_text()) if plan_path.exists() else {
    "started": time.time(), "baseline_load": os.getloadavg(), "device": str(dev),
    "torch": torch.__version__, "batch": T.BATCH, "clip_norm": T.CLIP_NORM, "val_every": T.VAL_EVERY,
    "tune_steps": TUNE_STEPS, "tune_seed": TUNE_SEED, "final_steps": FINAL_STEPS, "grids": GRIDS}
write_json(plan_path, plan)
consecutive_failures = 0

# ---------- phase 0: references ----------
ref_path = OUT / "ref" / "eval.json"
if not ref_path.exists():
    t0 = time.time()
    agg, utt = evaluate_checkpoint(pristine, model, cpu_model, vocoder, asr, test, dev,
                                   OUT / "ref" / "showcase" / "pretrained", manifest["showcase"])
    t_eval = time.time() - t0
    refs = evaluate_references(vocoder, asr, test, OUT / "ref" / "showcase", manifest["showcase"])
    write_json(ref_path, {"pretrained": {"agg": agg, "per_utt": utt},
                          **{k: {"agg": a, "per_utt": u} for k, (a, u) in refs.items()},
                          "eval_seconds_per_checkpoint": t_eval})
    log(f"references: pretrained {agg} | real {refs['ground_truth'][0]} | copy-synth {refs['copy_synthesis'][0]} "
        f"| eval took {t_eval:.0f}s")
eval_seconds = json.loads(ref_path.read_text())["eval_seconds_per_checkpoint"]


def do_run(run_id, opt, lr, seed, steps, budget, save):
    """Run once (or load the finished result). Returns the result dict."""
    global consecutive_failures
    d = OUT / run_id
    if (d / "result.json").exists():
        return json.loads((d / "result.json").read_text())
    log(f"run {run_id}: {opt} lr {lr:g} seed {seed} steps {steps} time budget {budget} load {os.getloadavg()[0]:.1f}")
    wall0 = time.time()
    try:
        res, ckpts = T.run(model, pristine, train, val, dev, opt, lr, seed, steps, budget, log)
        if save:
            d.mkdir(parents=True, exist_ok=True)
            for k, sd in ckpts.items():
                torch.save(sd, d / f"best_{k}.pt")
        consecutive_failures = 0
    except Exception:
        res = dict(optimizer=opt, lr=lr, seed=seed, status="failed", error=traceback.format_exc())
        consecutive_failures += 1
        log(f"run {run_id} FAILED:\n{res['error']}")
    res["wall_seconds"] = time.time() - wall0
    write_json(d / "result.json", res)
    log(f"run {run_id} done: status {res['status']}, {res.get('steps')} steps, train {res.get('train_time', 0):.0f}s, "
        f"wall {res['wall_seconds']:.0f}s, best {res.get('best')}")
    if consecutive_failures >= 3:
        log("three consecutive runs failed; stopping the job")
        sys.exit(2)
    return res


# ---------- phase 1: tuning ----------
chosen = plan.get("chosen_lr", {})
for opt in OPTS:
    if opt in chosen:
        continue
    grid, extensions = sorted(GRIDS[opt]), 0
    while True:
        results = {lr: do_run(f"tune_{opt}_lr{lr_id(lr)}", opt, lr, TUNE_SEED, TUNE_STEPS, None, False) for lr in grid}
        scores = {lr: (r.get("best", {}).get("steps", {}).get("val_mse") or math.inf) for lr, r in results.items()}
        best_lr = min(scores, key=scores.get)
        if extensions < MAX_EXTENSIONS and best_lr in (grid[0], grid[-1]) and math.isfinite(scores[best_lr]):
            grid = sorted(grid + [grid[0] / HALF_DECADE if best_lr == grid[0] else grid[-1] * HALF_DECADE])
            extensions += 1
            continue
        break
    chosen[opt] = best_lr
    plan.setdefault("tuning", {})[opt] = {lr_id(lr): s for lr, s in scores.items()}
    plan["chosen_lr"] = chosen
    write_json(plan_path, plan)
    log(f"tuning {opt}: val {plan['tuning'][opt]} -> lr {best_lr:g}{' (at grid edge after max extensions)' if best_lr in (grid[0], grid[-1]) else ''}")

# ---------- budgets and seed count ----------
if "time_budget" not in plan:
    tune = {opt: json.loads((OUT / f"tune_{opt}_lr{lr_id(chosen[opt])}" / "result.json").read_text()) for opt in OPTS}
    plan["time_budget"] = FINAL_STEPS * tune["adam"]["median_step_time"]
    per_seed = 0.0
    for opt in OPTS:
        r = tune[opt]
        step_t = r.get("median_step_time") or tune["adam"]["median_step_time"]
        overhead = max(0.0, r["wall_seconds"] - r.get("train_time", 0)) / max(1, r.get("steps", TUNE_STEPS)) * FINAL_STEPS
        per_seed += max(FINAL_STEPS * step_t, plan["time_budget"]) + overhead + 2 * eval_seconds
    elapsed = clock.active_seconds()
    n = len(SEEDS)  # largest seed count (5, then 4, floor 3) whose projection fits in MAX_HOURS
    while not args.smoke and n > 3 and (elapsed + n * per_seed) / 3600 > MAX_HOURS:
        n -= 1
    plan["seeds"] = SEEDS[:n]
    plan["projection"] = {"seconds_per_seed": per_seed, "elapsed_hours_at_decision": elapsed / 3600,
                          "projected_total_hours_5_seeds": (elapsed + len(SEEDS) * per_seed) / 3600,
                          "projected_finish": dt.datetime.fromtimestamp(time.time() + n * per_seed).isoformat(timespec="minutes")}
    write_json(plan_path, plan)
    log(f"time budget T = {plan['time_budget']:.0f}s; seeds {plan['seeds']}; projection {plan['projection']}")

# ---------- phase 2: final runs ----------
for si, seed in enumerate(plan["seeds"]):
    for opt in OPTS[si % len(OPTS):] + OPTS[: si % len(OPTS)]:
        run_id = f"final_{opt}_s{seed}"
        res = do_run(run_id, opt, chosen[opt], seed, FINAL_STEPS, plan["time_budget"], True)
        for budget in ("steps", "time"):
            ev, ck = OUT / run_id / f"eval_{budget}.json", OUT / run_id / f"best_{budget}.pt"
            if ev.exists() or not ck.exists():
                continue
            agg, utt = evaluate_checkpoint(torch.load(ck), model, cpu_model, vocoder, asr, test, dev,
                                           OUT / run_id / f"showcase_{budget}", manifest["showcase"])
            write_json(ev, {"agg": agg, "per_utt": utt, "checkpoint_step": res["best"][budget]["step"]})
            log(f"  eval {run_id} [{budget} budget, step {res['best'][budget]['step']}]: {agg}")
    done = sum((OUT / f"final_{o}_s{s}" / "eval_time.json").exists() or
               (OUT / f"final_{o}_s{s}" / "result.json").exists() for o in OPTS for s in plan["seeds"])
    left = len(plan["seeds"]) - si - 1
    eta = dt.datetime.now() + dt.timedelta(seconds=left * plan["projection"]["seconds_per_seed"])
    log(f"seed {seed} complete ({done}/{len(OPTS) * len(plan['seeds'])} final runs); elapsed "
        f"{clock.active_seconds() / 3600:.1f} h running; projected finish {eta:%Y-%m-%d %H:%M}")

log("=== all runs finished; summarising")
os.execv(sys.executable, [sys.executable, str(Path(__file__).with_name("summarize.py")), "--runs", str(OUT)])
