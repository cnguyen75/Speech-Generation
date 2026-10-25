"""Aggregate the study into runs/results.json and put a plain-text summary at the top of the log.

Rules fixed before the results were seen:
- Primary metric: teacher-forced post-net log-mel MSE on the test split at the 400-step budget.
  Secondary: MCD (dB), Whisper WER, stop-token failures; and everything again at the time budget.
- A difference to the best optimizer is called "clear" only if (a) the 95% paired bootstrap CI over
  test utterances (seed-averaged, 10,000 resamples) excludes 0, (b) the two optimizers' per-seed
  ranges do not overlap, and (c) the mean difference is at least 1% of the best mean (1 percentage
  point for WER). Otherwise it is reported as within noise.
- Load flag: a run is flagged when its mean 1-minute load average exceeds 6, or its median step
  time is more than 15% above the median for the same optimizer across final runs.
"""

import argparse
import json
import os
from pathlib import Path
import statistics as st
import sys
from typing import Any

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from speechopt.common import RUNS, load_manifest, write_json  # noqa: E402


OPTS = ["sgd", "adam", "adamw", "rmsprop", "lbfgs"]
NAMES = {
    "sgd": "SGD+momentum",
    "adam": "Adam",
    "adamw": "AdamW",
    "rmsprop": "RMSprop",
    "lbfgs": "L-BFGS",
}
METRICS = ["mel_mse", "mcd", "wer"]

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default=str(RUNS))
OUT = Path(ap.parse_args().runs)
plan = json.loads((OUT / "plan.json").read_text())
refs = json.loads((OUT / "ref" / "eval.json").read_text())
seeds = plan["seeds"]
rng = np.random.default_rng(0)


def load(p: Path) -> dict[str, Any] | None:
    return json.loads(p.read_text()) if p.exists() else None


runs = {o: {s: load(OUT / f"final_{o}_s{s}" / "result.json") for s in seeds} for o in OPTS}
evals = {
    b: {o: {s: load(OUT / f"final_{o}_s{s}" / f"eval_{b}.json") for s in seeds} for o in OPTS}
    for b in ("steps", "time")
}

base = 6.0
med_step = {
    o: st.median(
        [r["median_step_time"] for r in runs[o].values() if r and "median_step_time" in r] or [0]
    )
    for o in OPTS
}
flags = []
for o in OPTS:
    for s, r in runs[o].items():
        if not r or "mean_load1" not in r:
            continue
        why = []
        if r["mean_load1"] > base:
            why.append(f"mean load {r['mean_load1']:.1f} > {base:.1f}")
        if med_step[o] and r["median_step_time"] > 1.15 * med_step[o]:
            why.append(f"median step {r['median_step_time']:.2f}s > 1.15 x {med_step[o]:.2f}s")
        if why:
            flags.append({"run": f"final_{o}_s{s}", "reasons": why})


def per_utt_matrix(budget: str, o: str, key: str) -> tuple[np.ndarray, np.ndarray | None] | None:
    rows = [e["per_utt"] for e in evals[budget][o].values() if e]
    if not rows:
        return None
    if key == "wer":
        return np.array([[u["word_errors"] for u in r] for r in rows]), np.array(
            [[u["ref_words"] for u in r] for r in rows]
        )
    return np.array([[u[key] for u in r] for r in rows]), None


def stat(a: np.ndarray, w: np.ndarray | None) -> float:
    return a.sum() / w.sum() if w is not None else a.mean()


def boot_diff(budget: str, o: str, best: str, key: str, n: int = 10000) -> list[float] | None:
    A, B = per_utt_matrix(budget, o, key), per_utt_matrix(budget, best, key)
    if A is None or B is None:
        return None
    a, wa = A[0].mean(0), (A[1].mean(0) if A[1] is not None else None)
    b, wb = B[0].mean(0), (B[1].mean(0) if B[1] is not None else None)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    if key == "wer":
        d = a[idx].sum(1) / wa[idx].sum(1) - b[idx].sum(1) / wb[idx].sum(1)
    else:
        d = a[idx].mean(1) - b[idx].mean(1)
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


results = {
    "plan": plan,
    "references": {k: v["agg"] for k, v in refs.items() if isinstance(v, dict)},
    "load_flags": flags,
    "budgets": {},
}
lines = []
for budget in ("steps", "time"):
    table = {}
    for o in OPTS:
        per_seed = {}
        for s in seeds:
            e, r = evals[budget][o][s], runs[o][s]
            per_seed[s] = dict(
                status=(r or {}).get("status", "missing"),
                checkpoint_step=(e or {}).get("checkpoint_step"),
                **({k: e["agg"][k] for k in METRICS + ["stop_failures"]} if e else {}),
            )
        vals = {k: [p[k] for p in per_seed.values() if k in p] for k in METRICS + ["stop_failures"]}
        table[o] = dict(
            lr=plan["chosen_lr"][o],
            per_seed=per_seed,
            n_ok=sum(p["status"] == "ok" for p in per_seed.values()),
            n_diverged=sum(p["status"] == "diverged" for p in per_seed.values()),
            n_failed=sum(p["status"] in ("failed", "missing") for p in per_seed.values()),
            **{
                k: dict(
                    mean=float(np.mean(v)) if v else None,
                    sd=float(np.std(v, ddof=1)) if len(v) > 1 else None,
                    min=min(v) if v else None,
                    max=max(v) if v else None,
                )
                for k, v in vals.items()
            },
        )
    ranked = sorted(
        [o for o in OPTS if table[o]["mel_mse"]["mean"] is not None],
        key=lambda o: table[o]["mel_mse"]["mean"],
    )
    best = ranked[0] if ranked else None
    for o in ranked[1:]:
        table[o]["vs_best"] = {}
        for k in METRICS:
            ci = boot_diff(budget, o, best, k)
            overlap = not (
                table[o][k]["min"] > table[best][k]["max"]
                or table[o][k]["max"] < table[best][k]["min"]
            )
            diff = table[o][k]["mean"] - table[best][k]["mean"]
            big = abs(diff) >= (0.01 if k == "wer" else 0.01 * abs(table[best][k]["mean"]))
            table[o]["vs_best"][k] = dict(
                ci95=ci,
                seed_ranges_overlap=overlap,
                mean_diff=diff,
                verdict="clear"
                if ci and (ci[0] > 0 or ci[1] < 0) and not overlap and big
                else "within noise",
            )
    results["budgets"][budget] = dict(ranking_by_mel_mse=ranked, table=table)

    hdr = (
        "400-step budget"
        if budget == "steps"
        else f"time budget ({plan['time_budget'] / 60:.1f} min of training)"
    )
    lines.append(f"\n{hdr} -- mean ± sd over seeds (test split)")
    lines.append(
        f"  {'optimizer':<13}{'lr':>8}  {'mel MSE':>16}  {'MCD dB':>13}  {'WER %':>13}  "
        "stop-fail  ok/div/fail  vs best (mel MSE)"
    )
    for o in ranked + [o for o in OPTS if o not in ranked]:
        t = table[o]
        f = lambda k, m=1: (
            f"{t[k]['mean'] * m:.3f}±{(t[k]['sd'] or 0) * m:.3f}"
            if t[k]["mean"] is not None
            else "n/a"
        )
        vb = t.get("vs_best", {}).get("mel_mse")
        vs = (
            "best"
            if o == best
            else (
                f"{vb['verdict']} (CI {vb['ci95'][0]:+.4f}..{vb['ci95'][1]:+.4f})"
                if vb and vb["ci95"]
                else ""
            )
        )
        lines.append(
            f"  {NAMES[o]:<13}{t['lr']:>8.0e}  {f('mel_mse'):>16}  {f('mcd'):>13}  "
            f"{f('wer', 100):>13}  "
            f"{t['stop_failures']['mean'] if t['stop_failures']['mean'] is not None else 'n/a':>9}"
            "  "
            f"{t['n_ok']}/{t['n_diverged']}/{t['n_failed']:<9}  {vs}"
        )

ref = results["references"]
manifest = load_manifest()
head = [
    "=" * 100,
    "SUMMARY (written at the end of the job; full log follows)",
    f"Target reader: LibriTTS-R speaker {manifest['speaker']}; clips train/val/test = "
    + " / ".join(
        f"{manifest['splits'][k]['clips']} ({manifest['splits'][k]['minutes']} min)"
        for k in ("train", "val", "test")
    ),
    f"Seeds: {seeds}; chosen learning rates: {plan['chosen_lr']}; "
    f"time budget {plan['time_budget']:.0f}s",
    "References on the test split:",
]
for k, name in (
    ("pretrained", "pretrained LJSpeech model"),
    ("copy_synthesis", "real mel -> HiFi-GAN"),
    ("ground_truth", "real recording"),
):
    a = ref[k]
    mse = f"{a['mel_mse']:.3f}" if "mel_mse" in a else "n/a (no prediction)"
    mcd_s = "n/a (is the reference)" if k == "ground_truth" else f"{a['mcd']:.2f} dB"
    head.append(f"  {name:<28} mel MSE {mse}  MCD {mcd_s}  WER {100 * a['wer']:.1f}%")
head += lines
for budget in ("steps", "time"):
    r = results["budgets"][budget]
    if r["ranking_by_mel_mse"]:
        b = r["ranking_by_mel_mse"][0]
        clear = [
            o
            for o in r["ranking_by_mel_mse"][1:]
            if r["table"][o]["vs_best"]["mel_mse"]["verdict"] == "clear"
        ]
        noise = [o for o in r["ranking_by_mel_mse"][1:] if o not in clear]
        head.append(
            f"[{budget} budget] lowest mean test mel MSE: {NAMES[b]}. "
            f"Clearly worse: {', '.join(NAMES[o] for o in clear) or 'none'}. "
            f"Within noise of the best: {', '.join(NAMES[o] for o in noise) or 'none'}."
        )
head.append(
    "Load flags: "
    + ("; ".join(f"{f['run']} ({', '.join(f['reasons'])})" for f in flags) if flags else "none")
)
head.append("=" * 100)
write_json(OUT / "results.json", results)
log_path = OUT / "study.log"
old = log_path.read_text() if log_path.exists() else ""
tmp = log_path.with_suffix(".tmp")
tmp.write_text("\n".join(head) + "\n\n" + old)
os.replace(tmp, log_path)
print("\n".join(head))
