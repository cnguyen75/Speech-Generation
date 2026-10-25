"""Write results/results.json: every number the README, figures and site use, read from runs/.

Usage: python scripts/export_results.py <commit-of-the-runs>
"""

import json
from pathlib import Path
import statistics as st
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from speechopt.common import ROOT, RUNS, load_manifest, write_json  # noqa: E402


COMMIT = sys.argv[1] if len(sys.argv) > 1 else "unknown"
OPTS = ["sgd", "adam", "adamw", "rmsprop", "lbfgs"]
plan = json.loads((RUNS / "plan.json").read_text())
summary = json.loads((RUNS / "results.json").read_text())
ref = json.loads((RUNS / "ref" / "eval.json").read_text())
man = load_manifest()
load = lambda p: json.loads(Path(p).read_text())


def agg(vals: list[float]) -> dict[str, float]:
    return dict(mean=st.mean(vals), sd=st.stdev(vals), min=min(vals), max=max(vals))


final = {}
for o in OPTS:
    final[o] = {"learning_rate": plan["chosen_lr"][o], "seeds": {}}
    for s in range(5):
        r = load(RUNS / f"final_{o}_s{s}" / "result.json")
        row = dict(
            status=r["status"],
            steps=r["steps"],
            train_seconds=r["train_time"],
            loss_evaluations=r["loss_evals"],
            median_step_seconds=r["median_step_time"],
            mean_load_1min=r["mean_load1"],
        )
        for b in ("steps", "time"):
            e = load(RUNS / f"final_{o}_s{s}" / f"eval_{b}.json")
            a = e["agg"]
            row[f"{b}_budget"] = dict(
                checkpoint_step=e["checkpoint_step"],
                mel_mse=a["mel_mse"],
                mcd_db=a["mcd"],
                wer=a["wer"],
                stop_failures=a["stop_failures"],
                wer_capped_posthoc=sum(min(u["word_errors"], u["ref_words"]) for u in e["per_utt"])
                / sum(u["ref_words"] for u in e["per_utt"]),
            )
        final[o]["seeds"][str(s)] = row
    for b in ("steps", "time"):
        final[o][f"{b}_budget_summary"] = {
            k: agg([final[o]["seeds"][str(s)][f"{b}_budget"][k] for s in range(5)])
            for k in ("mel_mse", "mcd_db", "wer", "stop_failures", "wer_capped_posthoc")
        }

verdicts = {
    b: {
        o: summary["budgets"][b]["table"][o].get("vs_best")
        for o in summary["budgets"][b]["ranking_by_mel_mse"][1:]
    }
    for b in ("steps", "time")
}

# post-hoc: stop failures at every evaluated checkpoint, and failed sentences the recogniser scored as error-free
by_ckpt, failed_total, failed_perfect, example = [], 0, 0, None
for o in OPTS:
    for s in range(5):
        seen = {}
        for b in ("steps", "time"):
            e = load(RUNS / f"final_{o}_s{s}" / f"eval_{b}.json")
            seen[e["checkpoint_step"]] = sum(u["stop_failure"] for u in e["per_utt"])
            if b == "steps":
                for u in e["per_utt"]:
                    if u["stop_failure"]:
                        failed_total += 1
                        if u["word_errors"] == 0:
                            failed_perfect += 1
                            if o == "adam" and s == 4 and example is None:
                                example = dict(
                                    run=f"final_{o}_s{s}",
                                    id=u["id"],
                                    ref_words=u["ref_words"],
                                    transcript=u["hyp"].strip(),
                                )
        by_ckpt += [
            dict(optimizer=o, seed=s, step=k, stop_failures=v) for k, v in sorted(seen.items())
        ]

audio = load(ROOT / "results" / "audio" / "manifest.json")
step_spread = {
    o: dict(
        min=min(final[o]["seeds"][str(s)]["median_step_seconds"] for s in range(5)),
        max=max(final[o]["seeds"][str(s)]["median_step_seconds"] for s in range(5)),
        median=st.median(final[o]["seeds"][str(s)]["median_step_seconds"] for s in range(5)),
    )
    for o in OPTS
}

write_json(
    ROOT / "results" / "results.json",
    {
        "about": "Optimizer study: fine-tuning a pretrained Tacotron 2 (LJSpeech) to a new LibriTTS-R reader. "
        "All numbers are read from runs/ by scripts/export_results.py.",
        "runs_commit": COMMIT,
        "models": {
            "tts": "speechbrain/tts-tacotron2-ljspeech (Apache-2.0)",
            "vocoder": "speechbrain/tts-hifigan-ljspeech (Apache-2.0), frozen",
            "asr": "openai/whisper-small.en (Apache-2.0), greedy decoding, Whisper English normaliser",
        },
        "data": {
            "source": man["source"],
            "speaker": man["speaker"],
            "splits": man["splits"],
            "clips_excluded_by_1_to_10_s_rule": man["excluded_by_length"],
            "showcase_ids": man["showcase"],
        },
        "protocol": {
            "batch": plan["batch"],
            "grad_clip_first_order": plan["clip_norm"],
            "validate_every": plan["val_every"],
            "tune_steps": plan["tune_steps"],
            "tune_seed": plan["tune_seed"],
            "grids": plan["grids"],
            "final_steps": plan["final_steps"],
            "time_budget_seconds": plan["time_budget"],
            "seeds": plan["seeds"],
            "selection": "best validation teacher-forced mel MSE within each budget",
            "primary_metric": "test teacher-forced post-net log-mel MSE at the step budget",
            "clear_rule": "95% paired bootstrap CI excludes 0 AND per-seed ranges do not overlap AND |mean diff| >= 1% of best (1 point for WER)",
            "device": plan["device"],
            "torch": plan["torch"],
        },
        "tuning": {
            "validation_mel_mse": plan["tuning"],
            "chosen_lr": plan["chosen_lr"],
            "any_at_grid_edge": False,
        },
        "references_test": {
            k: ref[k]["agg"] for k in ("pretrained", "ground_truth", "copy_synthesis")
        },
        "final": final,
        "ranking_by_mel_mse": {
            b: summary["budgets"][b]["ranking_by_mel_mse"] for b in ("steps", "time")
        },
        "verdicts_vs_best": verdicts,
        "load": {
            "flag_rule": "mean 1-min load > 6, or median step time > 1.15 x the optimizer's median",
            "flagged": summary["load_flags"],
            "median_step_seconds_spread": step_spread,
        },
        "post_hoc": {
            "note": "Outside the pre-set rules; existing checkpoints only; no retraining or retuning.",
            "stop_failures_by_checkpoint": by_ckpt,
            "failed_to_stop_scored_error_free_by_asr": {
                "failed_sentences": failed_total,
                "with_zero_word_errors": failed_perfect,
                "scope": "all 25 runs, step-budget checkpoints, 50 test sentences each",
                "example": example,
            },
            "wer_capped": "per-sentence word errors capped at the sentence's word count; see final.*.wer_capped_posthoc",
        },
        "audio": {
            "manifest": "audio/manifest.json",
            "tail_rule": audio["tail_rule"],
            "seed_rule": audio["seed_rule"],
            "reduced_clips": [
                c["file"] for c in audio["clips"] if c.get("tail") and c["tail"]["reduced_db"]
            ],
        },
        "commands": [
            "python scripts/stream_extract.py (LibriTTS-R train-clean-100 stream, female readers)",
            "python scripts/prepare_data.py",
            "scripts/overnight.sh (runs scripts/run_study.py, then summarize.py)",
            "python scripts/export_audio.py",
            "python scripts/make_figures.py <commit>",
            "python scripts/export_results.py <commit>",
            "pytest tests",
        ],
    },
)
r = load(ROOT / "results" / "results.json")
print("failed sentences:", failed_total, "scored error-free:", failed_perfect, "example:", example)
print("reduced clips:", r["audio"]["reduced_clips"])
