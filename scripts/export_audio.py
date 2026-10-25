"""Export the showcase audio: 5 fixed test sentences x (real recording, pretrained, 5 optimizers).

Rules (fixed before export, see README):
- Sentences: the 5 showcase test clips chosen by prepare_data.py before training. Only text the
  reader actually recorded is synthesised; each clip is paired with the real recording.
- Each optimizer's clips come from its median-seed run by test mel MSE (step budget), checkpoint
  best_steps.pt, generated with the evaluation seed, so they are the exact audio that was scored.
- Clips are published as generated, mono 22.05 kHz MP3, with one exception: if a clip failed to
  stop, its tail starts 20 frames (0.23 s) after attention first reaches its furthest character,
  and when that tail is louder (RMS) than the speech before it, the tail is reduced by 20 dB
  (one tenth of the amplitude) with a 50 ms fade-in of the reduction. Quiet failures are untouched.
"""

import json
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from speechopt.common import (
    ROOT,
    RUNS,
    SR,
    load_manifest,
    load_tacotron2,
    load_vocoder,
    text_to_ids,
    write_json,
)  # noqa: E402
from speechopt.evaluate import SYNTH_SEED, real_wav  # noqa: E402
from speechopt.train import Clips  # noqa: E402


OUT = ROOT / "results" / "audio"
OPTS = ["sgd", "adam", "adamw", "rmsprop", "lbfgs"]
HOP, MARGIN_FRAMES, TAIL_DB, FADE_S, MP3_LEVEL, MAX_BYTES = 256, 20, -20.0, 0.05, 0.6, 150 * 1024

manifest = load_manifest()
test = Clips(manifest, "test")
ids_all = [c["id"] for c in test.items]
showcase = manifest["showcase"]
model, vocoder = load_tacotron2(), load_vocoder()
OUT.mkdir(parents=True, exist_ok=True)


def write_mp3(path: Path, wav: np.ndarray) -> int:
    sf.write(
        str(path),
        np.clip(wav, -1.0, 1.0),
        SR,
        format="MP3",
        subtype="MPEG_LAYER_III",
        compression_level=MP3_LEVEL,
        bitrate_mode="CONSTANT",
    )
    size = path.stat().st_size
    assert size < MAX_BYTES, f"{path.name} is {size} bytes"
    return size


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2))) if len(x) else 0.0


@torch.no_grad()
def generate(uid: str) -> tuple[np.ndarray, int, np.ndarray]:
    i = ids_all.index(uid)
    torch.manual_seed(SYNTH_SEED + i)  # same seed as evaluate.synthesize
    ids = text_to_ids(test.items[i]["text"])[None]
    mel, mel_len, align = model.infer(ids, torch.tensor([ids.shape[1]]))
    return vocoder.decode_batch(mel).squeeze().numpy(), int(mel_len[0]), align[0].argmax(-1).numpy()


clips = []
for n, uid in enumerate(showcase, 1):
    text = next(c["text"] for c in test.items if c["id"] == uid)
    real = real_wav(uid)
    clips.append(
        dict(
            sentence=n,
            id=uid,
            text=text,
            system="ground_truth",
            file=f"s{n}-ground-truth.mp3",
            seconds=round(len(real) / SR, 2),
            bytes=write_mp3(OUT / f"s{n}-ground-truth.mp3", real),
        )
    )
ref = json.loads((RUNS / "ref" / "eval.json").read_text())
ref_utt = {u["id"]: u for u in ref["pretrained"]["per_utt"]}

systems = [("pretrained", None, None)]
for o in OPTS:
    mse = {
        s: json.loads((RUNS / f"final_{o}_s{s}" / "eval_steps.json").read_text())["agg"]["mel_mse"]
        for s in range(5)
    }
    systems.append((o, sorted(mse, key=mse.get)[2], mse))

for system, seed, _mse in systems:
    if system == "pretrained":
        model.load_state_dict({k: v for k, v in load_tacotron2().state_dict().items()})
        per_utt, step = ref_utt, 0
    else:
        run = RUNS / f"final_{system}_s{seed}"
        model.load_state_dict(torch.load(run / "best_steps.pt"))
        ev = json.loads((run / "eval_steps.json").read_text())
        per_utt, step = {u["id"]: u for u in ev["per_utt"]}, ev["checkpoint_step"]
    model.eval()
    for n, uid in enumerate(showcase, 1):
        wav, frames, argmax = generate(uid)
        u = per_utt[uid]
        assert bool(frames >= 1000) == bool(u["stop_failure"]), (
            "regenerated clip disagrees with the evaluation"
        )
        info = dict(
            sentence=n,
            id=uid,
            system=system,
            seed=seed,
            checkpoint_step=step,
            stop_failure=bool(u["stop_failure"]),
            asr_transcript=u["hyp"].strip(),
            word_errors=u["word_errors"],
            ref_words=u["ref_words"],
            mcd_db=round(u["mcd"], 2),
            seconds=round(len(wav) / SR, 2),
            tail=None,
        )
        if u["stop_failure"]:
            start_frame = int(np.argmax(argmax == argmax.max())) + MARGIN_FRAMES
            cut = min(len(wav), start_frame * HOP)
            before, after = rms(wav[:cut]), rms(wav[cut:])
            reduced = after > before
            if reduced:
                gain = np.ones_like(wav)
                fade = int(FADE_S * SR)
                ramp = np.linspace(1.0, 10 ** (TAIL_DB / 20), fade)
                gain[cut : cut + fade] = ramp[: len(gain[cut : cut + fade])]
                gain[cut + fade :] = 10 ** (TAIL_DB / 20)
                wav = wav * gain
            info["tail"] = dict(
                starts_at_s=round(cut / SR, 2),
                speech_rms=round(before, 4),
                tail_rms_as_generated=round(after, 4),
                reduced_db=TAIL_DB if reduced else 0.0,
            )
        name = f"s{n}-{system}.mp3"
        info.update(file=name, bytes=write_mp3(OUT / name, wav))
        clips.append(info)
        print(
            name,
            "FAILED TO STOP" if info["stop_failure"] else "",
            info["tail"] or "",
            f"{info['bytes'] // 1024} KB",
        )

write_json(
    OUT / "manifest.json",
    {
        "note": "Synthetic speech in the voice of LibriTTS-R reader 40 (CC BY 4.0), "
        "test-set sentences only, each paired "
        "with the real recording. Published as generated except where noted in `tail`.",
        "tail_rule": {
            "margin_frames": MARGIN_FRAMES,
            "reduction_db": TAIL_DB,
            "fade_s": FADE_S,
            "condition": "clip failed to stop AND tail RMS > speech RMS",
        },
        "seed_rule": "median seed by test mel MSE at the 400-step budget",
        "mp3": {"compression_level": MP3_LEVEL, "sample_rate": SR, "channels": 1},
        "sentences": [
            dict(n=n, id=uid, text=next(c["text"] for c in test.items if c["id"] == uid))
            for n, uid in enumerate(showcase, 1)
        ],
        "clips": clips,
    },
)
