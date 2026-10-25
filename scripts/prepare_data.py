"""Choose the target reader, split their clips, and cache 22.05 kHz audio and log-mels.

Fixed rules, written down before any training:
- Reader: the female train-clean-100 reader with the most retained LibriTTS-R audio
  (sum of durations of clips that have a normalized transcript); ties -> lowest speaker ID.
- Clips: 1.0-10.0 s long (bounds memory/step time and stays under the 1000-step decoder limit).
- Split: sorted clip IDs shuffled with seed 20261012, then 70% train / 10% validation / 20% test.
- Showcase: test clips of 3-7 s, sorted by ID, five evenly spaced picks.
- Other readers' audio is deleted afterwards.
"""

import shutil
import sys

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF


sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from speechopt.common import (  # noqa: E402
    DATA,
    SPEAKER_DIR,
    SR,
    text_to_ids,
    wav_to_mel,
    write_json,
)


RAW = DATA / "raw"
SEED = 20261012

totals = {}
for spk in sorted(RAW.iterdir(), key=lambda p: int(p.name)):
    total = 0.0
    for txt in spk.glob("*.normalized.txt"):
        wav = txt.with_name(txt.name.replace(".normalized.txt", ".wav"))
        if wav.exists():
            info = sf.info(str(wav))
            total += info.frames / info.samplerate
    totals[spk.name] = round(total, 3)
best = max(totals.values())
chosen = min((s for s, t in totals.items() if t == best), key=int)
print(f"{len(totals)} female readers; chosen {chosen} with {best / 60:.2f} min retained")

clips = []
for txt in sorted((RAW / chosen).glob("*.normalized.txt")):
    uid = txt.name.replace(".normalized.txt", "")
    wav_path = RAW / chosen / f"{uid}.wav"
    info = sf.info(str(wav_path))
    dur = info.frames / info.samplerate
    if 1.0 <= dur <= 10.0:
        clips.append({"id": uid, "text": txt.read_text().strip(), "orig_duration": round(dur, 3)})
ids = sorted(c["id"] for c in clips)
perm = np.random.default_rng(SEED).permutation(len(ids))
n_train, n_val = int(round(0.7 * len(ids))), int(round(0.1 * len(ids)))
split_of = {}
for rank, idx in enumerate(perm):
    split_of[ids[idx]] = (
        "train" if rank < n_train else ("val" if rank < n_train + n_val else "test")
    )

(SPEAKER_DIR / "wav22k").mkdir(parents=True, exist_ok=True)
(SPEAKER_DIR / "mel").mkdir(parents=True, exist_ok=True)
for c in clips:
    y, sr = sf.read(str(RAW / chosen / f"{c['id']}.wav"), dtype="float32")
    assert y.ndim == 1
    y22 = AF.resample(torch.from_numpy(y), sr, SR)
    sf.write(str(SPEAKER_DIR / "wav22k" / f"{c['id']}.wav"), y22.numpy(), SR, subtype="PCM_16")
    mel = wav_to_mel(y22)
    torch.save(mel.clone(), SPEAKER_DIR / "mel" / f"{c['id']}.pt")
    c.update(
        split=split_of[c["id"]],
        n_frames=int(mel.shape[1]),
        n_chars=int(len(text_to_ids(c["text"]))),
    )

test_ids = sorted(
    c["id"] for c in clips if c["split"] == "test" and 3.0 <= c["orig_duration"] <= 7.0
)
showcase = [test_ids[int(i)] for i in np.linspace(0, len(test_ids) - 1, 5).round()]
summary = {
    k: {
        "clips": sum(c["split"] == k for c in clips),
        "minutes": round(sum(c["orig_duration"] for c in clips if c["split"] == k) / 60, 2),
    }
    for k in ("train", "val", "test")
}
write_json(
    SPEAKER_DIR / "manifest.json",
    {
        "speaker": chosen,
        "source": "LibriTTS-R train-clean-100 (CC BY 4.0)",
        "rule_seed": SEED,
        "reader_minutes": {s: round(t / 60, 2) for s, t in totals.items()},
        "excluded_by_length": sum(1 for _ in (RAW / chosen).glob("*.normalized.txt")) - len(clips),
        "splits": summary,
        "showcase": showcase,
        "clips": clips,
    },
)
print(summary, "showcase:", showcase)
shutil.copytree(RAW / chosen, SPEAKER_DIR / "raw24k", dirs_exist_ok=True)
shutil.rmtree(RAW)
print("deleted other readers' audio")
