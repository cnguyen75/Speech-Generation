"""Paths, pretrained models and feature extraction shared by every script."""

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch


if TYPE_CHECKING:
    from speechbrain.inference.vocoders import HIFIGAN


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SPEAKER_DIR = DATA / "speaker"
RUNS = ROOT / "runs"
PRETRAINED = ROOT / "pretrained"
SR = 22050

# SpeechBrain LJSpeech Tacotron 2 recipe (recipes/LJSpeech/TTS/tacotron2/hparams/train.yaml);
# the released HiFi-GAN vocoder uses the same frame settings.
MEL_KWARGS = dict(
    sample_rate=SR,
    hop_length=256,
    win_length=1024,
    n_fft=1024,
    n_mels=80,
    f_min=0.0,
    f_max=8000.0,
    power=1,
    normalized=False,
    norm="slaney",
    mel_scale="slaney",
    compression=True,
)
TEXT_CLEANERS = ["english_cleaners"]


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def sync(dev: torch.device) -> None:
    if dev.type == "mps":
        torch.mps.synchronize()


def load_tacotron2() -> torch.nn.Module:
    """Return a fresh SpeechBrain Tacotron 2 module (CPU) with pretrained LJSpeech weights."""
    from speechbrain.inference.TTS import Tacotron2

    tts = Tacotron2.from_hparams(
        source="speechbrain/tts-tacotron2-ljspeech",
        savedir=str(PRETRAINED / "tts-tacotron2-ljspeech"),
        run_opts={"device": "cpu"},
    )
    model = tts.mods.model
    model.requires_grad_(True)
    return model


def load_vocoder() -> "HIFIGAN":
    from speechbrain.inference.vocoders import HIFIGAN

    return HIFIGAN.from_hparams(
        source="speechbrain/tts-hifigan-ljspeech",
        savedir=str(PRETRAINED / "tts-hifigan-ljspeech"),
        run_opts={"device": "cpu"},
    )


def text_to_ids(text: str) -> torch.Tensor:
    from speechbrain.utils.text_to_sequence import text_to_sequence

    return torch.tensor(text_to_sequence(text, TEXT_CLEANERS), dtype=torch.long)


def wav_to_mel(wav: torch.Tensor) -> torch.Tensor:
    """wav: 1-D float tensor at 22.05 kHz -> (80, frames) log-mel, as in the recipe."""
    from speechbrain.lobes.models.Tacotron2 import mel_spectrogram

    return mel_spectrogram(audio=wav, **MEL_KWARGS)


def load_manifest() -> dict[str, Any]:
    return json.loads((SPEAKER_DIR / "manifest.json").read_text())


def write_json(path: str | Path, obj: Any) -> None:
    """Atomic JSON write so a crash never leaves a half-written result file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)
