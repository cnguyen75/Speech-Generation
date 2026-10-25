"""Score a Tacotron 2 checkpoint on a split: teacher-forced mel MSE, then free-running synthesis
through the frozen HiFi-GAN, scored by Whisper word error rate and MCD against the real recording."""

import jiwer
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

from .common import SPEAKER_DIR, SR, text_to_ids
from .mcd import mcd
from .train import teacher_forced_mse


ASR_MODEL = "openai/whisper-small.en"
SYNTH_SEED = 777  # per-sentence seed = SYNTH_SEED + index, identical for every checkpoint
MAX_DECODER_STEPS = 1000


class ASR:
    def __init__(self, dev):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.proc = WhisperProcessor.from_pretrained(ASR_MODEL)
        self.model = (
            WhisperForConditionalGeneration.from_pretrained(ASR_MODEL, use_safetensors=True)
            .to(dev)
            .eval()
        )
        self.dev = dev

    def norm(self, text):
        return self.proc.tokenizer.normalize(text)

    @torch.no_grad()
    def transcribe(self, wavs_22k, batch=8):
        out = []
        for s in range(0, len(wavs_22k), batch):
            chunk = [
                AF.resample(torch.as_tensor(w, dtype=torch.float32), SR, 16000).numpy()
                for w in wavs_22k[s : s + batch]
            ]
            feats = self.proc(chunk, sampling_rate=16000, return_tensors="pt").input_features.to(
                self.dev
            )
            ids = self.model.generate(feats, num_beams=1, do_sample=False, max_new_tokens=200)
            out += self.proc.batch_decode(ids, skip_special_tokens=True)
        return out


def word_errors(asr, refs, hyps):
    rows = []
    for r, h in zip(refs, hyps):
        rn, hn = asr.norm(r), asr.norm(h)
        o = jiwer.process_words(rn if rn.strip() else "<empty>", hn if hn.strip() else "")
        rows.append(
            (o.substitutions + o.deletions + o.insertions, o.substitutions + o.deletions + o.hits)
        )
    return rows


def real_wav(uid):
    return sf.read(str(SPEAKER_DIR / "wav22k" / f"{uid}.wav"), dtype="float32")[0]


@torch.no_grad()
def synthesize(cpu_model, vocoder, texts):
    cpu_model.eval()
    wavs, failed = [], []
    for i, t in enumerate(texts):
        torch.manual_seed(SYNTH_SEED + i)
        ids = text_to_ids(t)[None]
        mel, mel_len, _ = cpu_model.infer(ids, torch.tensor([ids.shape[1]]))
        failed.append(int(mel_len[0]) >= MAX_DECODER_STEPS)
        wavs.append(vocoder.decode_batch(mel).squeeze().numpy())
    return wavs, failed


def score(utts, wavs, failed, asr, extra=None):
    refs_real = [real_wav(u["id"]) for u in utts]
    hyps = asr.transcribe(wavs)
    errs = word_errors(asr, [u["text"] for u in utts], hyps)
    per_utt = []
    for k, u in enumerate(utts):
        per_utt.append(
            dict(
                id=u["id"],
                hyp=hyps[k],
                word_errors=errs[k][0],
                ref_words=errs[k][1],
                mcd=mcd(refs_real[k], wavs[k]),
                stop_failure=bool(failed[k]),
                **({} if extra is None else {"mel_mse": extra[u["id"]]}),
            )
        )
    agg = dict(
        n=len(utts),
        wer=sum(r["word_errors"] for r in per_utt) / max(1, sum(r["ref_words"] for r in per_utt)),
        mcd=float(np.mean([r["mcd"] for r in per_utt])),
        stop_failures=int(sum(failed)),
    )
    if extra is not None:
        agg["mel_mse"] = float(np.mean(list(extra.values())))
    return agg, per_utt


def evaluate_checkpoint(
    state, train_model, cpu_model, vocoder, asr, clips, dev, showcase_dir=None, showcase=()
):
    """state: Tacotron 2 state_dict. Returns (aggregate, per-utterance) on `clips`."""
    train_model.load_state_dict(state)
    train_model.to(dev)
    tf_total, tf_utt = teacher_forced_mse(train_model, clips, dev, per_utt=True)
    cpu_model.load_state_dict(state)
    wavs, failed = synthesize(cpu_model, vocoder, [u["text"] for u in clips.items])
    agg, per_utt = score(clips.items, wavs, failed, asr, extra=tf_utt)
    agg["mel_mse_pooled"] = tf_total
    if showcase_dir is not None:
        showcase_dir.mkdir(parents=True, exist_ok=True)
        for u, w in zip(clips.items, wavs):
            if u["id"] in showcase:
                sf.write(str(showcase_dir / f"{u['id']}.wav"), w, SR, subtype="PCM_16")
    return agg, per_utt


@torch.no_grad()
def evaluate_references(vocoder, asr, clips, showcase_dir=None, showcase=()):
    """Real recordings (ASR floor) and copy-synthesis (real mel -> HiFi-GAN; vocoder ceiling)."""
    real = [real_wav(u["id"]) for u in clips.items]
    gt_agg, gt_utt = score(clips.items, real, [False] * len(real), asr)
    copy = [vocoder.decode_batch(m[None]).squeeze().numpy() for m in clips.mel]
    cs_agg, cs_utt = score(clips.items, copy, [False] * len(copy), asr)
    if showcase_dir is not None:
        for name, ws in (("ground_truth", real), ("copy_synthesis", copy)):
            (showcase_dir / name).mkdir(parents=True, exist_ok=True)
            for u, w in zip(clips.items, ws):
                if u["id"] in showcase:
                    sf.write(str(showcase_dir / name / f"{u['id']}.wav"), w, SR, subtype="PCM_16")
    return {"ground_truth": (gt_agg, gt_utt), "copy_synthesis": (cs_agg, cs_utt)}
