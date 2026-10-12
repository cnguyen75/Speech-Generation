"""One fine-tuning run: an optimizer, a learning rate, a seed and the step/time budgets."""
import math
import os
import time

import numpy as np
import torch
from speechbrain.lobes.models.Tacotron2 import Loss

from .common import SPEAKER_DIR, sync, text_to_ids

BATCH = 16
CLIP_NORM = 1.0
VAL_EVERY = 25
DIVERGE_FACTOR = 10.0
VAL_SEED = 12345

# Final phase of the pretrained model's recipe: guided attention is hard-stopped after epoch 50,
# so the loss is mel MSE (pre- and post-net) + stop-token BCE. epoch=10**6 keeps it switched off.
criterion = Loss(gate_loss_weight=1.0, guided_attention_sigma=0.2, guided_attention_weight=0)


class Clips:
    """In-memory (text ids, log-mel) pairs for one split."""

    def __init__(self, manifest, split):
        self.items = [c for c in manifest["clips"] if c["split"] == split]
        self.items.sort(key=lambda c: c["id"])
        self.text = [text_to_ids(c["text"]) for c in self.items]
        self.mel = [torch.load(SPEAKER_DIR / "mel" / f"{c['id']}.pt") for c in self.items]

    def __len__(self):
        return len(self.items)

    def batch(self, idx, dev):
        idx = sorted(idx, key=lambda i: -len(self.text[i]))  # encoder packs sequences sorted by length
        il = torch.tensor([len(self.text[i]) for i in idx])
        ol = torch.tensor([self.mel[i].shape[1] for i in idx])
        txt = torch.zeros(len(idx), int(il.max()), dtype=torch.long)
        mel = torch.zeros(len(idx), 80, int(ol.max()))
        gate = torch.zeros(len(idx), int(ol.max()))
        for b, i in enumerate(idx):
            txt[b, : il[b]] = self.text[i]
            mel[b, :, : ol[b]] = self.mel[i]
            gate[b, ol[b] - 1:] = 1.0
        return dict(txt=txt.to(dev), il=il.to(dev), mel=mel.to(dev), gate=gate.to(dev), ol=ol.to(dev),
                    max_len=int(ol.max()), ids=[self.items[i]["id"] for i in idx])


def forward_loss(model, b):
    out = model((b["txt"], b["il"], b["mel"], b["max_len"], b["ol"]))
    return out, criterion(out[:4], (b["mel"], b["gate"]), b["il"], b["ol"], 10**6).loss


@torch.no_grad()
def teacher_forced_mse(model, clips, dev, per_utt=False):
    """Masked post-net log-mel MSE (mean over valid frames x 80 bins), fixed prenet-dropout seed."""
    model.eval()
    torch.manual_seed(VAL_SEED)
    sq, cnt, utt = 0.0, 0, {}
    for s in range(0, len(clips), BATCH):
        b = clips.batch(list(range(s, min(s + BATCH, len(clips)))), dev)
        post = model((b["txt"], b["il"], b["mel"], b["max_len"], b["ol"]))[1]
        for k, uid in enumerate(b["ids"]):
            n = int(b["ol"][k])
            e = float(((post[k, :, :n] - b["mel"][k, :, :n]) ** 2).sum())
            sq += e
            cnt += n * 80
            utt[uid] = e / (n * 80)
    model.train()
    return (sq / cnt, utt) if per_utt else sq / cnt


def make_optimizer(name, params, lr):
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr)
    if name == "lbfgs":
        return torch.optim.LBFGS(params, lr=lr, max_iter=1, history_size=10, line_search_fn="strong_wolfe")
    raise ValueError(name)


def batch_order(n, seed, lengths=None, bucket=4):
    """Endless sequence of shuffled epochs; identical for every optimizer with the same seed.

    With `lengths`, each epoch's permutation is cut into groups of bucket*BATCH clips, each group
    is sorted by length and split into batches, and the batches are shuffled. Less padding means
    shorter steps; the decoder's step count is set by the longest clip in a batch.
    """
    rng = np.random.default_rng(seed)
    while True:
        perm = rng.permutation(n)
        if lengths is None:
            batches = [perm[s: s + BATCH].tolist() for s in range(0, n - BATCH + 1, BATCH)]
        else:
            batches = []
            for g in range(0, n, bucket * BATCH):
                group = sorted(perm[g: g + bucket * BATCH].tolist(), key=lambda i: lengths[i])
                batches += [group[s: s + BATCH] for s in range(0, len(group) - BATCH + 1, BATCH)]
            batches = [batches[i] for i in rng.permutation(len(batches))]
        yield from batches


def run(model, pristine, train, val, dev, opt_name, lr, seed, max_steps, time_budget, log):
    """Train from the pretrained weights. Returns (result dict, {budget: best state_dict on CPU}).

    Stops once both `max_steps` steps and `time_budget` seconds of training time are done
    (time_budget None -> steps only), or on divergence. Validation every VAL_EVERY steps, plus
    exactly when each budget closes; validation time is not counted as training time.
    """
    model.load_state_dict(pristine)
    model.to(dev).train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = make_optimizer(opt_name, params, lr)
    order = batch_order(len(train), seed, [m.shape[1] for m in train.mel])
    torch.manual_seed(seed)

    curve, step_times, evals = [], [], 0
    best = {"steps": [math.inf, None, None]} | ({"time": [math.inf, None, None]} if time_budget else {})
    open_ = {k: True for k in best}
    train_time, step, first_loss, window = 0.0, 0, None, []
    status, v0 = "ok", teacher_forced_mse(model, val, dev)
    curve.append(dict(step=0, train_time=0.0, val_mse=v0, train_loss=None, load1=os.getloadavg()[0], evals=0))
    log(f"  step 0 val_mse {v0:.4f}")

    while True:
        b = train.batch(next(order), dev)
        step_seed = seed * 1_000_003 + step
        t0 = time.perf_counter()
        if opt_name == "lbfgs":
            def closure():
                nonlocal evals
                torch.manual_seed(step_seed)  # same dropout masks for every line-search evaluation
                opt.zero_grad(set_to_none=True)
                loss = forward_loss(model, b)[1]
                loss.backward()
                evals += 1
                return loss
            loss = opt.step(closure)
        else:
            torch.manual_seed(step_seed)
            opt.zero_grad(set_to_none=True)
            loss = forward_loss(model, b)[1]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, CLIP_NORM)
            opt.step()
            evals += 1
        loss = float(loss)
        sync(dev)
        dt = time.perf_counter() - t0
        train_time += dt
        step_times.append(dt)
        step += 1
        first_loss = loss if first_loss is None else first_loss
        window.append(loss)
        bad = not math.isfinite(loss) or loss > DIVERGE_FACTOR * first_loss
        if not bad and any(not torch.isfinite(p).all() for p in params[:3] + params[-3:]):
            bad = True
        if bad:
            status = "diverged"

        closing = {"steps": step == max_steps,
                   "time": bool(time_budget) and open_.get("time", False) and train_time >= time_budget}
        if step % VAL_EVERY == 0 or any(closing.values()) or bad:
            v = teacher_forced_mse(model, val, dev) if not bad else math.inf
            v = v if math.isfinite(v) else math.inf
            curve.append(dict(step=step, train_time=round(train_time, 3), val_mse=v if v < math.inf else None,
                              train_loss=float(np.mean(window)), load1=os.getloadavg()[0], evals=evals))
            window = []
            for k in best:
                if open_[k] and v < best[k][0]:
                    best[k] = [v, step, {n: t.detach().to("cpu", copy=True) for n, t in model.state_dict().items()}]
                if closing.get(k):
                    open_[k] = False
            log(f"  step {step} t {train_time:6.1f}s loss {curve[-1]['train_loss']:.4f} val_mse "
                f"{v:.4f} load {curve[-1]['load1']:.1f}" + (" DIVERGED" if bad else ""))
        if bad:
            break
        if step >= max_steps and not (time_budget and train_time < time_budget):
            break

    result = dict(optimizer=opt_name, lr=lr, seed=seed, status=status, steps=step, train_time=train_time,
                  loss_evals=evals, median_step_time=float(np.median(step_times)),
                  mean_load1=float(np.mean([c["load1"] for c in curve])), max_steps=max_steps,
                  time_budget=time_budget, curve=curve,
                  best={k: dict(val_mse=(v[0] if v[0] < math.inf else None), step=v[1]) for k, v in best.items()})
    return result, {k: v[2] for k, v in best.items() if v[2] is not None}
