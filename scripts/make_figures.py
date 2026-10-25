"""Build the study's SVG figures in the site style (paper card, Archivo, blue data, grey
references).

Every value is read from runs/ (result.json, eval_*.json, ref/eval.json); nothing is typed in.
Usage: python scripts/make_figures.py <commit>
"""

import json
from pathlib import Path
import statistics as st
import sys
from xml.sax.saxutils import escape


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from speechopt.common import ROOT, RUNS  # noqa: E402


COMMIT = sys.argv[1] if len(sys.argv) > 1 else "unknown"
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
CARD, INK, INK2, INK3, RULE, BLUE, ORANGE, GREY = (
    "#F6F2E8",
    "#231F1B",
    "#3D372F",
    "#4F483E",
    "#B3A992",
    "#1F6FB5",
    "#C0601F",
    "#8A8171",
)
OPTS = ["sgd", "adam", "adamw", "rmsprop", "lbfgs"]
NAMES = {
    "sgd": "SGD + momentum",
    "adam": "Adam",
    "adamw": "AdamW",
    "rmsprop": "RMSprop",
    "lbfgs": "L-BFGS",
}
plan = json.loads((RUNS / "plan.json").read_text())
LR = plan["chosen_lr"]
T_BUDGET = plan["time_budget"]
runs = {
    o: [json.loads((RUNS / f"final_{o}_s{s}" / "result.json").read_text()) for s in range(5)]
    for o in OPTS
}
evals = {
    b: {
        o: [json.loads((RUNS / f"final_{o}_s{s}" / f"eval_{b}.json").read_text()) for s in range(5)]
        for o in OPTS
    }
    for b in ("steps", "time")
}
ref = json.loads((RUNS / "ref" / "eval.json").read_text())


class Svg:
    def __init__(self, w: int, h: int, title: str, desc: str) -> None:
        self.w, self.h = w, h
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
            f'aria-labelledby="t d" font-family="Archivo, sans-serif">',
            f'<title id="t">{escape(title)}</title>',
            f'<desc id="d">{escape(desc)}</desc>',
            f'<rect width="{w}" height="{h}" fill="{CARD}"/>',
        ]

    def add(self, s: str) -> None:
        self.parts.append(s)

    def text(
        self,
        x: float,
        y: float,
        s: str,
        size: int = 14,
        fill: str = INK,
        weight: int = 400,
        anchor: str = "start",
    ) -> None:
        self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}">{escape(s)}</text>'
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        stroke: str = RULE,
        width: float = 1,
        dash: str | None = None,
        opacity: float = 1.0,
        tip: str | None = None,
    ) -> None:
        d = f' stroke-dasharray="{dash}"' if dash else ""
        body = (
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" '
            f'stroke-width="{width}"{d} stroke-opacity="{opacity}" stroke-linecap="round"'
        )
        self.add(body + (f"><title>{escape(tip)}</title></line>" if tip else "/>"))

    def poly(
        self,
        pts: list[tuple[float, float]],
        stroke: str = BLUE,
        width: float = 2,
        opacity: float = 1.0,
        tip: str | None = None,
    ) -> None:
        p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        body = (
            f'<polyline points="{p}" fill="none" stroke="{stroke}" stroke-width="{width}" '
            f'stroke-opacity="{opacity}" stroke-linejoin="round" stroke-linecap="round"'
        )
        self.add(body + (f"><title>{escape(tip)}</title></polyline>" if tip else "/>"))

    def dot(
        self,
        x: float,
        y: float,
        r: float = 5,
        fill: str = BLUE,
        tip: str | None = None,
        hollow: bool = False,
        opacity: float = 1.0,
    ) -> None:
        f = (
            f'fill="{CARD}" stroke="{fill}" stroke-width="2"'
            if hollow
            else f'fill="{fill}" fill-opacity="{opacity}" stroke="{CARD}" stroke-width="2"'
        )
        self.add(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" {f}>'
            + (f"<title>{escape(tip)}</title>" if tip else "")
            + "</circle>"
        )

    def save(self, name: str) -> None:
        self.add("</svg>")
        (OUT / name).write_text("\n".join(self.parts) + "\n")
        print("wrote", OUT / name)


def header(svg: Svg, title: str, subtitle: list[str]) -> None:
    svg.text(24, 34, title, size=20, weight=700)
    for i, line in enumerate(subtitle):
        svg.text(24, 56 + 18 * i, line, size=14, fill=INK2)


def learning_curves(
    xkey: str,
    xmax: float,
    xticks: list[float],
    xlabel: str,
    fname: str,
    title: str,
    sub: list[str],
    budget_line: float | None = None,
    mark_budget: str = "steps",
) -> None:
    W, left, right, top, ph, gap = 920, 190, 860, 110, 104, 40
    H = top + len(OPTS) * (ph + gap) + 20
    svg = Svg(
        W,
        H,
        title,
        f"Validation post-net log-mel MSE for each of 5 seeds. "
        f"Dots: checkpoint kept for the {mark_budget} budget. "
        f"Source: runs/final_*/result.json, commit {COMMIT}.",
    )
    header(svg, title, sub)
    ylo, yhi = 0.3, 1.1

    def sx(v: float) -> float:
        return left + (right - left) * v / xmax

    for k, o in enumerate(OPTS):
        y0 = top + k * (ph + gap)

        def sy(v: float) -> float:
            return y0 + ph * (yhi - v) / (yhi - ylo)

        svg.text(24, y0 + 18, NAMES[o], size=15, weight=700)
        svg.text(24, y0 + 38, f"learning rate {LR[o]:g}", size=13, fill=INK3)
        best = [r["best"][mark_budget]["val_mse"] for r in runs[o]]
        svg.text(24, y0 + 58, f"best validation {st.mean(best):.3f}", size=13, fill=INK3)
        svg.text(24, y0 + 76, "(mean of 5 seeds)", size=13, fill=INK3)
        for v in (0.4, 0.6, 0.8, 1.0):
            svg.line(left, sy(v), right, sy(v), dash="3 3")
            svg.text(left - 8, sy(v) + 4, f"{v:.1f}", size=12, fill=INK3, anchor="end")
        svg.line(left, y0 + ph, right, y0 + ph)
        for t in xticks:
            svg.text(sx(t), y0 + ph + 16, f"{t:g}", size=12, fill=INK3, anchor="middle")
        svg.line(left, sy(0.9505), right, sy(0.9505), stroke=GREY, width=1.5, dash="6 4")
        if k == 0:
            svg.text(
                right,
                sy(0.9505) - 6,
                "pretrained model, before fine-tuning (0.951)",
                size=12,
                fill=INK3,
                anchor="end",
            )
        if budget_line is not None:
            svg.line(
                sx(budget_line), y0, sx(budget_line), y0 + ph, stroke=INK, width=1.5, dash="2 3"
            )
            if k == 0:
                svg.text(
                    sx(budget_line) + 6,
                    y0 + 12,
                    f"time budget, {budget_line:.0f} s",
                    size=12,
                    fill=INK,
                )
        for s, r in enumerate(runs[o]):
            pts = [
                (sx(p[xkey]), sy(min(yhi, p["val_mse"])))
                for p in r["curve"]
                if p["val_mse"] is not None
            ]
            svg.poly(pts, opacity=0.55, tip=f"{NAMES[o]}, seed {s}")
        for s, r in enumerate(runs[o]):
            step = r["best"][mark_budget]["step"]
            p = next(p for p in r["curve"] if p["step"] == step)
            svg.dot(
                sx(p[xkey]),
                sy(p["val_mse"]),
                r=5,
                tip=(
                    f"{NAMES[o]}, seed {s}: kept checkpoint at step {step}, "
                    f"validation {p['val_mse']:.3f}"
                ),
            )
    svg.text((left + right) / 2, H - 8, xlabel, size=13, fill=INK3, anchor="middle")
    svg.save(fname)


learning_curves(
    "step",
    400,
    [0, 100, 200, 300, 400],
    "training step",
    "learning-curves-steps.svg",
    "Validation loss by training step",
    [
        "Validation mel loss (lower is better), one line per seed, 5 seeds per optimizer. "
        "Dots mark the",
        "checkpoint kept within the 400-step budget. "
        "All five optimizers start from the same pretrained model.",
    ],
)
learning_curves(
    "train_time",
    1800,
    [0, 300, 600, 900, 1200, 1500, 1800],
    "training time (seconds, validation excluded)",
    "learning-curves-time.svg",
    "Validation loss by training time",
    [
        "Same runs against training time. "
        "Dots mark the checkpoint kept within the 828 s time budget.",
        "L-BFGS takes about twice as long per step, so it gets about 200 steps inside the budget.",
    ],
    budget_line=T_BUDGET,
    mark_budget="time",
)

METRICS = [
    ("mel_mse", "Mel loss (teacher-forced)", 0.0, 1.2, [0, 0.4, 0.8, 1.2], 1, "{:.3f}"),
    ("mcd", "MCD, dB (numpy method)", 3.0, 10.0, [4, 6, 8, 10], 1, "{:.2f}"),
    ("wer", "Word error rate, %", 0.0, 90.0, [0, 30, 60, 90], 100, "{:.1f}"),
    (
        "stop_failures",
        "Failed to stop, of 50 sentences",
        0.0,
        20.0,
        [0, 5, 10, 15, 20],
        1,
        "{:.1f}",
    ),
]
REFS = {
    "mel_mse": [("pretrained", ref["pretrained"]["agg"]["mel_mse"], "6 4")],
    "mcd": [
        ("pretrained", ref["pretrained"]["agg"]["mcd"], "6 4"),
        ("vocoder ceiling", ref["copy_synthesis"]["agg"]["mcd"], "1 3"),
    ],
    "wer": [
        ("pretrained", 100 * ref["pretrained"]["agg"]["wer"], "6 4"),
        ("real recordings", 100 * ref["ground_truth"]["agg"]["wer"], "1 3"),
    ],
    "stop_failures": [("pretrained", 0, "6 4")],
}
W, H = 920, 560
svg = Svg(
    W,
    H,
    "Test-set results by optimizer, 400-step budget",
    "Each dot is one seed; the black tick is the mean over 5 seeds. Grey lines are reference "
    f"systems. Source: runs/final_*/eval_steps.json and runs/ref/eval.json, commit {COMMIT}.",
)
header(
    svg,
    "Test-set results at the 400-step budget",
    [
        "50 held-out sentences. Each dot is one seed, the black tick is the mean. "
        "Lower is better everywhere.",
        "Grey lines: the pretrained model, the real recordings and the vocoder ceiling "
        "(real mels through HiFi-GAN).",
    ],
)
for idx, (key, label, lo, hi, ticks, mult, fmt) in enumerate(METRICS):
    col, row = idx % 2, idx // 2
    x0, y0 = 24 + col * 452, 110 + row * 225
    left, right = x0 + 118, x0 + 380

    def sx(v: float) -> float:
        return left + (right - left) * (v - lo) / (hi - lo)

    svg.text(x0, y0, label, size=15, weight=700)
    rows_y = [y0 + 30 + 28 * i for i in range(len(OPTS))]
    for t in ticks:
        svg.line(sx(t), y0 + 14, sx(t), rows_y[-1] + 14, dash="3 3")
        svg.text(sx(t), rows_y[-1] + 32, f"{t:g}", size=12, fill=INK3, anchor="middle")
    for name, v, dash in REFS[key]:
        assert lo <= v <= hi, f"{key} reference {name}={v} is outside the axis"
        svg.line(
            sx(v),
            y0 + 10,
            sx(v),
            rows_y[-1] + 16,
            stroke=GREY,
            width=1.5,
            dash=dash,
            tip=f"{name}: {v:.3g}",
        )
    lab_y = y0 + 186
    svg.text(
        x0,
        lab_y,
        " · ".join(f"{'dashed' if d == '6 4' else 'dotted'}: {n}" for n, _, d in REFS[key]),
        size=12,
        fill=INK3,
    )
    for i, o in enumerate(OPTS):
        y = rows_y[i]
        vals = [e["agg"][key] * mult for e in evals["steps"][o]]
        svg.text(x0, y + 5, NAMES[o], size=13)
        for s, v in enumerate(vals):
            svg.dot(
                sx(min(hi, v)),
                y,
                r=5,
                opacity=0.75,
                fill=ORANGE if key == "stop_failures" else BLUE,
                tip=f"{NAMES[o]}, seed {s}: {fmt.format(v)}",
            )
        m = st.mean(vals)
        svg.line(
            sx(m),
            y - 9,
            sx(m),
            y + 9,
            stroke=INK,
            width=2.5,
            tip=f"{NAMES[o]} mean: {fmt.format(m)}",
        )
        svg.text(right + 10, y + 5, fmt.format(m), size=13, weight=700)
svg.save("test-results.svg")

W, H = 920, 350
svg = Svg(
    W,
    H,
    "Post-hoc: failed-to-stop count at each saved checkpoint",
    "Post-hoc analysis outside the pre-set rules, using existing checkpoints only. "
    "Each dot is one evaluated checkpoint (50 test sentences); lines join two checkpoints "
    f"from the same run. Source: runs/final_*/eval_*.json, commit {COMMIT}.",
)
header(
    svg,
    "Post-hoc: when do stop failures appear?",
    [
        "Outside the pre-set rules; existing checkpoints only "
        "(two at most per run, so no full trajectories).",
        "Each dot is a saved checkpoint scored on the 50 test sentences; "
        "lines join two checkpoints of one run.",
    ],
)
top, ph, left0, pw, gap = 140, 150, 70, 138, 32
for k, o in enumerate(OPTS):
    left = left0 + k * (pw + gap)
    right = left + pw

    def sx(v: float) -> float:
        return left + pw * v / 400

    def sy(v: float) -> float:
        return top + ph * (1 - v / 20)

    svg.text(left, top - 14, NAMES[o], size=14, weight=700)
    for v in (0, 5, 10, 15, 20):
        svg.line(left, sy(v), right, sy(v), dash="3 3" if v else None)
        if k == 0:
            svg.text(left - 8, sy(v) + 4, f"{v}", size=12, fill=INK3, anchor="end")
    for t in (0, 200, 400):
        svg.text(sx(t), top + ph + 16, f"{t}", size=12, fill=INK3, anchor="middle")
    svg.dot(sx(0), sy(0), r=5, fill=GREY, hollow=True, tip="pretrained model (step 0): 0 failures")
    for s in range(5):
        pts = {}
        for b in ("steps", "time"):
            e = evals[b][o][s]
            pts[e["checkpoint_step"]] = sum(u["stop_failure"] for u in e["per_utt"])
        pts = sorted(pts.items())
        if len(pts) == 2:
            svg.poly([(sx(a), sy(b)) for a, b in pts], stroke=ORANGE, width=1.5, opacity=0.6)
        for step, f in pts:
            svg.dot(
                sx(step),
                sy(f),
                r=5,
                fill=ORANGE,
                opacity=0.8,
                tip=f"{NAMES[o]}, seed {s}, step {step}: {f} of 50 failed to stop",
            )
svg.text(
    left0 + 2.5 * (pw + gap),
    H - 22,
    "training step of the checkpoint (hollow grey dot: pretrained model, 0 failures)",
    size=13,
    fill=INK3,
    anchor="middle",
)
svg.text(24, top - 34, "failed to stop, of 50 test sentences", size=12, fill=INK3)
svg.save("stop-failures-by-checkpoint.svg")
