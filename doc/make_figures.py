"""Figures for the technical document (doc/Leicester_2016.tex).

    python -m possession.validate --idsse all --ens 5 --seed 0 --save data/cache/val_ens5_s0.npz
    python doc/make_figures.py data/cache/val_ens5_s0.npz

Every figure comes from validation numbers or from the held-out Metrica sample games. Nothing is
drawn from BAR-ATH tracking, labels or video.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
OUT = os.path.join(ROOT, "doc", "figures")

C = {"H": "#2b6cb0", "A": "#c53030", "D": "#9a9a9a", "X": "#ffffff", "ink": "#222222",
     "mute": "#777777", "accent": "#2f855a"}
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman",
                     "DejaVu Serif"], "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
COL = 3.33   # one column of the two-column layout, inches
FULL = 6.9


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", os.path.join(OUT, name))


def pipeline():
    fig, ax = plt.subplots(figsize=(FULL, 1.45))
    ax.set_xlim(0, 100); ax.set_ylim(0, 22); ax.axis("off")
    boxes = [
        (0, "Other games\nMetrica 1–3,\nIDSSE ×7 + labels", "#edf2f7"),
        (17, "Degrade to\nATD quality\n(ball loss, smoothing)", "#fefcbf"),
        (34, "Team-level\nfeatures\n(shape, touches, tags)", "#edf2f7"),
        (51, "Boosted trees ×5\nP(dead),\nP(team has ball)", "#e6fffa"),
        (68, "3-state Viterbi\nH / A / D\nswitch penalties", "#e6fffa"),
        (85, "frame,label\nfor BAR–ATH\n(X outside halves)", "#f0fff4"),
    ]
    for x, s, fc in boxes:
        ax.add_patch(FancyBboxPatch((x + 0.8, 5), 13.4, 13, boxstyle="round,pad=0.3,rounding_size=1.2",
                                    fc=fc, ec="#4a5568", lw=0.7))
        ax.text(x + 7.5, 11.5, s, ha="center", va="center", fontsize=6.1, linespacing=1.3)
    for x in (15, 32, 49, 66, 83):
        ax.add_patch(FancyArrowPatch((x - 0.3, 11.5), (x + 2.3, 11.5), arrowstyle="-|>",
                                     mutation_scale=7, lw=0.8, color="#4a5568"))
    ax.text(42, 1.2, "test time: BAR–ATH automatic tracking (ATD) + Smart Tagging enter here",
            ha="center", fontsize=6.4, color=C["mute"], style="italic")
    ax.add_patch(FancyArrowPatch((41.5, 3.0), (41.5, 4.9), arrowstyle="-|>", mutation_scale=6,
                                 lw=0.7, color=C["mute"]))
    ax.text(25, 20.5, "training only", ha="center", fontsize=6.4, color=C["mute"])
    ax.plot([1, 49], [19.6, 19.6], color=C["mute"], lw=0.5)
    save(fig, "pipeline.pdf")


def ablation():
    rows = [  # (label, seed scores) -- possession.validate, see Section 5
        ("state-dependent mask", [0.650]),
        ("+ IDSSE (7 matches)", [0.671, 0.654, 0.667]),
        ("+ ATD-matched ball smoothing", [0.666, 0.651, 0.664]),
        ("+ ball-touch features", [0.6769, 0.6646, 0.6877]),
        ("+ ensemble of 5 (final)", [0.6823, 0.6766, 0.6859]),
    ]
    fig, ax = plt.subplots(figsize=(2.15, 1.75))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, v) in zip(y, rows):
        final = "final" in lab
        ax.barh(yi, np.mean(v) - 0.62, left=0.62, height=0.55,
                color=C["accent"] if final else "#a0aec0")
        ax.scatter(v, [yi] * len(v), s=9, color=C["ink"], zorder=3, lw=0)
        ax.text(np.mean(v) + 0.0035 if len(v) == 1 else max(v) + 0.0035, yi,
                "{:.3f}".format(np.mean(v)), va="center", fontsize=6.8,
                weight="bold" if final else "normal")
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(0.62, 0.705)
    ax.set_xlabel("spellF1, leave-one-game-out (bar = mean, dots = seeds 0/1/2)")
    save(fig, "ablation.pdf")


def smoothing():
    lags = ["±1", "±3", "±6"]
    clean = [(0.4, 16.9), (7.4, 24.9), (12.9, 27.9)]
    atd = [(1.5, 2.9), (4.3, 8.7), (8.3, 16.5)]
    ours = [(1.6, 3.0), (4.6, 8.8), (8.5, 15.8)]
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.45), sharey=False)
    x = np.arange(3)
    for k, (ax, q) in enumerate(zip(axes, ("90th", "99th"))):
        for j, (name, v, col) in enumerate((("clean training ball", clean, "#cbd5e0"),
                                            ("BAR–ATH ATD (no labels)", atd, C["ink"]),
                                            ("training ball, σ = 6 frames", ours, C["accent"]))):
            ax.bar(x + (j - 1) * 0.26, [t[k] for t in v], 0.26, color=col, label=name)
        ax.set_xticks(x, [l + " fr" for l in lags]); ax.set_title(q + " percentile")
    axes[0].set_ylabel("ball velocity change (m/s)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False,
               fontsize=6, handlelength=1, bbox_to_anchor=(0.5, 0.98))
    fig.tight_layout(w_pad=0.6)
    save(fig, "smoothing.pdf")


def timeline(npz, game=2, start_min=20.0, minutes=3.0):
    from possession.model import decode
    z = np.load(npz)
    pD, pH, lab, per = (z["{}_{}".format(k, game)] for k in ("pD", "pH", "labels", "period"))
    a = int(start_min * 60 * 25); b = a + int(minutes * 60 * 25)
    vit = decode(pD, pH, per, switch=20, to_dead=16, from_dead=16)[a:b]
    arg = np.where(pD > 0.5, 2, np.where(pH > 0.5, 0, 1))[a:b]
    gt = lab[a:b]
    t = np.arange(b - a) / 25
    cols = [C["H"], C["A"], C["D"], C["X"]]
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(FULL, 1.75), sharex=True,
                                   gridspec_kw=dict(height_ratios=[1.2, 1]))
    for yi, (name, s) in enumerate((("ground truth", gt), ("frame argmax", arg),
                                    ("Viterbi (submitted)", vit))):
        y = 2 - yi
        ch = np.flatnonzero(np.diff(s)) + 1
        for u, v in zip(np.r_[0, ch], np.r_[ch, len(s)]):
            ax0.barh(y, (v - u) / 25, left=u / 25, height=0.7, color=cols[int(s[u])], lw=0)
        nsp = len(np.r_[0, ch])
        ax0.text(t[-1] + 1.5, y, "{} spells".format(nsp), va="center", fontsize=6.5)
    ax0.set_yticks([2, 1, 0], ["ground truth", "frame argmax", "Viterbi (output)"])
    ax0.spines["left"].set_visible(False); ax0.tick_params(left=False)
    for k, c in (("H", C["H"]), ("A", C["A"]), ("D", C["D"])):
        ax0.barh(-5, 0, color=c, label={"H": "home", "A": "away", "D": "dead"}[k])
    ax0.set_ylim(-0.5, 2.5)
    ax1.plot(t, pH[a:b], color=C["H"], lw=0.6, label="P(home | alive)")
    ax1.plot(t, pD[a:b], color=C["D"], lw=0.6, label="P(dead)")
    ax1.set_ylim(0, 1); ax1.set_yticks([0, 0.5, 1])
    ax1.set_xlabel("seconds (held-out Metrica sample game {}, minute {:.0f}–{:.0f})".format(
        game, start_min, start_min + minutes))
    h0, l0 = ax0.get_legend_handles_labels(); h1, l1 = ax1.get_legend_handles_labels()
    ax0.legend(h0 + h1, l0 + l1, ncol=5, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False)
    ax1.set_xlim(0, t[-1])
    save(fig, "timeline.pdf")


def recall(npz):
    import possession_metrics as pm
    from possession.model import decode
    z = np.load(npz)
    bins = [0, 2, 5, 10, 20, 1e9]
    names = ["<2", "2–5", "5–10", "10–20", ">20"]
    hit = np.zeros(len(names)); tot = np.zeros(len(names))
    for g in (1, 2, 3):
        pD, pH, lab, per = (z["{}_{}".format(k, g)] for k in ("pD", "pH", "labels", "period"))
        pred = decode(pD, pH, per, switch=20, to_dead=16, from_dead=16).astype(np.int64)
        gt = lab.astype(np.int64)
        runs = pm.valid_runs(gt)
        G, P = pm.possession_spells(gt, runs), pm.possession_spells(pred, runs)
        for a, b, k in G:
            best = 0
            for c, d, j in P:
                if j == k and c < b and d > a:
                    best = max(best, (min(b, d) - max(a, c)) / (max(b, d) - min(a, c)))
            i = np.digitize((b - a) / 25, bins) - 1
            tot[i] += 1; hit[i] += best >= 0.5
    fig, ax = plt.subplots(figsize=(COL, 1.45))
    r = hit / np.maximum(tot, 1)
    ax.bar(names, r, color=[C["A"], "#dd6b20", "#a0aec0", C["accent"], C["accent"]], width=0.65)
    for i, (v, n) in enumerate(zip(r, tot)):
        ax.text(i, v + 0.03, "{:.0%}\n(n={:d})".format(v, int(n)), ha="center", fontsize=6.3)
    ax.set_ylim(0, 1.18); ax.set_yticks([0, 0.5, 1], ["0", "50%", "100%"])
    ax.set_xlabel("true possession length (s)")
    ax.set_ylabel("recovered (IoU ≥ 0.5)")
    save(fig, "recall.pdf")
    print("recall by bin", dict(zip(names, np.round(r, 3))), "n", tot.astype(int).tolist())


if __name__ == "__main__":
    pipeline(); ablation(); smoothing()
    if len(sys.argv) > 1:
        timeline(sys.argv[1]); recall(sys.argv[1])
