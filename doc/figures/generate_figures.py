"""
Research report figure generator.
Outputs 7 PNG figures into doc/figures/ for embedding into the report.

Run:
    python doc/generate_figures.py
"""
from __future__ import annotations

import os
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ---- font setup (CJK-friendly) ---------------------------------------------
plt.rcParams["font.family"] = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "doc", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

TRAIN_CSV = os.path.join(ROOT, "data", "train_2022.csv")
TEST_CSV = os.path.join(ROOT, "data", "test_no_answer_2022.csv")


def save(fig, name: str) -> None:
    out = os.path.join(FIG_DIR, name)
    fig.savefig(out, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"[saved] {out}")


# ---------------------------------------------------------------------------
# Figure 1 : training-set class distribution
# ---------------------------------------------------------------------------
def fig1_class_distribution() -> None:
    df = pd.read_csv(TRAIN_CSV)
    counts = df["LABEL"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Negative (0)", "Positive (1)"], counts.values,
                  color=["#d9534f", "#5cb85c"], edgecolor="black")
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 10, f"{v}",
                ha="center", va="bottom", fontsize=11)
    total = counts.sum()
    ax.set_title(f"Figure 1. Training set class distribution (N={total})")
    ax.set_ylabel("Number of samples")
    ax.set_ylim(0, max(counts.values) * 1.15)
    save(fig, "fig1_class_distribution.png")


# ---------------------------------------------------------------------------
# Figure 2 : text length histograms (word count)
# ---------------------------------------------------------------------------
def fig2_text_length() -> None:
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    train_len = train["TEXT"].fillna("").str.split().str.len()
    test_len = test["TEXT"].fillna("").str.split().str.len()
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.arange(0, max(train_len.max(), test_len.max()) + 2, 2)
    ax.hist(train_len, bins=bins, alpha=0.6, label=f"Train (n={len(train_len)})",
            color="#337ab7", edgecolor="black")
    ax.hist(test_len, bins=bins, alpha=0.5, label=f"Test (n={len(test_len)})",
            color="#f0ad4e", edgecolor="black")
    ax.axvline(train_len.mean(), color="#337ab7", linestyle="--", linewidth=1,
               label=f"Train mean={train_len.mean():.1f}")
    ax.axvline(test_len.mean(), color="#f0ad4e", linestyle="--", linewidth=1,
               label=f"Test mean={test_len.mean():.1f}")
    ax.set_title("Figure 2. Text length distribution (word count)")
    ax.set_xlabel("Number of words")
    ax.set_ylabel("Frequency")
    ax.legend()
    save(fig, "fig2_text_length.png")


# ---------------------------------------------------------------------------
# Figure 3 : top-30 frequent words and polarity ratio
# ---------------------------------------------------------------------------
STOP = {"the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "but",
        "is", "it", "its", "this", "that", "with", "as", "at", "by", "be",
        "are", "was", "were", "from", "has", "have", "had", "s", "t", "you",
        "your", "they", "we", "i", "he", "she", "his", "her", "him", "their",
        "them", "than", "then", "so", "if", "no", "not", "do", "does", "did"}

WORD_RE = re.compile(r"[A-Za-z]+")


def fig3_top_words() -> None:
    df = pd.read_csv(TRAIN_CSV)
    pos_counter, neg_counter = Counter(), Counter()
    for text, label in zip(df["TEXT"].fillna(""), df["LABEL"]):
        words = [w.lower() for w in WORD_RE.findall(text) if w.lower() not in STOP and len(w) > 2]
        if label == 1:
            pos_counter.update(words)
        else:
            neg_counter.update(words)
    total_counter = pos_counter + neg_counter
    top30 = [w for w, _ in total_counter.most_common(30)]

    pos_ratio = []
    freq = []
    for w in top30:
        p, n = pos_counter[w], neg_counter[w]
        pos_ratio.append(p / (p + n))
        freq.append(p + n)

    order = np.argsort(pos_ratio)
    top30 = [top30[i] for i in order]
    pos_ratio = [pos_ratio[i] for i in order]
    freq = [freq[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ["#5cb85c" if r > 0.55 else "#d9534f" if r < 0.45 else "#bdbdbd"
              for r in pos_ratio]
    ax.barh(top30, pos_ratio, color=colors, edgecolor="black")
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="Neutral (0.5)")
    for i, (r, f) in enumerate(zip(pos_ratio, freq)):
        ax.text(r + 0.01, i, f"n={f}", va="center", fontsize=8)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Ratio appearing in positive class")
    ax.set_title("Figure 3. Top 30 frequent words and their positive-class ratio")
    ax.legend(loc="lower right")
    save(fig, "fig3_top_words.png")


# ---------------------------------------------------------------------------
# Figure 4 : v1 -> v6.6 methodology pipeline
# ---------------------------------------------------------------------------
def _box(ax, x, y, w, h, text, color):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.04",
                         linewidth=1.2, edgecolor="black", facecolor=color)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=9, wrap=True)


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=14, linewidth=1.0, color="black"))


def fig4_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    boxes = [
        ("v1\nTF-IDF (1-2g)\n+ LogReg\nF1=0.6622", "#fce4ec"),
        ("v2\nWord+Char TF-IDF\n+ LogReg\nF1=0.6697", "#f8bbd0"),
        ("v3\nDistilBERT\nfine-tune\nF1=0.8196", "#bbdefb"),
        ("v4\nRoBERTa-base\n5-Fold + Soft Vote\nF1=0.8348", "#90caf9"),
        ("v5\nRoBERTa-large\n+ Pseudo-label\nF1=0.8715", "#64b5f6"),
        ("v6.1\nTwo-Stage\nSelf-Training\nF1=0.8694", "#a5d6a7"),
        ("v6.6 ★\n3-Model Ensemble\nRo+De+EL\nF1=0.8864", "#66bb6a"),
    ]
    n = len(boxes)
    bw, bh = 1.55, 1.7
    gap = (13 - n * bw) / (n + 1)
    y = 1.4
    centers = []
    for i, (text, color) in enumerate(boxes):
        x = gap + i * (bw + gap)
        _box(ax, x, y, bw, bh, text, color)
        centers.append(x + bw / 2)
    for i in range(n - 1):
        _arrow(ax, centers[i] + bw / 2 - 0.05, y + bh / 2,
               centers[i + 1] - bw / 2 + 0.05, y + bh / 2)

    ax.text(6.5, 3.8, "Figure 4. Methodology evolution: TF-IDF -> Transformer -> Ensemble",
            ha="center", fontsize=12, fontweight="bold")
    ax.text(6.5, 0.6, "Bag-of-Words (Public ~0.66)              Single Transformer (Public 0.70 -> 0.79)              Multi-arch Ensemble (Public 0.80)",
            ha="center", fontsize=9, color="#444")

    save(fig, "fig4_pipeline.png")


# ---------------------------------------------------------------------------
# Figure 5 : v6.6 three-model ensemble architecture
# ---------------------------------------------------------------------------
def fig5_ensemble_arch() -> None:
    fig, ax = plt.subplots(figsize=(11, 6.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.6)
    ax.axis("off")

    # input
    _box(ax, 0.3, 2.5, 1.6, 1.0, "Test text\n(11,000 samples)", "#eeeeee")

    # three models
    _box(ax, 3.0, 4.4, 2.6, 1.0,
         "RoBERTa-large\n(MLM)\nOOF F1=0.8694", "#90caf9")
    _box(ax, 3.0, 2.5, 2.6, 1.0,
         "DeBERTa-v3-large\n(Disentangled MLM)\nOOF F1=0.8797", "#a5d6a7")
    _box(ax, 3.0, 0.6, 2.6, 1.0,
         "ELECTRA-large\n(RTD)\nOOF F1=0.8481", "#ffcc80")

    # arrows from input
    for y in (4.9, 3.0, 1.1):
        _arrow(ax, 1.9, 3.0, 3.0, y)

    # average box
    _box(ax, 6.6, 2.4, 2.0, 1.2, "Equal-weight\nprob average\n(1/3 each)", "#fff59d")

    # arrows to average
    for y in (4.9, 3.0, 1.1):
        _arrow(ax, 5.6, y, 6.6, 3.0)

    # threshold + submission
    _box(ax, 9.1, 2.4, 1.7, 1.2, "Threshold\ntau* = 0.610\n(OOF tuned)", "#ce93d8")
    _arrow(ax, 8.6, 3.0, 9.1, 3.0)

    # final
    _box(ax, 9.1, 0.7, 1.7, 1.0, "Submission\nv6.6\n0.79807 ★", "#80cbc4")
    _arrow(ax, 9.95, 2.4, 9.95, 1.7)

    ax.text(5.5, 6.3,
            "Figure 5. v6.6 Three-Model Ensemble Architecture",
            ha="center", fontsize=12, fontweight="bold")
    ax.text(5.5, 5.95,
            "Three diverse pretraining objectives -> probability averaging -> OOF-calibrated threshold",
            ha="center", fontsize=9, color="#444")

    save(fig, "fig5_ensemble_arch.png")


# ---------------------------------------------------------------------------
# Figure 6 : Public score progression line chart
# ---------------------------------------------------------------------------
def fig6_public_score_progression() -> None:
    versions = ["v2", "v3", "v4", "v5", "v6", "v6.1", "v6.4", "v6.5", "v6.6"]
    public = [0.66308, 0.70137, 0.74683, 0.74710, 0.78099, 0.79173, 0.78484, 0.78870, 0.79807]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(versions, public, "-o", color="#1976d2", linewidth=2, markersize=7)
    for i, (v, p) in enumerate(zip(versions, public)):
        offset = -0.014 if v == "v6.4" else 0.006
        va = "top" if v == "v6.4" else "bottom"
        ax.annotate(f"{p:.5f}", xy=(i, p), xytext=(0, 8 if va == "bottom" else -8),
                    textcoords="offset points", ha="center", va=va, fontsize=8.5)
    # highlight best
    best_i = public.index(max(public))
    ax.scatter([best_i], [public[best_i]], color="red", s=150, zorder=3, label="Best (v6.6)")
    # mark v6.4 regression
    v64_i = versions.index("v6.4")
    ax.scatter([v64_i], [public[v64_i]], color="orange", s=120, zorder=3,
               label="Circular self-training trap (v6.4)")
    ax.set_ylim(0.65, 0.82)
    ax.set_ylabel("Public Score")
    ax.set_title("Figure 6. Public-leaderboard progression across versions")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    save(fig, "fig6_public_progression.png")


# ---------------------------------------------------------------------------
# Figure 7 : OOF F1 vs Public Score scatter
# ---------------------------------------------------------------------------
def fig7_oof_vs_public() -> None:
    data = [
        ("v2", 0.6697, 0.66308),
        ("v3", 0.8196, 0.70137),
        ("v4", 0.8348, 0.74683),
        ("v5", 0.8715, 0.74710),
        ("v6", 0.8546, 0.78099),
        ("v6.1", 0.8694, 0.79173),
        ("v6.4", 0.8823, 0.78484),
        ("v6.5", 0.8797, 0.78870),
        ("v6.6", 0.8864, 0.79807),
    ]
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, oof, pub in data:
        if name == "v6.6":
            color, size = "#d32f2f", 200
        elif name == "v6.4":
            color, size = "#f57c00", 150
        else:
            color, size = "#1976d2", 90
        ax.scatter(oof, pub, c=color, s=size, edgecolor="black", zorder=3)
        dx, dy = 0.003, 0.002
        ax.annotate(name, (oof, pub), xytext=(oof + dx, pub + dy), fontsize=9)

    # diagonal reference (perfect alignment shifted)
    xs = np.linspace(0.65, 0.92, 10)
    ax.plot(xs, xs - 0.08, "--", color="grey", alpha=0.6,
            label="Constant gap = 0.08")
    ax.set_xlabel("OOF F1 (cross-validated)")
    ax.set_ylabel("Public Score")
    ax.set_xlim(0.65, 0.92)
    ax.set_ylim(0.64, 0.82)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")

    # annotate v6.4 bad gap
    ax.annotate("OOF up but Public down\n(circular self-training)",
                xy=(0.8823, 0.78484), xytext=(0.84, 0.71),
                fontsize=8, color="#b34700",
                arrowprops=dict(arrowstyle="->", color="#b34700"))
    save(fig, "fig7_oof_vs_public.png")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Output directory: {FIG_DIR}")
    fig1_class_distribution()
    fig2_text_length()
    fig3_top_words()
    fig4_pipeline()
    fig5_ensemble_arch()
    fig6_public_score_progression()
    fig7_oof_vs_public()
    print("All figures generated.")


if __name__ == "__main__":
    main()
