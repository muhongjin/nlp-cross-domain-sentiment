"""Stage 10: figures for the report and the video.

Run:  python src/visualize.py
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DOMAINS, FIGURES, RESULTS, ensure_dirs, load_split, read_json  # noqa: E402

DOMAIN_SHORT = {"imdb": "IMDb", "yelp": "Yelp", "amazon": "Amazon"}
MODELS = ["TF-IDF + LR", "TF-IDF + SVM", "DistilBERT"]
MODEL_COLORS = {"TF-IDF + LR": "#4C72B0", "TF-IDF + SVM": "#DD8452", "DistilBERT": "#55A868"}


def save(fig, name):
    path = os.path.join(FIGURES, name)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def fig_within():
    path = os.path.join(RESULTS, "within_domain.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.25
    xs = np.arange(len(DOMAINS))
    for i, model in enumerate(MODELS):
        vals = [df[(df.model == model) & (df.test_domain == d)]["macro_f1"].mean() for d in DOMAINS]
        ax.bar(xs + (i - 1) * width, vals, width, label=model, color=MODEL_COLORS[model])
        for x, v in zip(xs + (i - 1) * width, vals):
            ax.text(x, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(xs)
    ax.set_xticklabels([DOMAIN_SHORT[d] for d in DOMAINS])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("macro-F1")
    ax.set_title("Within-domain performance")
    ax.legend(fontsize=8)
    save(fig, "fig_within_domain.png")


def fig_cross_heatmap():
    path = os.path.join(RESULTS, "cross_domain.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    within = pd.read_csv(os.path.join(RESULTS, "within_domain.csv"))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for idx, (ax, model) in enumerate(zip(axes, MODELS)):
        mat = np.full((3, 3), np.nan)
        for i, s in enumerate(DOMAINS):
            for j, t in enumerate(DOMAINS):
                if s == t:
                    row = within[(within.model == model) & (within.test_domain == t)]
                    if not row.empty:
                        mat[i, j] = row["macro_f1"].iloc[0]
                else:
                    row = df[(df.model == model) & (df.source == s) & (df.target == t)]
                    if not row.empty:
                        mat[i, j] = row["macro_f1"].iloc[0]
        im = ax.imshow(mat, cmap="RdYlGn", vmin=0.5, vmax=1.0)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels([DOMAIN_SHORT[d] for d in DOMAINS])
        ax.set_yticklabels([DOMAIN_SHORT[d] for d in DOMAINS])
        ax.set_xlabel("test domain")
        if idx == 0:
            ax.set_ylabel("training domain")
        ax.set_title(model)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=axes, shrink=0.8, label="macro-F1")
    save(fig, "fig_heatmap_macro_f1.png")

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    drops = np.full((3, 3), np.nan)
    for i, s in enumerate(DOMAINS):
        for j, t in enumerate(DOMAINS):
            if s == t:
                continue
            row = df[(df.model == "DistilBERT") & (df.source == s) & (df.target == t)]
            if not row.empty:
                drops[i, j] = row["drop"].iloc[0]
    im = ax.imshow(drops, cmap="Reds")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels([DOMAIN_SHORT[d] for d in DOMAINS])
    ax.set_yticklabels([DOMAIN_SHORT[d] for d in DOMAINS])
    ax.set_xlabel("test domain")
    ax.set_ylabel("training domain")
    ax.set_title("DistilBERT macro-F1 drop when transferring")
    for i in range(3):
        for j in range(3):
            if not np.isnan(drops[i, j]):
                ax.text(j, i, f"{drops[i, j]:.3f}", ha="center", va="center", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8, label="drop vs within-domain")
    save(fig, "fig_heatmap_drop.png")


def fig_within_vs_cross():
    cross_path = os.path.join(RESULTS, "cross_domain.csv")
    within_path = os.path.join(RESULTS, "within_domain.csv")
    if not (os.path.exists(cross_path) and os.path.exists(within_path)):
        return
    cross = pd.read_csv(cross_path)
    within = pd.read_csv(within_path)
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = np.arange(len(MODELS))
    within_avg = [within[within.model == m]["macro_f1"].mean() for m in MODELS]
    cross_avg = [cross[cross.model == m]["macro_f1"].mean() for m in MODELS]
    width = 0.35
    ax.bar(xs - width / 2, within_avg, width, label="within-domain", color="#4C72B0")
    ax.bar(xs + width / 2, cross_avg, width, label="cross-domain", color="#C44E52")
    for x, a, b in zip(xs, within_avg, cross_avg):
        ax.text(x - width / 2, a + 0.01, f"{a:.3f}", ha="center", fontsize=8)
        ax.text(x + width / 2, b + 0.01, f"{b:.3f}", ha="center", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(MODELS)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("macro-F1")
    ax.set_title("Average performance within and across domains")
    ax.legend(fontsize=8)
    save(fig, "fig_within_vs_cross.png")


def fig_mixed():
    path = os.path.join(RESULTS, "mixed_domain.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, model in zip(axes, MODELS):
        sub = df[df.model == model]
        if sub.empty:
            continue
        xs = np.arange(len(DOMAINS))
        mixed = [sub[sub.target == d]["macro_f1"].mean() for d in DOMAINS]
        single = [sub[sub.target == d]["best_single_source_f1"].mean() for d in DOMAINS]
        ax.bar(xs - 0.2, single, 0.4, label="best single source", color="#8C8C8C")
        ax.bar(xs + 0.2, mixed, 0.4, label="mixed-domain", color="#55A868")
        for x, a, b in zip(xs, single, mixed):
            ax.text(x - 0.2, a + 0.01, f"{a:.3f}", ha="center", fontsize=7)
            ax.text(x + 0.2, b + 0.01, f"{b:.3f}", ha="center", fontsize=7)
        ax.set_xticks(xs)
        ax.set_xticklabels([DOMAIN_SHORT[d] for d in DOMAINS])
        ax.set_title(model)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("macro-F1")
    fig.suptitle("Mixed-domain training against the best single source")
    save(fig, "fig_mixed_vs_single.png")


def fig_confusion():
    path = os.path.join(RESULTS, "cross_metrics.json")
    if not os.path.exists(path):
        return
    metrics = read_json(path)
    keys = [k for k in metrics if k.startswith("distilbert_")]
    if not keys:
        return
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for ax, key in zip(axes.ravel(), sorted(keys)):
        cm = np.array(metrics[key]["confusion_matrix"])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(key.replace("distilbert_", "").replace("_to_", " to "), fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["neg", "pos"])
        ax.set_yticklabels(["neg", "pos"])
        ax.set_xlabel("predicted", fontsize=8)
        ax.set_ylabel("true", fontsize=8)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    fig.suptitle("DistilBERT cross-domain confusion matrices")
    save(fig, "fig_confusion_matrices.png")


def fig_lengths():
    fig, ax = plt.subplots(figsize=(7, 4))
    for domain in DOMAINS:
        df = load_split(domain, "test")
        lens = df["text"].str.split().str.len()
        ax.hist(lens, bins=60, alpha=0.55, label=DOMAIN_SHORT[domain], density=True)
    ax.set_xlabel("words per review")
    ax.set_ylabel("density")
    ax.set_xlim(0, 600)
    ax.set_title("Review length by domain")
    ax.legend()
    save(fig, "fig_length_distribution.png")


def fig_per_class():
    """Positive-class recall for DistilBERT by target domain and source domain."""
    path = os.path.join(RESULTS, "cross_metrics.json")
    if not os.path.exists(path):
        return
    metrics = read_json(path)
    rows = []
    for key, met in metrics.items():
        if not key.startswith("distilbert_"):
            continue
        source, target = key.replace("distilbert_", "").split("_to_")
        rows.append(
            {
                "source": source,
                "target": target,
                "positive_recall": met["per_class"]["positive"]["recall"],
                "negative_recall": met["per_class"]["negative"]["recall"],
            }
        )
    if not rows:
        return
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, column, title in (
        (axes[0], "positive_recall", "Recall on positive reviews"),
        (axes[1], "negative_recall", "Recall on negative reviews"),
    ):
        xs = np.arange(len(DOMAINS))
        width = 0.35
        for i, source in enumerate(["imdb", "yelp", "amazon"]):
            vals = []
            for target in DOMAINS:
                row = df[(df.source == source) & (df.target == target)]
                vals.append(row[column].iloc[0] if not row.empty else np.nan)
            ax.bar(xs + (i - 1) * width, vals, width, label=f"trained on {DOMAIN_SHORT[source]}")
        ax.set_xticks(xs)
        ax.set_xticklabels([DOMAIN_SHORT[d] for d in DOMAINS])
        ax.set_title(title)
        ax.set_ylim(0, 1.15)
        ax.set_xlabel("test domain")
    axes[0].set_ylabel("recall")
    axes[0].legend(fontsize=7, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("DistilBERT recall by class and domain")
    save(fig, "fig_per_class_recall.png")


def main():
    ensure_dirs()
    fig_within()
    fig_cross_heatmap()
    fig_within_vs_cross()
    fig_mixed()
    fig_confusion()
    fig_per_class()
    fig_lengths()


if __name__ == "__main__":
    main()
