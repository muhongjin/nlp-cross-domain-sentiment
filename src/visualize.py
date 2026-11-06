"""Figures for the report.

Run:  python src/visualize.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

from common import DOMAINS, FIGURES, RESULTS, ensure_dirs, fmt, load_split, read_json

DOMAIN_SHORT = {"imdb": "IMDb", "yelp": "Yelp", "amazon": "Amazon"}
MODELS = ["TF-IDF + LR", "TF-IDF + SVM", "DistilBERT"]
MODEL_COLORS = {"TF-IDF + LR": "#4C72B0", "TF-IDF + SVM": "#DD8452", "DistilBERT": "#55A868"}
SOURCE_COLORS = {"imdb": "#4C72B0", "yelp": "#DD8452", "amazon": "#55A868"}


def save(fig, name):
    path = os.path.join(FIGURES, name)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def annotate(ax, matrix, vmin, vmax, digits=3):
    """Write the value into each cell, switching to white on dark fills."""
    norm = Normalize(vmin=vmin, vmax=vmax)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isnan(value):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=9, color="#555555")
                continue
            color = "white" if norm(value) > 0.6 else "black"
            ax.text(j, i, fmt(value, digits), ha="center", va="center", fontsize=9, color=color)


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
            ax.text(x, v + 0.01, fmt(v), ha="center", fontsize=7)
    ax.set_xticks(xs)
    ax.set_xticklabels([DOMAIN_SHORT[d] for d in DOMAINS])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("macro-F1")
    ax.set_title("Within-domain performance")
    ax.legend(fontsize=8, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    save(fig, "fig_within_domain.png")


def cross_matrix(cross, within, model):
    mat = np.full((3, 3), np.nan)
    for i, source in enumerate(DOMAINS):
        for j, target in enumerate(DOMAINS):
            if source == target:
                row = within[(within.model == model) & (within.test_domain == target)]
            else:
                row = cross[(cross.model == model) & (cross.source == source) & (cross.target == target)]
            if not row.empty:
                mat[i, j] = row["macro_f1"].iloc[0]
    return mat


def fig_cross_heatmap():
    cross_path = os.path.join(RESULTS, "cross_domain.csv")
    if not os.path.exists(cross_path):
        return
    cross = pd.read_csv(cross_path)
    within = pd.read_csv(os.path.join(RESULTS, "within_domain.csv"))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    for idx, (ax, model) in enumerate(zip(axes, MODELS)):
        mat = cross_matrix(cross, within, model)
        im = ax.imshow(mat, cmap="RdYlGn", vmin=0.5, vmax=1.0)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels([DOMAIN_SHORT[d] for d in DOMAINS])
        ax.set_yticklabels([DOMAIN_SHORT[d] for d in DOMAINS])
        ax.set_xlabel("test domain")
        if idx == 0:
            ax.set_ylabel("training domain")
        ax.set_title(model)
        annotate(ax, mat, 0.5, 1.0)
    fig.colorbar(im, ax=axes, shrink=0.8, label="macro-F1")
    save(fig, "fig_heatmap_macro_f1.png")

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    drops = np.full((3, 3), np.nan)
    for i, source in enumerate(DOMAINS):
        for j, target in enumerate(DOMAINS):
            if source == target:
                continue
            row = cross[
                (cross.model == "DistilBERT") & (cross.source == source) & (cross.target == target)
            ]
            if not row.empty:
                drops[i, j] = row["drop"].iloc[0]
    cmap = plt.get_cmap("Reds").copy()
    cmap.set_bad("#E8E8E8")
    im = ax.imshow(np.ma.masked_invalid(drops), cmap=cmap)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels([DOMAIN_SHORT[d] for d in DOMAINS])
    ax.set_yticklabels([DOMAIN_SHORT[d] for d in DOMAINS])
    ax.set_xlabel("test domain")
    ax.set_ylabel("training domain")
    ax.set_title("DistilBERT macro-F1 drop when transferring")
    annotate(ax, drops, float(np.nanmin(drops)), float(np.nanmax(drops)))
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
        ax.text(x - width / 2, a + 0.01, fmt(a), ha="center", fontsize=8)
        ax.text(x + width / 2, b + 0.01, fmt(b), ha="center", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(MODELS)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("macro-F1")
    ax.set_title("Average performance within and across domains")
    ax.legend(fontsize=8, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    save(fig, "fig_within_vs_cross.png")


def fig_mixed():
    path = os.path.join(RESULTS, "mixed_domain.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True, constrained_layout=True)
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
            ax.text(x - 0.2, a + 0.01, fmt(a), ha="center", fontsize=7)
            ax.text(x + 0.2, b + 0.01, fmt(b), ha="center", fontsize=7)
        ax.set_xticks(xs)
        ax.set_xticklabels([DOMAIN_SHORT[d] for d in DOMAINS])
        ax.set_title(model)
        ax.set_ylim(0, 1.18)
        ax.legend(fontsize=7, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    axes[0].set_ylabel("macro-F1")
    save(fig, "fig_mixed_vs_single.png")


def fig_confusion():
    path = os.path.join(RESULTS, "cross_metrics.json")
    if not os.path.exists(path):
        return
    metrics = read_json(path)
    keys = sorted(k for k in metrics if k.startswith("distilbert_"))
    if not keys:
        return
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.6), constrained_layout=True)
    for ax, key in zip(axes.ravel(), keys):
        cm = np.array(metrics[key]["confusion_matrix"])
        im = ax.imshow(cm, cmap="Blues")
        title = key.replace("distilbert_", "").replace("_to_", " -> ")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["neg", "pos"], fontsize=8)
        ax.set_yticklabels(["neg", "pos"], fontsize=8)
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=9)
    fig.supxlabel("predicted", fontsize=9)
    fig.supylabel("true", fontsize=9)
    fig.colorbar(im, ax=axes, shrink=0.7, label="reviews")
    save(fig, "fig_confusion_matrices.png")


def fig_per_class():
    """Recall by class for each DistilBERT transfer, grouped by target domain.

    Each target is only scored by the two other source domains, so each group has two bars.
    """
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
    sources = ["imdb", "yelp", "amazon"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, constrained_layout=True)
    for ax, column, title in (
        (axes[0], "positive_recall", "Recall on positive reviews"),
        (axes[1], "negative_recall", "Recall on negative reviews"),
    ):
        width = 0.34
        for target_idx, target in enumerate(DOMAINS):
            others = [s for s in sources if s != target]
            for k, source in enumerate(others):
                row = df[(df.source == source) & (df.target == target)]
                if row.empty:
                    continue
                ax.bar(
                    target_idx + (k - 0.5) * width,
                    row[column].iloc[0],
                    width,
                    color=SOURCE_COLORS[source],
                    label=f"trained on {DOMAIN_SHORT[source]}",
                )
        ax.set_xticks(range(len(DOMAINS)))
        ax.set_xticklabels([DOMAIN_SHORT[d] for d in DOMAINS])
        ax.set_title(title)
        ax.set_ylim(0, 1.12)
        ax.set_xlabel("test domain")
    axes[0].set_ylabel("recall")
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[0].legend(
        unique.values(), unique.keys(), fontsize=7, loc="upper center", ncol=3,
        frameon=False, bbox_to_anchor=(0.5, 1.0),
    )
    save(fig, "fig_per_class_recall.png")


def fig_domain_distance():
    path = os.path.join(RESULTS, "domain_distance.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    mat = np.zeros((3, 3))
    for _, row in df.iterrows():
        i = DOMAINS.index(row["domain_a"])
        j = DOMAINS.index(row["domain_b"])
        mat[i, j] = mat[j, i] = row["js_divergence"]
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels([DOMAIN_SHORT[d] for d in DOMAINS])
    ax.set_yticklabels([DOMAIN_SHORT[d] for d in DOMAINS])
    ax.set_title("Jensen-Shannon divergence between domains")
    annotate(ax, mat, float(mat.min()), float(mat.max()))
    fig.colorbar(im, ax=ax, shrink=0.8, label="JSD (bits)")
    save(fig, "fig_domain_distance.png")


def fig_data_scaling():
    path = os.path.join(RESULTS, "data_scaling.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    order = ["5k", "10k", "20k"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for target in DOMAINS:
        sub = df[df.target == target].set_index("train_rows").reindex(order)
        if sub["macro_f1"].isna().all():
            continue
        ax.plot(order, sub["macro_f1"], marker="o", label=f"test on {DOMAIN_SHORT[target]}")
    ax.set_xlabel("IMDb training rows")
    ax.set_ylabel("macro-F1")
    ax.set_title("DistilBERT trained on IMDb, tested on each domain")
    ax.set_ylim(0.8, 1.0)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save(fig, "fig_data_scaling.png")


def fig_error_by_length():
    path = os.path.join(RESULTS, "error_by_length.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    order = ["<=50", "51-100", "101-200", "201-400", ">400"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True, constrained_layout=True)
    for ax, setting in zip(axes, ("within", "cross")):
        sub = df[df.setting == setting]
        for target in DOMAINS:
            for source in DOMAINS:
                if (source == target) != (setting == "within"):
                    continue
                part = sub[(sub.source == source) & (sub.target == target)]
                if part.empty:
                    continue
                series = part.set_index("length_bucket").reindex(order)["accuracy"]
                ax.plot(order, series, marker="o", alpha=0.8,
                        label=f"{DOMAIN_SHORT[source]} to {DOMAIN_SHORT[target]}")
        ax.set_title(f"{setting}-domain settings")
        ax.set_xlabel("review length in words")
        ax.set_ylim(0.5, 1.0)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("accuracy")
    axes[1].legend(fontsize=6, loc="lower left")
    save(fig, "fig_error_by_length.png")


def fig_lengths():
    """Needs the processed splits; skipped when they are not available."""
    try:
        frames = {d: load_split(d, "test") for d in DOMAINS}
    except Exception:
        print("skipping fig_length_distribution: processed splits not found")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for domain, df in frames.items():
        lens = df["text"].str.split().str.len()
        ax.hist(lens, bins=60, alpha=0.55, label=DOMAIN_SHORT[domain], density=True)
    ax.set_xlabel("words per review")
    ax.set_ylabel("density")
    ax.set_xlim(0, 600)
    ax.set_title("Review length by domain")
    ax.legend()
    save(fig, "fig_length_distribution.png")


def main():
    ensure_dirs()
    fig_within()
    fig_cross_heatmap()
    fig_within_vs_cross()
    fig_mixed()
    fig_confusion()
    fig_per_class()
    fig_domain_distance()
    fig_data_scaling()
    fig_error_by_length()
    fig_lengths()


if __name__ == "__main__":
    main()
