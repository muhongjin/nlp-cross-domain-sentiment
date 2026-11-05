"""Error rate as a function of review length, for the DistilBERT checkpoints.

The models truncate input at 256 tokens, and IMDb reviews average 229 words, so length is a
natural suspect for the transfer losses. This script re-predicts each setting and groups
accuracy by word count.

Writes results/error_by_length.csv and figures/fig_error_by_length.png.

Run:  python src/error_by_length.py
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DOMAINS, FIGURES, MODELS, RESULTS, compute_metrics, ensure_dirs, load_split  # noqa: E402
from transformer_utils import predict_texts  # noqa: E402

DOMAIN_LABEL = {"imdb": "IMDb", "yelp": "Yelp", "amazon": "Amazon"}
BUCKETS = [(0, 50), (50, 100), (100, 200), (200, 400), (400, 10**9)]
BUCKET_LABELS = ["<=50", "51-100", "101-200", "201-400", ">400"]


def main():
    ensure_dirs()
    rows = []
    for source in DOMAINS:
        model_dir = os.path.join(MODELS, f"distilbert_{source}")
        if not os.path.isdir(model_dir):
            print(f"missing {model_dir}")
            continue
        for target in DOMAINS:
            df = load_split(target, "test")
            preds = predict_texts(model_dir, df["text"].tolist())
            lengths = df["text"].str.split().str.len().to_numpy()
            correct = (np.array(preds) == df["label"].to_numpy())
            for (low, high), label in zip(BUCKETS, BUCKET_LABELS):
                mask = (lengths > low) & (lengths <= high)
                if mask.sum() < 20:
                    continue
                rows.append(
                    {
                        "source": source,
                        "target": target,
                        "setting": "within" if source == target else "cross",
                        "length_bucket": label,
                        "n": int(mask.sum()),
                        "accuracy": round(float(correct[mask].mean()), 4),
                    }
                )
            met = compute_metrics(df["label"].tolist(), list(preds))
            print(f"{source} -> {target}: {met['macro_f1']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "error_by_length.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, setting in zip(axes, ("within", "cross")):
        sub = df[df.setting == setting]
        for target in DOMAINS:
            for source in DOMAINS:
                if setting == "within" and source != target:
                    continue
                if setting == "cross" and source == target:
                    continue
                part = sub[(sub.source == source) & (sub.target == target)]
                if part.empty:
                    continue
                style = "-" if setting == "within" else "--"
                label = f"{DOMAIN_LABEL[source]} to {DOMAIN_LABEL[target]}"
                ax.plot(BUCKET_LABELS, part.set_index("length_bucket").reindex(BUCKET_LABELS)["accuracy"],
                        style, marker="o", label=label, alpha=0.8)
        ax.set_title(f"{setting}-domain settings")
        ax.set_xlabel("review length in words")
        ax.set_ylim(0.5, 1.0)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("accuracy")
    axes[1].legend(fontsize=6, loc="lower left")
    fig.suptitle("DistilBERT accuracy by review length")
    fig.savefig(os.path.join(FIGURES, "fig_error_by_length.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote results/error_by_length.csv and figures/fig_error_by_length.png")


if __name__ == "__main__":
    main()
