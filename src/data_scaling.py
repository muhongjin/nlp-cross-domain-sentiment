"""Does more in-domain training data shrink the cross-domain gap?

Evaluates DistilBERT checkpoints trained on IMDb at 5,000, 10,000 and 20,000 rows on all
three test sets. Writes results/data_scaling.csv and figures/fig_data_scaling.png.

Run:  python src/data_scaling.py
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DOMAINS, FIGURES, MODELS, RESULTS, ensure_dirs, load_split  # noqa: E402
from transformer_utils import predict_texts  # noqa: E402
from common import compute_metrics  # noqa: E402

DOMAIN_LABEL = {"imdb": "IMDb", "yelp": "Yelp", "amazon": "Amazon"}
RUNS = [
    ("5k", os.path.join(MODELS, "distilbert_scale5k")),
    ("10k", os.path.join(MODELS, "distilbert_scale10k")),
    ("20k", os.path.join(MODELS, "distilbert_imdb")),
]


def main():
    ensure_dirs()
    rows = []
    for label, path in RUNS:
        if not os.path.isdir(path):
            print(f"missing checkpoint {path}, skipping")
            continue
        for target in DOMAINS:
            df = load_split(target, "test")
            preds = predict_texts(path, df["text"].tolist())
            met = compute_metrics(df["label"].tolist(), preds.tolist())
            rows.append(
                {
                    "train_rows": label,
                    "source": "imdb",
                    "target": target,
                    "macro_f1": round(met["macro_f1"], 4),
                }
            )
            print(f"{label} -> {target}: {met['macro_f1']:.4f}")

    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "data_scaling.csv"), index=False)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    order = ["5k", "10k", "20k"]
    for target in DOMAINS:
        sub = df[df.target == target].set_index("train_rows").reindex(order)
        ax.plot(order, sub["macro_f1"], marker="o", label=f"test on {DOMAIN_LABEL[target]}")
    ax.set_xlabel("IMDb training rows")
    ax.set_ylabel("macro-F1")
    ax.set_title("DistilBERT trained on IMDb, tested on each domain")
    ax.set_ylim(0.8, 1.0)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(FIGURES, "fig_data_scaling.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote results/data_scaling.csv and figures/fig_data_scaling.png")


if __name__ == "__main__":
    main()
