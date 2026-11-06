"""Does more in-domain training data shrink the cross-domain gap?

Evaluates DistilBERT checkpoints trained on IMDb at 5,000, 10,000 and 20,000 rows on all
three test sets. Writes results/data_scaling.csv. The figure is drawn by src/visualize.py.

Run:  python src/data_scaling.py
"""
import os

import pandas as pd

from common import DOMAINS, MODELS, RESULTS, compute_metrics, ensure_dirs, load_split
from transformer_utils import predict_texts

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
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "data_scaling.csv"), index=False)


if __name__ == "__main__":
    main()
