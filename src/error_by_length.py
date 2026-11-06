"""Error rate as a function of review length, for the DistilBERT checkpoints.

The models truncate input at 256 tokens, and IMDb reviews average 229 words, so length is a
natural suspect for the transfer losses. This script re-predicts each setting and groups
accuracy by word count. Writes results/error_by_length.csv; the figure is drawn by
src/visualize.py.

Run:  python src/error_by_length.py
"""
import os

import numpy as np
import pandas as pd

from common import DOMAINS, MODELS, RESULTS, compute_metrics, ensure_dirs, load_split
from transformer_utils import predict_texts

BUCKETS = [(0, 50), (50, 100), (100, 200), (200, 400), (400, 10**9)]
BUCKET_LABELS = ["<=50", "51-100", "101-200", "201-400", ">400"]

# Buckets with fewer reviews than this are not reported; a handful of reviews would give a
# noisy accuracy estimate.
MIN_BUCKET_SIZE = 20


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
            correct = np.array(preds) == df["label"].to_numpy()
            for (low, high), label in zip(BUCKETS, BUCKET_LABELS):
                mask = (lengths > low) & (lengths <= high)
                if mask.sum() < MIN_BUCKET_SIZE:
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

    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "error_by_length.csv"), index=False)


if __name__ == "__main__":
    main()
