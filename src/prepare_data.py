"""Stage 3: load, align, deduplicate and subsample the three review datasets.

Produces data/processed/{domain}/{train,val,test}.parquet with a shared schema:
    text  (string)
    label (int, 0 = negative, 1 = positive)

Train and validation are drawn disjointly from the original training split, so a review
never appears in both. Duplicate removal runs across every split and every dataset.

Run:  python src/prepare_data.py
"""
import os
import re
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    DOMAINS,
    PROCESSED,
    RESULTS,
    SEED,
    TEST_PER_CLASS,
    TRAIN_PER_CLASS,
    VAL_PER_CLASS,
    ensure_dirs,
    set_seed,
    write_json,
)

BUFFER = 1.05

TEXT_FIELD = {"imdb": "text", "yelp": "text", "amazon": "content"}

SOURCE_NOTE = {
    "imdb": ("stanfordnlp/imdb", "rating >= 7 positive, <= 4 negative, 5-6 dropped"),
    "yelp": (
        "fancyzhx/yelp_polarity",
        "1-2 stars negative, 3-4 stars positive, 5 stars dropped",
    ),
    "amazon": (
        "fancyzhx/amazon_polarity",
        "1-2 stars negative, 4-5 stars positive, 3 stars dropped",
    ),
}


def load_raw(domain):
    from datasets import load_dataset

    if domain == "imdb":
        return load_dataset("stanfordnlp/imdb")
    if domain == "yelp":
        return load_dataset("fancyzhx/yelp_polarity")
    if domain == "amazon":
        return load_dataset("fancyzhx/amazon_polarity")
    raise ValueError(domain)


def sample_train_val(ds, field, train_per_class, val_per_class, rng):
    """Disjoint balanced samples for train and validation from the original train split."""
    labels = np.asarray(ds["train"]["label"])
    n_train = int(np.ceil(train_per_class * BUFFER))
    n_val = int(np.ceil(val_per_class * BUFFER))
    train_idx, val_idx = [], []
    for cls in (0, 1):
        idx = np.where(labels == cls)[0].copy()
        rng.shuffle(idx)
        take_train = min(n_train, len(idx))
        train_idx.append(idx[:take_train])
        rest = idx[take_train:]
        take_val = min(n_val, len(rest))
        val_idx.append(rest[:take_val])
    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    train = ds["train"].select(train_idx.tolist())
    val = ds["train"].select(val_idx.tolist())
    return (
        pd.DataFrame({"text": train[field], "label": train["label"]}),
        pd.DataFrame({"text": val[field], "label": val["label"]}),
    )


def sample_test(ds, field, per_class, rng):
    labels = np.asarray(ds["test"]["label"])
    n = int(np.ceil(per_class * BUFFER))
    picked = []
    for cls in (0, 1):
        idx = np.where(labels == cls)[0]
        picked.append(rng.choice(idx, size=min(n, len(idx)), replace=False))
    sel = np.concatenate(picked)
    rng.shuffle(sel)
    subset = ds["test"].select(sel.tolist())
    return pd.DataFrame({"text": subset[field], "label": subset["label"]})


def normalise(text):
    t = str(text).lower()
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def trim_balanced(df, per_class, seen_hashes):
    """Drop already-seen texts, then trim back to `per_class` rows per class."""
    keep, dropped = [], 0
    for _, row in df.iterrows():
        h = hash(normalise(row["text"]))
        if h in seen_hashes:
            dropped += 1
            continue
        seen_hashes.add(h)
        keep.append(row)
    out = pd.DataFrame(keep)
    if len(out) == 0:
        return out, dropped
    balanced = []
    for cls in (0, 1):
        balanced.append(out[out["label"] == cls].head(per_class))
    out = pd.concat(balanced).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return out, dropped


def describe(df, domain, split):
    lens = df["text"].str.split().str.len()
    return {
        "domain": domain,
        "split": split,
        "rows": int(len(df)),
        "positive_rate": round(float(df["label"].mean()), 4),
        "mean_words": round(float(lens.mean()), 1),
        "median_words": float(lens.median()),
        "p95_words": float(lens.quantile(0.95)),
        "max_words": int(lens.max()),
    }


def main():
    ensure_dirs()
    set_seed()
    rng = np.random.default_rng(SEED)
    seen_hashes = set()
    dedup_report = {}
    stats_rows = []

    for domain in DOMAINS:
        t0 = time.time()
        print(f"[{domain}] loading", flush=True)
        ds = load_raw(domain)
        field = TEXT_FIELD[domain]
        print(f"[{domain}] train={len(ds['train'])} test={len(ds['test'])}", flush=True)

        train_df, val_df = sample_train_val(ds, field, TRAIN_PER_CLASS, VAL_PER_CLASS, rng)
        test_df = sample_test(ds, field, TEST_PER_CLASS, rng)

        domain_dropped = {}
        for split_name, df, per_class in (
            ("train", train_df, TRAIN_PER_CLASS),
            ("val", val_df, VAL_PER_CLASS),
            ("test", test_df, TEST_PER_CLASS),
        ):
            df, dropped = trim_balanced(df, per_class, seen_hashes)
            domain_dropped[split_name] = dropped
            out_dir = os.path.join(PROCESSED, domain)
            os.makedirs(out_dir, exist_ok=True)
            df.to_parquet(os.path.join(out_dir, f"{split_name}.parquet"), index=False)
            print(
                f"[{domain}] {split_name}: {len(df)} rows, "
                f"pos_rate={df['label'].mean():.3f}, dropped {dropped} duplicates",
                flush=True,
            )
            stats_rows.append(describe(df, domain, split_name))

        dedup_report[domain] = domain_dropped
        print(f"[{domain}] done in {time.time() - t0:.1f}s", flush=True)

    write_json(os.path.join(RESULTS, "dedup_report.json"), dedup_report)
    pd.DataFrame(stats_rows).to_csv(os.path.join(RESULTS, "data_stats.csv"), index=False)

    lines = ["# Data preparation report", ""]
    lines.append("Generated by `src/prepare_data.py` (Stage 3).")
    lines.append("")
    lines.append("## Split sizes")
    lines.append("")
    lines.append(
        "| Domain | Split | Rows | Positive rate | Mean words | Median words | P95 words | Max words |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in stats_rows:
        lines.append(
            f"| {row['domain']} | {row['split']} | {row['rows']} | {row['positive_rate']:.2f} | "
            f"{row['mean_words']} | {row['median_words']} | {row['p95_words']} | {row['max_words']} |"
        )
    lines.append("")
    lines.append("## Duplicate removal")
    lines.append("")
    lines.append(
        "Duplicates are detected on normalised text (lowercase, whitespace collapsed) across "
        "all splits and all datasets. The first occurrence wins, so a review that appears in "
        "both IMDb and Amazon is kept in IMDb only. Screening covers exact normalised matches; "
        "near-duplicates with different wording are not removed and are noted as a limitation."
    )
    lines.append("")
    lines.append("| Domain | Train dropped | Val dropped | Test dropped |")
    lines.append("|---|---:|---:|---:|")
    for domain, dropped in dedup_report.items():
        lines.append(
            f"| {domain} | {dropped['train']} | {dropped['val']} | {dropped['test']} |"
        )
    lines.append("")
    lines.append("## Label construction")
    lines.append("")
    lines.append("| Domain | Source | Positive class |")
    lines.append("|---|---|---|")
    for domain in DOMAINS:
        source, positive = SOURCE_NOTE[domain]
        lines.append(f"| {domain} | {source} | {positive} |")
    lines.append("")
    lines.append(
        "The three datasets define the positive class differently. Yelp labels 3-4 stars as "
        "positive and drops 5-star reviews, so its positive class is weaker than IMDb's "
        "(7+/10) and Amazon's (4-5 stars). This is a known confound and is discussed in the report."
    )

    with open(os.path.join(RESULTS, "data_stats.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote results/data_stats.md and results/data_stats.csv")


if __name__ == "__main__":
    main()
