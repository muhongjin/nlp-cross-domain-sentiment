"""Hyperparameter selection on validation data only.

TF-IDF models: small grid over C and n-gram range, scored on each domain's validation split.
DistilBERT: the learning rate is chosen from results/transformer/tune_*.json, which are
produced by running train_transformer.py with --tag tune_lr<value> on IMDb. The winner is
applied to every domain.

Writes configs/final.yaml.

Run:  python src/tune.py
"""
import glob
import os

import pandas as pd
import yaml

from baselines import build_model
from common import (
    CONFIGS,
    DOMAINS,
    RESULTS,
    SEED,
    compute_metrics,
    ensure_dirs,
    load_split,
    read_json,
    set_seed,
)

TFIDF_GRID = [
    {"C": c, "ngram_range": ng}
    for c in (0.5, 1.0, 2.0)
    for ng in ((1, 1), (1, 2))
]


def tune_tfidf():
    results = []
    for domain in DOMAINS:
        train = load_split(domain, "train")
        val = load_split(domain, "val")
        for params in TFIDF_GRID:
            for kind in ("tfidf_lr", "tfidf_svm"):
                model = build_model(kind, params)
                model.fit(train["text"].tolist(), train["label"].tolist())
                pred = model.predict(val["text"].tolist()).tolist()
                met = compute_metrics(val["label"].tolist(), pred)
                results.append(
                    {
                        "domain": domain,
                        "model": kind,
                        "C": params["C"],
                        "ngram_range": str(params["ngram_range"]),
                        "val_macro_f1": round(met["macro_f1"], 4),
                    }
                )
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS, "tuning_tfidf.csv"), index=False)
    best = df.sort_values("val_macro_f1", ascending=False).groupby(["domain", "model"]).first()
    print(best.to_string())
    return df


def pick_distilbert_lr():
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS, "transformer", "tune_*.json"))):
        res = read_json(path)
        rows.append(
            {
                "tag": res["tag"],
                "lr": res["lr"],
                "val_macro_f1": res["val"]["macro_f1"],
                "file": os.path.basename(path),
            }
        )
    if not rows:
        raise SystemExit("no DistilBERT tuning runs found under results/transformer")
    df = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)
    df.to_csv(os.path.join(RESULTS, "tuning_distilbert.csv"), index=False)
    best = df.iloc[0]
    print(f"[distilbert] best lr={best['lr']} val macro_f1={best['val_macro_f1']}")
    return {"lr": float(best["lr"]), "val_macro_f1": float(best["val_macro_f1"])}


def main():
    ensure_dirs()
    set_seed()

    tfidf_df = tune_tfidf()
    best_tfidf = (
        tfidf_df.sort_values("val_macro_f1", ascending=False)
        .groupby(["domain", "model"], as_index=False)
        .first()
    )
    distilbert = pick_distilbert_lr()

    config = {
        "seed": SEED,
        "data": {
            "train_per_class": 10000,
            "test_per_class": 5000,
            "val_rows_prepared": 2000,
            "val_rows_used": {"within": 1000, "mixed": 4000},
        },
        "tfidf": {
            "selection": "per-domain grid over C in {0.5,1,2} and ngram in {(1,1),(1,2)}",
            "best": best_tfidf.to_dict(orient="records"),
        },
        "distilbert": {
            "model": "distilbert-base-uncased",
            "max_len": 256,
            "batch_size": 32,
            "epochs": 3,
            "learning_rate": distilbert["lr"],
            "selection": "learning rate chosen on IMDb validation; applied to every domain",
            "val_macro_f1": distilbert["val_macro_f1"],
        },
    }
    with open(os.path.join(CONFIGS, "final.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    print("wrote configs/final.yaml")


if __name__ == "__main__":
    main()
