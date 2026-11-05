"""Stage 7: collect within-domain results into one table.

Reads results/baseline_within.csv (Stage 4) and results/transformer/*.json (Stage 5).

Run:  python src/run_within.py
"""
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DOMAINS, RESULTS, ensure_dirs, read_json  # noqa: E402

MODEL_LABEL = {
    "majority": "Majority class",
    "tfidf_lr": "TF-IDF + LR",
    "tfidf_svm": "TF-IDF + SVM",
    "distilbert": "DistilBERT",
}


def main():
    ensure_dirs()
    rows = []

    baseline_path = os.path.join(RESULTS, "baseline_within.csv")
    if os.path.exists(baseline_path):
        base = pd.read_csv(baseline_path)
        for _, r in base.iterrows():
            rows.append(
                {
                    "setting": "within",
                    "model": MODEL_LABEL.get(r["model"], r["model"]),
                    "train_domain": r["train_domains"],
                    "test_domain": r["test_domain"],
                    "accuracy": round(float(r["accuracy"]), 4),
                    "macro_f1": round(float(r["macro_f1"]), 4),
                    "val_macro_f1": round(float(r["val_macro_f1"]), 4),
                }
            )

    for path in sorted(glob.glob(os.path.join(RESULTS, "transformer", "*_to_*.json"))):
        res = read_json(path)
        if res.get("setting") != "within":
            continue
        if str(res.get("tag", "")).startswith("tune_"):
            continue
        if str(res.get("tag", "")).startswith("seed"):
            continue
        domain = res["test_domain"]
        rows.append(
            {
                "setting": "within",
                "model": MODEL_LABEL["distilbert"],
                "train_domain": domain,
                "test_domain": domain,
                "accuracy": round(res["test"]["accuracy"], 4),
                "macro_f1": round(res["test"]["macro_f1"], 4),
                "val_macro_f1": round(res["val"]["macro_f1"], 4),
            }
        )

    df = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(MODEL_LABEL.values())}
    df["_o"] = df["model"].map(order)
    df = df.sort_values(["_o", "test_domain"]).drop(columns=["_o"])
    out = os.path.join(RESULTS, "within_domain.csv")
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
