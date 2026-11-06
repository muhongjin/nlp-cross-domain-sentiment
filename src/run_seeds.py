"""Aggregate the multi-seed DistilBERT runs and report mean and spread.

Seeds 43 and 44 repeat the within-domain training; seed 42 is the main run already used
in the results tables. Writes results/seed_robustness.csv.

Run:  python src/run_seeds.py
"""
import glob
import os

import numpy as np
import pandas as pd

from common import DOMAINS, RESULTS, ensure_dirs, read_json

DOMAIN_LABEL = {"imdb": "IMDb", "yelp": "Yelp", "amazon": "Amazon"}

def main():
    ensure_dirs()
    rows = []
    for domain in DOMAINS:
        scores = []
        main_path = os.path.join(RESULTS, "transformer", f"{domain}_to_{domain}.json")
        if os.path.exists(main_path):
            res = read_json(main_path)
            scores.append(("42", res["test"]["macro_f1"], res["val"]["macro_f1"]))
        for path in sorted(glob.glob(os.path.join(RESULTS, "transformer", f"seed*_{domain}_to_{domain}.json"))):
            res = read_json(path)
            scores.append((str(res.get("seed", "?")), res["test"]["macro_f1"], res["val"]["macro_f1"]))
        if not scores:
            continue
        test_scores = [s[1] for s in scores]
        rows.append(
            {
                "domain": DOMAIN_LABEL[domain],
                "seeds": ", ".join(s[0] for s in scores),
                "runs": len(scores),
                "test_macro_f1_mean": round(float(np.mean(test_scores)), 4),
                "test_macro_f1_std": round(float(np.std(test_scores, ddof=1)) if len(test_scores) > 1 else 0.0, 4),
                "test_macro_f1_min": round(float(np.min(test_scores)), 4),
                "test_macro_f1_max": round(float(np.max(test_scores)), 4),
            }
        )

    if not rows:
        print("no seed runs found")
        return
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "seed_robustness.csv"), index=False)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
