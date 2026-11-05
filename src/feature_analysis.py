"""Inspect which words each domain's classifier relies on.

Fits a TF-IDF + Logistic Regression model per domain and reports the strongest positive and
negative coefficients. The overlap of those word lists between domains shows how much of the
sentiment signal is shared and how much is domain vocabulary.

Writes results/top_features.csv and results/feature_overlap.csv.

Run:  python src/feature_analysis.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DOMAINS, RESULTS, SEED, ensure_dirs, load_split, set_seed  # noqa: E402
from baselines import build_model  # noqa: E402

TOP_K = 30


def top_terms(model, k=TOP_K):
    vectoriser = model.named_steps["tfidf"]
    classifier = model.named_steps["clf"]
    names = vectoriser.get_feature_names_out()
    coefs = classifier.coef_[0]
    order = coefs.argsort()
    negative = [names[i] for i in order[:k]]
    positive = [names[i] for i in order[-k:]][::-1]
    return positive, negative


def main():
    ensure_dirs()
    set_seed()
    rows = []
    terms = {}
    for domain in DOMAINS:
        train = load_split(domain, "train")
        model = build_model("tfidf_lr")
        model.fit(train["text"].tolist(), train["label"].tolist())
        positive, negative = top_terms(model)
        terms[domain] = {"positive": positive, "negative": negative}
        for rank, (pos, neg) in enumerate(zip(positive, negative), start=1):
            rows.append({"domain": domain, "rank": rank, "positive": pos, "negative": neg})

    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "top_features.csv"), index=False)

    overlap_rows = []
    for polarity in ("positive", "negative"):
        for i, a in enumerate(DOMAINS):
            for b in DOMAINS[i + 1 :]:
                set_a = set(terms[a][polarity][:20])
                set_b = set(terms[b][polarity][:20])
                overlap_rows.append(
                    {
                        "polarity": polarity,
                        "domain_a": a,
                        "domain_b": b,
                        "shared_in_top20": len(set_a & set_b),
                        "jaccard": round(len(set_a & set_b) / len(set_a | set_b), 3),
                        "shared_terms": ", ".join(sorted(set_a & set_b)),
                    }
                )
    overlap = pd.DataFrame(overlap_rows)
    overlap.to_csv(os.path.join(RESULTS, "feature_overlap.csv"), index=False)

    for domain in DOMAINS:
        print(f"[{domain}] positive: {', '.join(terms[domain]['positive'][:12])}")
        print(f"[{domain}] negative: {', '.join(terms[domain]['negative'][:12])}")
    print(overlap.to_string(index=False))


if __name__ == "__main__":
    main()
