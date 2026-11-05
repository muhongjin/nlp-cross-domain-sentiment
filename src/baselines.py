"""Stage 4: majority-class baseline, TF-IDF + Logistic Regression, TF-IDF + Linear SVM.

Fits one model per domain on the training split, then evaluates on validation and test.
The TF-IDF vectoriser is fitted on training text only.

Run:  python src/baselines.py
"""
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    DOMAINS,
    MODELS,
    RESULTS,
    SEED,
    compute_metrics,
    ensure_dirs,
    load_split,
    set_seed,
    write_json,
)


def build_model(kind, params=None):
    params = params or {}
    if kind == "majority":
        return DummyClassifier(strategy="most_frequent")
    if kind == "tfidf_lr":
        return Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=tuple(params.get("ngram_range", (1, 2))),
                        min_df=params.get("min_df", 2),
                        max_features=params.get("max_features", 200000),
                        sublinear_tf=True,
                        strip_accents="unicode",
                        lowercase=True,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=params.get("C", 1.0),
                        max_iter=2000,
                        solver="liblinear",
                        random_state=SEED,
                    ),
                ),
            ]
        )
    if kind == "tfidf_svm":
        return Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=tuple(params.get("ngram_range", (1, 2))),
                        min_df=params.get("min_df", 2),
                        max_features=params.get("max_features", 200000),
                        sublinear_tf=True,
                        strip_accents="unicode",
                        lowercase=True,
                    ),
                ),
                ("clf", LinearSVC(C=params.get("C", 1.0), random_state=SEED)),
            ]
        )
    raise ValueError(kind)


def predict(model, texts):
    return model.predict(list(texts))


def train_and_evaluate(domain, kind, params=None, eval_splits=("val", "test")):
    train = load_split(domain, "train")
    model = build_model(kind, params)
    t0 = time.time()
    model.fit(train["text"].tolist(), train["label"].tolist())
    train_time = time.time() - t0

    out = {"domain": domain, "model": kind, "train_time_s": round(train_time, 2)}
    for split in eval_splits:
        df = load_split(domain, split)
        pred = predict(model, df["text"].tolist())
        out[split] = compute_metrics(df["label"].tolist(), pred.tolist())
    return model, out


def main():
    ensure_dirs()
    set_seed()
    rows = []

    for domain in DOMAINS:
        for kind in ("majority", "tfidf_lr", "tfidf_svm"):
            model, res = train_and_evaluate(domain, kind)
            path = os.path.join(MODELS, f"{kind}_{domain}.joblib")
            if kind != "majority":
                joblib.dump(model, path)
            write_json(os.path.join(RESULTS, "baseline", f"{kind}_{domain}.json"), res)
            rows.append(
                {
                    "setting": "within",
                    "train_domains": domain,
                    "test_domain": domain,
                    "model": kind,
                    "accuracy": res["test"]["accuracy"],
                    "macro_f1": res["test"]["macro_f1"],
                    "val_macro_f1": res["val"]["macro_f1"],
                }
            )
            print(
                f"[{domain}] {kind}: test acc={res['test']['accuracy']:.4f} "
                f"macro_f1={res['test']['macro_f1']:.4f}"
            )

    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "baseline_within.csv"), index=False)
    print("wrote results/baseline_within.csv")


if __name__ == "__main__":
    main()
