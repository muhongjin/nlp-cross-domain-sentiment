"""Majority-class baseline, TF-IDF + Logistic Regression and TF-IDF + Linear SVM.

Fits one model per domain on the training split, then evaluates on validation and test.
The TF-IDF vectoriser is fitted on training text only.

Run:  python src/baselines.py
"""
import os
import time

import joblib
import pandas as pd
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from common import (
    CONFIGS,
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


def load_tuned_params():
    """Per-domain TF-IDF settings chosen by src/tune.py, if that has been run."""
    path = os.path.join(CONFIGS, "final.yaml")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    tuned = {}
    for row in cfg.get("tfidf", {}).get("best", []):
        ngram = row.get("ngram_range", "(1, 2)")
        if isinstance(ngram, str):
            ngram = tuple(int(x) for x in ngram.strip("()").split(","))
        tuned[(row["domain"], row["model"])] = {"C": float(row["C"]), "ngram_range": tuple(ngram)}
    return tuned


def vectoriser(params):
    return TfidfVectorizer(
        ngram_range=tuple(params.get("ngram_range", (1, 2))),
        min_df=params.get("min_df", 2),
        max_features=params.get("max_features", 200000),
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )


def build_model(kind, params=None):
    params = params or {}
    if kind == "majority":
        return DummyClassifier(strategy="most_frequent")
    if kind == "tfidf_lr":
        return Pipeline(
            [
                ("tfidf", vectoriser(params)),
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
                ("tfidf", vectoriser(params)),
                ("clf", LinearSVC(C=params.get("C", 1.0), random_state=SEED)),
            ]
        )
    raise ValueError(kind)


def train_and_evaluate(domain, kind, params=None):
    train = load_split(domain, "train")
    model = build_model(kind, params)
    t0 = time.time()
    model.fit(train["text"].tolist(), train["label"].tolist())
    train_time = time.time() - t0

    out = {"domain": domain, "model": kind, "train_time_s": round(train_time, 2)}
    for split in ("val", "test"):
        df = load_split(domain, split)
        pred = model.predict(df["text"].tolist()).tolist()
        out[split] = compute_metrics(df["label"].tolist(), pred)
    return model, out


def main():
    ensure_dirs()
    set_seed()
    tuned = load_tuned_params()
    rows = []

    for domain in DOMAINS:
        for kind in ("majority", "tfidf_lr", "tfidf_svm"):
            params = tuned.get((domain, kind))
            model, res = train_and_evaluate(domain, kind, params)
            res["params"] = params or {}
            if kind != "majority":
                joblib.dump(model, os.path.join(MODELS, f"{kind}_{domain}.joblib"))
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
