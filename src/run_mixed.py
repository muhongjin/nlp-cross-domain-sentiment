"""Mixed-domain training.

Train on two source domains and test on the third, unseen domain. Training size is held
constant at 20,000 rows total (10,000 per source domain, balanced), so any gain comes from
domain diversity rather than from seeing more examples.

Run:  python src/run_mixed.py
"""
import glob
import os

import joblib
import pandas as pd

from common import (
    DOMAINS,
    MIXED_TRAIN_TOTAL,
    RESULTS,
    SEED,
    compute_metrics,
    ensure_dirs,
    load_split,
    read_json,
    set_seed,
)

MODEL_LABEL = {"tfidf_lr": "TF-IDF + LR", "tfidf_svm": "TF-IDF + SVM", "distilbert": "DistilBERT"}

def mixed_train_frame(sources, total=MIXED_TRAIN_TOTAL):
    per_domain = total // len(sources)
    parts = []
    for domain in sources:
        df = load_split(domain, "train")
        n = per_domain // 2
        parts.append(pd.concat([df[df.label == 0].head(n), df[df.label == 1].head(n)]))
    return pd.concat(parts).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

def single_source_scores():
    """Best cross-domain score on a target from any single source, per model."""
    best = {}
    path = os.path.join(RESULTS, "cross_domain.csv")
    if not os.path.exists(path):
        return best
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        key = (r["model"], r["target"])
        best[key] = max(best.get(key, 0.0), float(r["macro_f1"]))
    return best

def main():
    ensure_dirs()
    set_seed()
    from baselines import build_model

    single = single_source_scores()
    rows = []

    for target in DOMAINS:
        sources = [d for d in DOMAINS if d != target]
        train_df = mixed_train_frame(sources)
        print(f"mixed {sources} -> {target}: {len(train_df)} training rows")
        test_df = load_split(target, "test")

        for kind in ("tfidf_lr", "tfidf_svm"):
            model = build_model(kind)
            model.fit(train_df["text"].tolist(), train_df["label"].tolist())
            pred = model.predict(test_df["text"].tolist()).tolist()
            met = compute_metrics(test_df["label"].tolist(), pred)
            joblib.dump(model, os.path.join("models", f"{kind}_mixed_{target}.joblib"))
            label = MODEL_LABEL[kind]
            rows.append(
                {
                    "setting": "mixed",
                    "model": label,
                    "train_domains": "+".join(sources),
                    "target": target,
                    "train_size": int(len(train_df)),
                    "accuracy": round(met["accuracy"], 4),
                    "macro_f1": round(met["macro_f1"], 4),
                    "best_single_source_f1": round(single.get((label, target), float("nan")), 4),
                }
            )
            print(f"  {kind}: acc={met['accuracy']:.4f} macro_f1={met['macro_f1']:.4f}")

    for path in sorted(glob.glob(os.path.join(RESULTS, "transformer", "*_to_*.json"))):
        res = read_json(path)
        if res.get("setting") != "mixed":
            continue
        label = MODEL_LABEL["distilbert"]
        target = res["test_domain"]
        rows.append(
            {
                "setting": "mixed",
                "model": label,
                "train_domains": "+".join(res["train_domains"]),
                "target": target,
                "train_size": int(res["train_size"]),
                "accuracy": round(res["test"]["accuracy"], 4),
                "macro_f1": round(res["test"]["macro_f1"], 4),
                "best_single_source_f1": round(single.get((label, target), float("nan")), 4),
            }
        )

    df = pd.DataFrame(rows)
    df["delta_vs_best_single"] = (df["macro_f1"] - df["best_single_source_f1"]).round(4)
    df = df.sort_values(["model", "target"])
    out = os.path.join(RESULTS, "mixed_domain.csv")
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"wrote {out} ({len(df)} rows)")

if __name__ == "__main__":
    main()
