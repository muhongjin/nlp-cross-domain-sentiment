"""Stage 8: cross-domain evaluation.

Every model trained on a source domain is evaluated on the other two domains' test sets.
Cross-domain evaluation reuses the within-domain checkpoints, so no retraining happens here.

Run:  python src/run_cross.py
"""
import glob
import json
import os
import sys

import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    DOMAINS,
    MODELS,
    RESULTS,
    compute_metrics,
    ensure_dirs,
    load_split,
    read_json,
    set_seed,
)
from transformer_utils import predict_texts  # noqa: E402

MODEL_LABEL = {
    "tfidf_lr": "TF-IDF + LR",
    "tfidf_svm": "TF-IDF + SVM",
    "distilbert": "DistilBERT",
}


def load_within_scores():
    scores = {}
    path = os.path.join(RESULTS, "within_domain.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            scores[(r["model"], r["train_domain"], r["test_domain"])] = float(r["macro_f1"])
    return scores


def dump_errors(rows, path, limit=120):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows[:limit], f, indent=1)


def main():
    ensure_dirs()
    set_seed()
    within = load_within_scores()
    out_rows = []
    error_bucket = {}
    full_metrics = {}

    # ---- TF-IDF baselines: reuse the model fitted on the source domain ----
    for source in DOMAINS:
        for kind in ("tfidf_lr", "tfidf_svm"):
            model_path = os.path.join(MODELS, f"{kind}_{source}.joblib")
            if not os.path.exists(model_path):
                continue
            model = joblib.load(model_path)
            for target in DOMAINS:
                if target == source:
                    continue
                df = load_split(target, "test")
                pred = model.predict(df["text"].tolist()).tolist()
                met = compute_metrics(df["label"].tolist(), pred)
                full_metrics[f"{kind}_{source}_to_{target}"] = met
                base = within.get((MODEL_LABEL[kind], source, source))
                out_rows.append(
                    {
                        "setting": "cross",
                        "model": MODEL_LABEL[kind],
                        "source": source,
                        "target": target,
                        "accuracy": round(met["accuracy"], 4),
                        "macro_f1": round(met["macro_f1"], 4),
                        "within_macro_f1": base,
                        "drop": round(base - met["macro_f1"], 4) if base else None,
                    }
                )
                wrong = [
                    {
                        "text": t,
                        "true": int(y),
                        "pred": int(p),
                        "length": len(str(t).split()),
                    }
                    for t, y, p in zip(df["text"], df["label"], pred)
                    if y != p
                ]
                error_bucket[f"{kind}_{source}_to_{target}"] = wrong
                print(f"{kind}: {source} -> {target} macro_f1={met['macro_f1']:.4f}")

    # ---- DistilBERT checkpoints ----
    for source in DOMAINS:
        model_dir = os.path.join(MODELS, f"distilbert_{source}")
        if not os.path.isdir(model_dir):
            continue
        for target in DOMAINS:
            if target == source:
                continue
            df = load_split(target, "test")
            pred = predict_texts(model_dir, df["text"].tolist())
            met = compute_metrics(df["label"].tolist(), pred.tolist())
            full_metrics[f"distilbert_{source}_to_{target}"] = met
            base = within.get((MODEL_LABEL["distilbert"], source, source))
            out_rows.append(
                {
                    "setting": "cross",
                    "model": MODEL_LABEL["distilbert"],
                    "source": source,
                    "target": target,
                    "accuracy": round(met["accuracy"], 4),
                    "macro_f1": round(met["macro_f1"], 4),
                    "within_macro_f1": base,
                    "drop": round(base - met["macro_f1"], 4) if base else None,
                }
            )
            wrong = [
                {
                    "text": t,
                    "true": int(y),
                    "pred": int(p),
                    "length": len(str(t).split()),
                }
                for t, y, p in zip(df["text"], df["label"], pred)
                if y != p
            ]
            error_bucket[f"distilbert_{source}_to_{target}"] = wrong
            print(f"distilbert: {source} -> {target} macro_f1={met['macro_f1']:.4f}")

    out = pd.DataFrame(out_rows)
    if not out.empty:
        out = out.sort_values(["model", "source", "target"])
        out.to_csv(os.path.join(RESULTS, "cross_domain.csv"), index=False)
        print(out.to_string(index=False))
        print(f"wrote results/cross_domain.csv ({len(out)} rows)")

    for key, rows in error_bucket.items():
        dump_errors(rows, os.path.join(RESULTS, "errors", f"cross_{key}.json"))

    with open(os.path.join(RESULTS, "cross_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(full_metrics, f, indent=1)


if __name__ == "__main__":
    main()
