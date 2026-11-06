"""Cross-domain evaluation.

Every model trained on a source domain is evaluated on the other two domains' test sets.
Cross-domain evaluation reuses the within-domain checkpoints, so no retraining happens here.

Each setting also dumps a sample of its misclassified reviews for the error analysis. The
sample is capped at MAX_ERRORS_PER_SETTING; the full error counts live in
results/cross_metrics.json, which is built from the confusion matrices.

Run:  python src/run_cross.py
"""
import json
import os

import joblib
import pandas as pd

from common import (
    DOMAINS,
    MODELS,
    RESULTS,
    compute_metrics,
    ensure_dirs,
    load_split,
    set_seed,
)
from transformer_utils import predict_texts

MODEL_LABEL = {
    "tfidf_lr": "TF-IDF + LR",
    "tfidf_svm": "TF-IDF + SVM",
    "distilbert": "DistilBERT",
}

# Enough for qualitative reading without bloating the repository.
MAX_ERRORS_PER_SETTING = 120


def load_within_scores():
    scores = {}
    path = os.path.join(RESULTS, "within_domain.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            scores[(r["model"], r["train_domain"], r["test_domain"])] = float(r["macro_f1"])
    return scores


def record(kind, source, target, model_label, pred, frame, within, state):
    met = compute_metrics(frame["label"].tolist(), list(pred))
    state["metrics"][f"{kind}_{source}_to_{target}"] = met
    base = within.get((model_label, source, source))
    state["rows"].append(
        {
            "setting": "cross",
            "model": model_label,
            "source": source,
            "target": target,
            "accuracy": round(met["accuracy"], 4),
            "macro_f1": round(met["macro_f1"], 4),
            "within_macro_f1": base,
            "drop": round(base - met["macro_f1"], 4) if base else None,
        }
    )
    state["errors"][f"{kind}_{source}_to_{target}"] = [
        {"text": t, "true": int(y), "pred": int(p), "length": len(str(t).split())}
        for t, y, p in zip(frame["text"], frame["label"], pred)
        if y != p
    ]
    print(f"{model_label}: {source} -> {target} macro_f1={met['macro_f1']:.4f}")


def main():
    ensure_dirs()
    set_seed()
    within = load_within_scores()
    state = {"rows": [], "errors": {}, "metrics": {}}

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
                record(kind, source, target, MODEL_LABEL[kind], pred, df, within, state)

    for source in DOMAINS:
        model_dir = os.path.join(MODELS, f"distilbert_{source}")
        if not os.path.isdir(model_dir):
            continue
        for target in DOMAINS:
            if target == source:
                continue
            df = load_split(target, "test")
            pred = predict_texts(model_dir, df["text"].tolist()).tolist()
            record("distilbert", source, target, MODEL_LABEL["distilbert"], pred, df, within, state)

    out = pd.DataFrame(state["rows"])
    if not out.empty:
        out = out.sort_values(["model", "source", "target"])
        out.to_csv(os.path.join(RESULTS, "cross_domain.csv"), index=False)
        print(out.to_string(index=False))
        print(f"wrote results/cross_domain.csv ({len(out)} rows)")

    errors_dir = os.path.join(RESULTS, "errors")
    os.makedirs(errors_dir, exist_ok=True)
    for key, rows in state["errors"].items():
        with open(os.path.join(errors_dir, f"cross_{key}.json"), "w", encoding="utf-8") as f:
            json.dump(rows[:MAX_ERRORS_PER_SETTING], f, indent=1)

    with open(os.path.join(RESULTS, "cross_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(state["metrics"], f, indent=1)


if __name__ == "__main__":
    main()
