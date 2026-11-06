"""Shared paths, constants and metric helpers for the cross-domain sentiment project."""
import json
import os
import random

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

SEED = 42
DOMAINS = ["imdb", "yelp", "amazon"]
DOMAIN_LABEL = {
    "imdb": "IMDb (movie reviews)",
    "yelp": "Yelp (business reviews)",
    "amazon": "Amazon (product reviews)",
}
LABEL_NAMES = ["negative", "positive"]

TRAIN_PER_CLASS = 10000
VAL_PER_CLASS = 1000
TEST_PER_CLASS = 5000
MIXED_TRAIN_TOTAL = 20000

MAX_LEN = 256

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(ROOT, "data", "processed")
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
MODELS = os.path.join(ROOT, "models")
CONFIGS = os.path.join(ROOT, "configs")

def set_seed(seed=SEED):
    """Seed the process. Torch seeds are set separately by the training scripts."""
    random.seed(seed)
    np.random.seed(seed)

def fmt(value, digits=3):
    """Format a score the same way in tables and figures."""
    return f"{value:.{digits}f}"

def ensure_dirs():
    for d in (PROCESSED, RESULTS, FIGURES, MODELS, CONFIGS):
        os.makedirs(d, exist_ok=True)

def write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_metrics(y_true, y_pred):
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "per_class": {
            LABEL_NAMES[i]: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(s[i]),
            }
            for i in range(2)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }

def load_split(domain, split):
    return pd.read_parquet(os.path.join(PROCESSED, domain, f"{split}.parquet"))
