"""Smoke test: load each saved checkpoint and score a few reviews.

Run:  python scripts/smoke_test.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from transformer_utils import predict_texts  # noqa: E402

SAMPLES = [
    "This film was a joy from start to finish, the performances carry it.",
    "The food arrived cold and the waiter was rude about it.",
]

for domain in ("imdb", "yelp", "amazon"):
    path = os.path.join(ROOT, "models", f"distilbert_{domain}")
    if not os.path.isdir(path):
        print(f"{domain}: MISSING {path}")
        continue
    preds = predict_texts(path, SAMPLES)
    print(f"{domain}: {preds.tolist()} from {path}")
print("smoke test finished")
