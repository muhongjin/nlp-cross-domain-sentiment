"""Near-duplicate screening across the training and test splits.

Exact duplicates were already removed when the splits were built. This script checks how
many near-duplicates remain by sampling a few shingles per review as an index key, then
scoring candidate pairs with an exact Jaccard similarity over word 5-grams.

A pair counts as a near-duplicate when the Jaccard similarity is at least 0.8. Pairs are
grouped by where they occur: inside one split, across the train and test split of the same
domain, or across domains.

Writes results/near_duplicates.json.

Run:  python src/near_duplicates.py
"""
import hashlib
import json
import os
from collections import defaultdict
from itertools import combinations

from common import DOMAINS, RESULTS, ensure_dirs, load_split, write_json

SIGNATURE_SIZE = 8
JACCARD_THRESHOLD = 0.8
SPLITS = ("train", "test")

def shingles(text, k=5):
    words = str(text).lower().split()
    if len(words) < k:
        return {" ".join(words)}
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}

def signature(shingle_set):
    """Pick the same shingles on every run so the candidate set is reproducible."""
    return sorted(shingle_set, key=_shingle_key)[:SIGNATURE_SIZE]

def _shingle_key(shingle):
    return hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest()

def jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)

def main():
    ensure_dirs()
    docs = []
    for domain in DOMAINS:
        for split in SPLITS:
            df = load_split(domain, split)
            for text in df["text"]:
                docs.append((domain, split, str(text)))
    print(f"pooled {len(docs)} reviews from train and test")

    shingle_sets = [shingles(text) for _, _, text in docs]
    index = defaultdict(list)
    for doc_id, shingle_set in enumerate(shingle_sets):
        for key in signature(shingle_set):
            index[key].append(doc_id)

    candidates = set()
    for posting in index.values():
        if 1 < len(posting) <= 50:
            for a, b in combinations(sorted(posting), 2):
                candidates.add((a, b))
    print(f"{len(candidates)} candidate pairs from the signature index")

    near = []
    for a, b in candidates:
        score = jaccard(shingle_sets[a], shingle_sets[b])
        if score >= JACCARD_THRESHOLD:
            near.append((a, b, score))

    buckets = defaultdict(int)
    for a, b, _ in near:
        dom_a, split_a, _ = docs[a]
        dom_b, split_b, _ = docs[b]
        if dom_a == dom_b and split_a == split_b:
            buckets[f"{dom_a}_within_{split_a}"] += 1
        elif dom_a == dom_b:
            buckets[f"{dom_a}_train_test_leak"] += 1
        else:
            buckets["cross_domain"] += 1

    summary = {
        "pooled_reviews": len(docs),
        "splits_checked": list(SPLITS),
        "candidate_pairs": len(candidates),
        "near_duplicate_pairs": len(near),
        "jaccard_threshold": JACCARD_THRESHOLD,
        "signature_size": SIGNATURE_SIZE,
        "buckets": dict(sorted(buckets.items())),
        "examples": [
            {
                "domain_a": docs[a][0],
                "split_a": docs[a][1],
                "domain_b": docs[b][0],
                "split_b": docs[b][1],
                "jaccard": round(score, 3),
                "text_a": docs[a][2][:200],
                "text_b": docs[b][2][:200],
            }
            for a, b, score in sorted(near, key=lambda x: -x[2])[:20]
        ],
    }
    write_json(os.path.join(RESULTS, "near_duplicates.json"), summary)
    print(json.dumps({k: summary[k] for k in ("near_duplicate_pairs", "buckets")}, indent=2))

if __name__ == "__main__":
    main()
