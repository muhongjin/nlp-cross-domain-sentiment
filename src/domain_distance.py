"""Quantify how far apart the three domains are, in vocabulary and in length.

Two measures per domain pair:
  - Jensen-Shannon divergence between unigram distributions over the test split
  - Jaccard overlap of the 5,000 most frequent words

Writes results/domain_distance.csv and figures/fig_domain_distance.png.

Run:  python src/domain_distance.py
"""
import os
import sys
from collections import Counter
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DOMAINS, FIGURES, RESULTS, ensure_dirs, load_split  # noqa: E402

DOMAIN_LABEL = {"imdb": "IMDb", "yelp": "Yelp", "amazon": "Amazon"}
TOP_WORDS = 5000


def unigram_counts(texts):
    counter = Counter()
    for text in texts:
        counter.update(str(text).lower().split())
    return counter


def js_divergence(counter_a, counter_b):
    vocab = set(counter_a) | set(counter_b)
    total_a = sum(counter_a.values())
    total_b = sum(counter_b.values())
    pa = np.array([counter_a.get(w, 0) / total_a for w in vocab])
    pb = np.array([counter_b.get(w, 0) / total_b for w in vocab])
    m = 0.5 * (pa + pb)
    mask_a = pa > 0
    mask_b = pb > 0
    kl_a = np.sum(pa[mask_a] * np.log2(pa[mask_a] / m[mask_a]))
    kl_b = np.sum(pb[mask_b] * np.log2(pb[mask_b] / m[mask_b]))
    return float(0.5 * (kl_a + kl_b))


def main():
    ensure_dirs()
    counters = {}
    lengths = {}
    top_words = {}
    for domain in DOMAINS:
        df = load_split(domain, "test")
        counters[domain] = unigram_counts(df["text"])
        lengths[domain] = float(df["text"].str.split().str.len().mean())
        top_words[domain] = {w for w, _ in counters[domain].most_common(TOP_WORDS)}

    rows = []
    matrix = pd.DataFrame(np.nan, index=DOMAINS, columns=DOMAINS, dtype=float)
    for a, b in combinations(DOMAINS, 2):
        jsd = js_divergence(counters[a], counters[b])
        overlap = len(top_words[a] & top_words[b]) / len(top_words[a] | top_words[b])
        rows.append(
            {
                "domain_a": a,
                "domain_b": b,
                "js_divergence": round(jsd, 4),
                "top5000_jaccard": round(overlap, 4),
                "mean_words_a": round(lengths[a], 1),
                "mean_words_b": round(lengths[b], 1),
            }
        )
        matrix.loc[a, b] = jsd
        matrix.loc[b, a] = jsd
    for domain in DOMAINS:
        matrix.loc[domain, domain] = 0.0

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "domain_distance.csv"), index=False)
    print(out.to_string(index=False))

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    im = ax.imshow(matrix.values, cmap="Blues")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels([DOMAIN_LABEL[d] for d in DOMAINS])
    ax.set_yticklabels([DOMAIN_LABEL[d] for d in DOMAINS])
    ax.set_title("Jensen-Shannon divergence between domains")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{matrix.values[i, j]:.3f}", ha="center", va="center", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8, label="JSD (bits)")
    fig.savefig(os.path.join(FIGURES, "fig_domain_distance.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote results/domain_distance.csv and figures/fig_domain_distance.png")


if __name__ == "__main__":
    main()
