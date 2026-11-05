"""Stage 11: error analysis.

Takes the misclassified reviews dumped by run_cross.py and groups them with simple
linguistic heuristics: negation, contrast, mixed opinion, length, and domain-specific
vocabulary. Writes results/error_taxonomy.csv, results/error_samples.json and
results/error_analysis.md.

Run:  python src/error_analysis.py
"""
import glob
import json
import os
import re
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, ensure_dirs  # noqa: E402

NEGATION = re.compile(r"\b(not|n't|never|no|nothing|hardly|barely|neither|nor|without)\b", re.I)
CONTRAST = re.compile(r"\b(but|however|although|though|yet|still|despite|instead)\b", re.I)
POSITIVE = re.compile(
    r"\b(good|great|excellent|love|loved|best|amazing|wonderful|perfect|enjoy|enjoyed|"
    r"recommend|fantastic|awesome|nice|brilliant|delightful|impressive)\b",
    re.I,
)
NEGATIVE = re.compile(
    r"\b(bad|terrible|awful|hate|hated|worst|boring|poor|disappoint|disappointed|"
    r"waste|annoying|horrible|dreadful|useless|stupid|mess)\b",
    re.I,
)

DOMAIN_WORDS = {
    "movie": re.compile(r"\b(film|movie|plot|actor|actress|director|scene|script|cast|cinema)\b", re.I),
    "business": re.compile(r"\b(restaurant|service|staff|waiter|food|menu|price|hotel|salon)\b", re.I),
    "product": re.compile(r"\b(amazon|product|shipping|seller|quality|price|order|delivery|item)\b", re.I),
}


def categorise(row):
    text = row["text"]
    cats = []
    if NEGATION.search(text):
        cats.append("negation")
    if CONTRAST.search(text):
        cats.append("contrast")
    has_pos = bool(POSITIVE.search(text))
    has_neg = bool(NEGATIVE.search(text))
    if has_pos and has_neg:
        cats.append("mixed opinion")
    words = len(str(text).split())
    if words > 300:
        cats.append("long review")
    elif words < 15:
        cats.append("very short review")
    if not cats:
        cats.append("other")
    row["categories"] = cats
    row["words"] = words
    return row


def main():
    ensure_dirs()
    files = sorted(glob.glob(os.path.join(RESULTS, "errors", "cross_*.json")))
    if not files:
        print("no error dumps found; run src/run_cross.py first")
        return

    rows = []
    samples = {}
    for path in files:
        key = os.path.basename(path)[len("cross_") : -len(".json")]
        with open(path, "r", encoding="utf-8") as f:
            errs = json.load(f)
        if not errs:
            continue
        for err in errs:
            err = categorise(dict(err))
            err["setting"] = key
            rows.append(err)
        samples[key] = errs[:12]

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "error_cases.csv"), index=False)

    taxonomy = (
        df.explode("categories")
        .groupby(["setting", "categories"])
        .size()
        .reset_index(name="count")
    )
    taxonomy["share_of_errors"] = (
        taxonomy["count"] / taxonomy.groupby("setting")["count"].transform("sum")
    ).round(3)
    taxonomy.to_csv(os.path.join(RESULTS, "error_taxonomy.csv"), index=False)

    overall = (
        df.explode("categories")
        .groupby("categories")
        .agg(count=("categories", "size"), mean_words=("words", "mean"))
        .reset_index()
        .sort_values("count", ascending=False)
    )
    overall["share"] = (overall["count"] / overall["count"].sum()).round(3)
    overall["mean_words"] = overall["mean_words"].round(1)
    overall.to_csv(os.path.join(RESULTS, "error_taxonomy_overall.csv"), index=False)

    with open(os.path.join(RESULTS, "error_samples.json"), "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=1)

    lines = ["# Error analysis", ""]
    lines.append(
        "Categories come from keyword heuristics applied to the misclassified reviews, so "
        "they overlap: one review can be tagged as both negation and contrast. Counts below "
        "are therefore shares of tagged errors, not a partition."
    )
    lines.append("")
    lines.append("## Overall error categories")
    lines.append("")
    lines.append(overall.to_markdown(index=False))
    lines.append("")
    lines.append("## Per-setting breakdown")
    lines.append("")
    lines.append(taxonomy.to_markdown(index=False))
    lines.append("")
    lines.append("## Confusion direction")
    lines.append("")
    lines.append(
        df.groupby(["setting", "true", "pred"]).size().reset_index(name="count").to_markdown(index=False)
    )

    with open(os.path.join(RESULTS, "error_analysis.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(overall.to_string(index=False))
    print("wrote results/error_analysis.md, error_taxonomy.csv, error_samples.json")


if __name__ == "__main__":
    main()
