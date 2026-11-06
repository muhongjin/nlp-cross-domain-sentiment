"""Group misclassified reviews with simple linguistic heuristics.

Reads the error dumps written by run_cross.py and tags each review for negation, contrast,
mixed opinion, length and so on. Writes results/error_cases.csv, results/error_taxonomy.csv,
results/error_taxonomy_overall.csv, results/error_samples.json and results/error_analysis.md.

The share columns are fractions of tags, not of reviews: a review can carry two tags, so the
per-setting shares add up to one only because the denominator is the tag count.

Run:  python src/error_analysis.py
"""
import glob
import json
import os
import re

import pandas as pd

from common import RESULTS, ensure_dirs

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


def categorise(row):
    text = row["text"]
    tags = []
    if NEGATION.search(text):
        tags.append("negation")
    if CONTRAST.search(text):
        tags.append("contrast")
    if POSITIVE.search(text) and NEGATIVE.search(text):
        tags.append("mixed opinion")
    words = len(str(text).split())
    if words > 300:
        tags.append("long review")
    elif words < 15:
        tags.append("very short review")
    if not tags:
        tags.append("other")
    row["categories"] = tags
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
            errors = json.load(f)
        if not errors:
            continue
        for error in errors:
            tagged = categorise(dict(error))
            tagged["setting"] = key
            rows.append(tagged)
        samples[key] = errors[:12]

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "error_cases.csv"), index=False)

    exploded = df.explode("categories")
    taxonomy = exploded.groupby(["setting", "categories"]).size().reset_index(name="count")
    taxonomy["share_of_errors"] = (
        taxonomy["count"] / taxonomy.groupby("setting")["count"].transform("sum")
    ).round(3)
    taxonomy.to_csv(os.path.join(RESULTS, "error_taxonomy.csv"), index=False)

    overall = (
        exploded.groupby("categories")
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
        "Categories come from keyword heuristics, so they overlap: one review can be tagged "
        "negation and contrast at once. Shares are fractions of tags."
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
