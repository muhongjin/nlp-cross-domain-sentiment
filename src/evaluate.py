"""Build the result tables used by the report.

Reads the CSV and JSON outputs of the earlier scripts and writes results/tables/. The raw
per-run files stay in results/; this script only produces the pivoted views.

Run:  python src/evaluate.py
"""
import json
import os

import pandas as pd

from common import DOMAINS, RESULTS, ensure_dirs, fmt

TABLES = os.path.join(RESULTS, "tables")
MODELS = ["Majority class", "TF-IDF + LR", "TF-IDF + SVM", "DistilBERT"]
REAL_MODELS = ["TF-IDF + LR", "TF-IDF + SVM", "DistilBERT"]


def formatted(df, digits=3):
    """Render float columns with the same formatter the figures use."""
    out = df.copy()
    for column in out.columns:
        if pd.api.types.is_float_dtype(out[column]):
            out[column] = out[column].map(lambda v: fmt(v, digits) if pd.notna(v) else "-")
    return out


def markdown(df):
    # numparse off keeps the formatted strings intact; tabulate would otherwise re-parse
    # "0.900" as a number and print "0.9".
    return df.to_markdown(index=False, disable_numparse=True)


def main():
    ensure_dirs()
    os.makedirs(TABLES, exist_ok=True)
    lines = ["# Result tables", ""]

    within_path = os.path.join(RESULTS, "within_domain.csv")
    if os.path.exists(within_path):
        within = pd.read_csv(within_path)
        wide = within.pivot_table(index="model", columns="test_domain", values="macro_f1")
        wide = wide.reindex(index=[m for m in MODELS if m in wide.index], columns=DOMAINS)
        wide.loc["Average of the three classifiers"] = wide.loc[
            [m for m in REAL_MODELS if m in wide.index]
        ].mean()
        wide = wide.reset_index()
        lines += ["## Within-domain macro-F1", "", markdown(formatted(wide)), ""]

    cross_path = os.path.join(RESULTS, "cross_domain.csv")
    if os.path.exists(cross_path):
        cross = pd.read_csv(cross_path)
        lines += ["## Cross-domain results", "", markdown(formatted(cross)), ""]
        for model in cross["model"].unique():
            mat = (
                cross[cross.model == model]
                .pivot_table(index="source", columns="target", values="macro_f1")
                .reindex(index=DOMAINS, columns=DOMAINS)
            )
            lines += [
                f"### {model}: macro-F1 by source and target",
                "",
                markdown(formatted(mat.reset_index(), 3)),
                "",
            ]

    mixed_path = os.path.join(RESULTS, "mixed_domain.csv")
    if os.path.exists(mixed_path):
        mixed = pd.read_csv(mixed_path)
        lines += ["## Mixed-domain results", "", markdown(formatted(mixed)), ""]

    metrics_path = os.path.join(RESULTS, "cross_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        rows = []
        for key, met in sorted(metrics.items()):
            if not key.startswith("distilbert_"):
                continue
            source, target = key.replace("distilbert_", "").split("_to_")
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "positive_precision": met["per_class"]["positive"]["precision"],
                    "positive_recall": met["per_class"]["positive"]["recall"],
                    "negative_precision": met["per_class"]["negative"]["precision"],
                    "negative_recall": met["per_class"]["negative"]["recall"],
                }
            )
        if rows:
            table = pd.DataFrame(rows)
            table.round(4).to_csv(os.path.join(TABLES, "per_class_distilbert.csv"), index=False)
            lines += [
                "## Per-class metrics, DistilBERT cross-domain",
                "",
                markdown(formatted(table)),
                "",
            ]

    with open(os.path.join(TABLES, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {TABLES}/summary.md")


if __name__ == "__main__":
    main()
