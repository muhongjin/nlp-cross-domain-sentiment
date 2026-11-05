"""Stage 10: build the result tables used by the report.

Reads the per-stage CSV and JSON outputs and writes markdown tables to results/tables/.

Run:  python src/evaluate.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, ensure_dirs  # noqa: E402

TABLES = os.path.join(RESULTS, "tables")


def markdown_table(df, index=False):
    try:
        return df.to_markdown(index=index)
    except Exception:
        return df.to_string(index=index)


def main():
    ensure_dirs()
    os.makedirs(TABLES, exist_ok=True)
    lines = ["# Result tables", ""]

    within_path = os.path.join(RESULTS, "within_domain.csv")
    if os.path.exists(within_path):
        w = pd.read_csv(within_path)
        wide = w.pivot_table(index="model", columns="test_domain", values="macro_f1")
        wide = wide[["imdb", "yelp", "amazon"]]
        wide.loc["Average"] = wide.mean()
        wide = wide.round(4).reset_index()
        wide.to_csv(os.path.join(TABLES, "within_domain.csv"), index=False)
        lines += ["## Within-domain macro-F1", "", markdown_table(wide), ""]

    cross_path = os.path.join(RESULTS, "cross_domain.csv")
    if os.path.exists(cross_path):
        c = pd.read_csv(cross_path)
        c.to_csv(os.path.join(TABLES, "cross_domain.csv"), index=False)
        lines += ["## Cross-domain results", "", markdown_table(c), ""]
        for model in c["model"].unique():
            sub = c[c["model"] == model]
            mat = sub.pivot_table(index="source", columns="target", values="macro_f1")
            mat = mat.reindex(index=["imdb", "yelp", "amazon"], columns=["imdb", "yelp", "amazon"])
            mat = mat.round(4)
            lines += [f"### {model}: macro-F1 by source and target", "", markdown_table(mat, index=True), ""]

    mixed_path = os.path.join(RESULTS, "mixed_domain.csv")
    if os.path.exists(mixed_path):
        m = pd.read_csv(mixed_path)
        m.to_csv(os.path.join(TABLES, "mixed_domain.csv"), index=False)
        lines += ["## Mixed-domain results", "", markdown_table(m), ""]

    metrics_path = os.path.join(RESULTS, "cross_metrics.json")
    if os.path.exists(metrics_path):
        import json

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
                    "positive_precision": round(met["per_class"]["positive"]["precision"], 4),
                    "positive_recall": round(met["per_class"]["positive"]["recall"], 4),
                    "negative_precision": round(met["per_class"]["negative"]["precision"], 4),
                    "negative_recall": round(met["per_class"]["negative"]["recall"], 4),
                }
            )
        if rows:
            table = pd.DataFrame(rows)
            table.to_csv(os.path.join(TABLES, "per_class_distilbert.csv"), index=False)
            lines += ["## Per-class metrics, DistilBERT cross-domain", "", markdown_table(table), ""]

    with open(os.path.join(TABLES, "results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {TABLES}/results.md")


if __name__ == "__main__":
    main()
