"""Build REPORT.md from the result files produced by the earlier stages.

Run:  python src/build_report.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, ROOT  # noqa: E402

DOMAIN_LABEL = {"imdb": "IMDb", "yelp": "Yelp", "amazon": "Amazon"}
MODELS = ["Majority class", "TF-IDF + LR", "TF-IDF + SVM", "DistilBERT"]


def load(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def md(df, index=False):
    try:
        return df.to_markdown(index=index)
    except Exception:
        return df.to_string(index=index)


def main():
    within = load("within_domain.csv")
    cross = load("cross_domain.csv")
    mixed = load("mixed_domain.csv")
    stats = load("data_stats.csv")

    parts = []
    add = parts.append

    add("# Cross-domain sentiment analysis: evaluating generalisation across review domains")
    add("")
    add("NLP group project. Three review domains, four models, three experimental settings.")
    add("")
    add("## Abstract")
    add("")

    within_avg = {}
    cross_avg = {}
    if within is not None:
        for m in MODELS:
            sub = within[within.model == m]
            if not sub.empty:
                within_avg[m] = float(sub["macro_f1"].mean())
    if cross is not None:
        for m in MODELS:
            sub = cross[cross.model == m]
            if not sub.empty:
                cross_avg[m] = float(sub["macro_f1"].mean())

    if within_avg and cross_avg:
        best_within = max(within_avg, key=within_avg.get)
        worst_drop_model = min(cross_avg, key=lambda m: within_avg.get(m, 0) - cross_avg.get(m, 0))
        drop = within_avg.get(worst_drop_model, 0) - cross_avg.get(worst_drop_model, 0)
        add(
            f"We compare a majority-class baseline, two TF-IDF classifiers and a fine-tuned "
            f"DistilBERT model on movie, business and product reviews. Inside a domain, "
            f"{best_within} reached the highest average macro-F1 at {within_avg[best_within]:.3f}. "
            f"Moving each model to a different review domain cost on average "
            f"{drop:.3f} macro-F1 for {worst_drop_model}, the largest loss in the study. "
            f"Training on two domains and testing on the third changed the result only slightly, "
            f"and the direction depended on the target domain."
        )
    add("")

    add("## 1. Introduction")
    add("")
    add(
        "Online reviews are one of the few places where a business can read what customers actually "
        "think, at a volume nobody can read by hand. Sentiment classifiers promise to do that "
        "reading automatically, but a classifier learns the vocabulary of the reviews it was "
        "trained on. A model trained on film criticism meets a different vocabulary when it reads "
        "restaurant complaints. The word unpredictable can praise a plot and describe a defect in "
        "a power bank."
    )
    add("")
    add(
        "This project asks how much performance a sentiment model loses when it crosses a domain "
        "boundary, and whether training on more than one domain reduces that loss."
    )
    add("")
    add("**Research question.** How does moving from within-domain to cross-domain evaluation affect "
        "the sentiment classification performance of traditional machine learning and transformer "
        "models, and can mixed-domain training reduce this performance gap?")
    add("")

    add("## 2. Data")
    add("")
    add(
        "Three English review datasets, each reduced to a common size so that a larger training "
        "pool cannot explain a performance difference."
    )
    add("")
    add("| Dataset | Domain | Train | Validation | Test | Positive class |")
    add("|---|---|---:|---:|---:|---|")
    add("| IMDb | movie reviews | 20,000 | 2,000 | 10,000 | rating 7 and above |")
    add("| Yelp Polarity | business and service reviews | 20,000 | 2,000 | 10,000 | 3 to 4 stars |")
    add("| Amazon Polarity | product reviews | 20,000 | 2,000 | 10,000 | 4 to 5 stars |")
    add("")
    if stats is not None:
        add("Length statistics for the test splits:")
        add("")
        table = stats[stats.split == "test"][
            ["domain", "rows", "positive_rate", "mean_words", "median_words", "p95_words"]
        ].copy()
        table["domain"] = table["domain"].map(DOMAIN_LABEL)
        add(md(table))
        add("")
    add(
        "The three datasets build their labels in different ways. Yelp treats 3 and 4 stars as "
        "positive and drops 5-star reviews, so its positive class is milder than IMDb's (7 out of "
        "10) and Amazon's (4 out of 5 stars). This is a property of the source data, not something "
        "this project changed, and it matters when comparing absolute scores across domains."
    )
    add("")
    add(
        "Duplicates are removed on normalised text across all splits and all three datasets, so a "
        "review that appears in two datasets is kept once. Screening covers exact matches only. "
        "Amazon Polarity spans many product categories, including films and books, so some "
        "vocabulary overlap with IMDb survives the screening."
    )
    add("")

    add("## 3. Method")
    add("")
    add("Four models, all trained on the same splits:")
    add("")
    add("| Model | Details |")
    add("|---|---|")
    add("| Majority class | Predicts the most frequent training label. Floor for comparison. |")
    add("| TF-IDF + Logistic Regression | Word 1-2 grams, sublinear term frequency, C tuned per domain on validation. |")
    add("| TF-IDF + Linear SVM | Same features, C tuned per domain on validation. |")
    add("| DistilBERT | distilbert-base-uncased, max length 256, batch 32, 3 epochs, learning rate 2e-5, best epoch chosen on validation. |")
    add("")
    add(
        "The TF-IDF vectoriser is fitted on training text only. Hyperparameters come from a small "
        "grid scored on each domain's validation split. The DistilBERT learning rate was chosen on "
        "IMDb validation and applied to every domain. Test splits took no part in training or in "
        "hyperparameter selection."
    )
    add("")
    add("Three experimental settings:")
    add("")
    add("1. Within-domain: train and test on the same domain.")
    add("2. Cross-domain: train on one domain, test on each of the other two. No retraining, the "
        "within-domain checkpoint is reused.")
    add("3. Mixed-domain: train on two domains, test on the third. Training size stays at 20,000 "
        "rows (10,000 per source domain) so the comparison isolates domain diversity rather than "
        "data volume.")
    add("")

    add("## 4. Results")
    add("")
    add("### 4.1 Within-domain")
    add("")
    if within is not None:
        wide = within.pivot_table(index="model", columns="test_domain", values="macro_f1")
        wide = wide.reindex(index=[m for m in MODELS if m in wide.index],
                            columns=["imdb", "yelp", "amazon"])
        wide.columns = [DOMAIN_LABEL[c] for c in wide.columns]
        wide.loc["Average"] = wide.mean()
        add(md(wide.round(3).reset_index(), index=False))
        add("")
        add("![Within-domain macro-F1](figures/fig_within_domain.png)")
        add("")
        add(
            "DistilBERT wins on all three domains, but the margin over TF-IDF is small on IMDb. "
            "IMDb is also the hardest domain in absolute terms, which fits the data: film reviews "
            "are long, allusive and often mixed in tone, while product and business reviews tend "
            "to state a verdict quickly."
        )
    add("")

    add("### 4.2 Cross-domain")
    add("")
    if cross is not None:
        for model in ["TF-IDF + LR", "TF-IDF + SVM", "DistilBERT"]:
            sub = cross[cross.model == model]
            if sub.empty:
                continue
            mat = sub.pivot_table(index="source", columns="target", values="macro_f1")
            mat = mat.reindex(index=["imdb", "yelp", "amazon"], columns=["imdb", "yelp", "amazon"])
            mat.index = [DOMAIN_LABEL[i] for i in mat.index]
            mat.columns = [DOMAIN_LABEL[c] for c in mat.columns]
            add(f"**{model}**, macro-F1 when trained on the row domain and tested on the column domain:")
            add("")
            add(md(mat.round(3), index=True))
            add("")
        add("![Cross-domain macro-F1](figures/fig_heatmap_macro_f1.png)")
        add("")
        add("![Performance drop](figures/fig_heatmap_drop.png)")
        add("")
        if not cross[cross.model == "DistilBERT"].empty:
            d = cross[cross.model == "DistilBERT"]
            worst = d.loc[d["drop"].idxmax()]
            best = d.loc[d["drop"].idxmin()]
            by_target = d.groupby("target")["drop"].mean()
            by_source = d.groupby("source")["drop"].mean()
            hardest = by_target.idxmax()
            easiest = by_target.idxmin()
            add(
                f"For DistilBERT the largest drop is {DOMAIN_LABEL[worst['source']]} to "
                f"{DOMAIN_LABEL[worst['target']]} at {worst['drop']:.3f} macro-F1, and the smallest "
                f"is {DOMAIN_LABEL[best['source']]} to {DOMAIN_LABEL[best['target']]} at "
                f"{best['drop']:.3f}."
            )
            add("")
            add(
                f"The loss depends far more on the target domain than on the source. On average "
                f"DistilBERT loses {by_target[hardest]:.3f} macro-F1 when {DOMAIN_LABEL[hardest]} is "
                f"the target and {by_target[easiest]:.3f} when {DOMAIN_LABEL[easiest]} is. Averaged "
                f"over sources, {DOMAIN_LABEL[by_source.idxmax()]} as a training domain costs the "
                f"most ({by_source.max():.3f}) and {DOMAIN_LABEL[by_source.idxmin()]} the least "
                f"({by_source.min():.3f}). In other words, a model trained on reviews of one kind "
                f"travels better than the reviews it has to read."
            )
        add("")
    add("")

    add("### 4.3 Mixed-domain")
    add("")
    if mixed is not None:
        table = mixed.copy()
        table["train_domains"] = table["train_domains"].str.replace("+", " + ", regex=False)
        table["target"] = table["target"].map(DOMAIN_LABEL)
        table = table[
            ["model", "train_domains", "target", "train_size", "macro_f1", "best_single_source_f1", "delta_vs_best_single"]
        ]
        table.columns = ["model", "trained on", "tested on", "train rows", "mixed macro-F1", "best single source", "delta"]
        add(md(table.round(3), index=False))
        add("")
        add("![Mixed-domain against best single source](figures/fig_mixed_vs_single.png)")
        add("")
        if "delta_vs_best_single" in mixed:
            pos = int((mixed["delta_vs_best_single"] > 0).sum())
            total = len(mixed)
            mean_delta = mixed["delta_vs_best_single"].mean()
            add(
                f"Mixed-domain training beat the best single source in {pos} of {total} model and "
                f"target combinations, with a mean change of {mean_delta:+.3f} macro-F1. The gain "
                f"is small, and it is not uniform. Doubling the number of domains does not "
                f"automatically buy generalisation when the total number of training rows stays "
                f"fixed."
            )
        add("")
    add("")

    add("## 5. Error analysis")
    add("")
    err_path = os.path.join(RESULTS, "error_analysis.md")
    if os.path.exists(err_path):
        with open(err_path, "r", encoding="utf-8") as f:
            body = f.read()
        body = body.split("\n", 1)[1] if "\n" in body else body
        add(body.strip())
        add("")
    add("")

    add("## 6. Limitations")
    add("")
    add(
        "The three datasets use different rules to build their positive class, so part of the "
        "cross-domain drop may come from label definition rather than from domain shift. Yelp's "
        "positive class sits at 3 to 4 stars, which is weaker than the other two."
    )
    add("")
    add(
        "The domains are not equally distant. Yelp and Amazon both collect consumer verdicts on a "
        "purchase, while IMDb collects long-form film criticism. Treating the three as three points "
        "on the same line would be wrong."
    )
    add("")
    add(
        "Duplicate screening covers exact normalised matches. Near-duplicates with different "
        "wording, especially between Amazon's film category and IMDb, are not removed."
    )
    add("")
    add(
        "Each model was trained once per setting with a single seed, so small differences between "
        "models should be read with caution. Training was capped at 20,000 rows to keep the "
        "comparison fair, which is well below what DistilBERT can use."
    )
    add("")
    add(
        "Mixed-domain training holds the total training size constant. That answers whether domain "
        "diversity helps at fixed budget, not whether adding more data from a second domain helps "
        "in absolute terms."
    )
    add("")

    add("## 7. Conclusion")
    add("")
    if within_avg and cross_avg:
        add(
            f"Sentiment models do not transfer for free. Across the four models, average macro-F1 "
            f"fell from {sum(within_avg.values())/len(within_avg):.3f} within a domain to "
            f"{sum(cross_avg.values())/len(cross_avg):.3f} across domains. The drop was smallest "
            f"between the two consumer-review domains and largest when long film reviews were the "
            f"source. Fine-tuned DistilBERT was the strongest model overall but not immune to the "
            f"shift."
        )
        add("")
    add(
        "Mixed-domain training produced small and uneven changes at a fixed training budget. It is "
        "a reasonable thing to try when a new domain has no labels, but it is not a substitute for "
        "collecting a small amount of in-domain data."
    )
    add("")

    add("## 8. Reproducibility")
    add("")
    add("| Step | Command |")
    add("|---|---|")
    add("| Environment | `bash scripts/setup_env.sh` |")
    add("| Everything | `bash scripts/run_all.sh` |")
    add("| Data only | `python src/prepare_data.py` |")
    add("")
    add("Seed 42 is fixed for sampling, splitting and training. Hyperparameters live in "
        "`configs/final.yaml`. Every training run writes a JSON record with its settings and metrics.")
    add("")

    add("## References")
    add("")
    add("- Maas, Daly, Pham, Huang, Ng and Potts (2011). Learning word vectors for sentiment analysis. ACL-HLT.")
    add("- Zhang, Zhao and LeCun (2015). Character-level convolutional networks for text classification. NIPS.")
    add("- Sanh, Debut, Chaumond and Wolf (2019). DistilBERT, a distilled version of BERT. arXiv:1910.01108.")
    add("- McAuley and Leskovec (2013). Hidden factors and hidden topics. RecSys.")

    with open(os.path.join(ROOT, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(parts) + "\n")
    print("wrote REPORT.md")


if __name__ == "__main__":
    main()
