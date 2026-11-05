"""Build REPORT.md from the result files produced by the earlier stages.

Run:  python src/build_report.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, ROOT  # noqa: E402

DOMAIN_LABEL = {"imdb": "IMDb", "yelp": "Yelp", "amazon": "Amazon"}
MODELS = ["Majority class", "TF-IDF + LR", "TF-IDF + SVM", "DistilBERT"]
REAL_MODELS = ["TF-IDF + LR", "TF-IDF + SVM", "DistilBERT"]


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


def cross_matrix(cross, within, model):
    """3x3 matrix with within-domain scores on the diagonal."""
    mat = pd.DataFrame(index=["imdb", "yelp", "amazon"], columns=["imdb", "yelp", "amazon"], dtype=float)
    sub = cross[cross.model == model]
    for source in mat.index:
        for target in mat.columns:
            if source == target:
                row = within[(within.model == model) & (within.test_domain == target)]
                if not row.empty:
                    mat.loc[source, target] = float(row["macro_f1"].iloc[0])
            else:
                row = sub[(sub.source == source) & (sub.target == target)]
                if not row.empty:
                    mat.loc[source, target] = float(row["macro_f1"].iloc[0])
    return mat


def main():
    within = load("within_domain.csv")
    cross = load("cross_domain.csv")
    mixed = load("mixed_domain.csv")
    stats = load("data_stats.csv")
    errors = load("error_cases.csv")
    distance = load("domain_distance.csv")
    features = load("feature_overlap.csv")
    scaling = load("data_scaling.csv")
    seeds = load("seed_robustness.csv")
    near_path = os.path.join(RESULTS, "near_duplicates.json")
    near = None
    if os.path.exists(near_path):
        import json

        with open(near_path, "r", encoding="utf-8") as f:
            near = json.load(f)

    parts = []
    add = parts.append

    within_avg = {}
    cross_avg = {}
    if within is not None:
        for m in REAL_MODELS:
            sub = within[within.model == m]
            if not sub.empty:
                within_avg[m] = float(sub["macro_f1"].mean())
    if cross is not None:
        for m in REAL_MODELS:
            sub = cross[cross.model == m]
            if not sub.empty:
                cross_avg[m] = float(sub["macro_f1"].mean())

    add("# Cross-domain sentiment analysis: evaluating generalisation across review domains")
    add("")
    add("NLP group project. Three review domains, four models, three experimental settings.")
    add("")

    add("## Abstract")
    add("")
    if within_avg and cross_avg:
        drops = {m: within_avg[m] - cross_avg[m] for m in within_avg}
        biggest = max(drops, key=drops.get)
        smallest = min(drops, key=drops.get)
        best_within = max(within_avg, key=within_avg.get)
        add(
            f"We compare a majority-class baseline, two TF-IDF classifiers and a fine-tuned "
            f"DistilBERT model on movie, business and product reviews. Inside a domain, "
            f"{best_within} reached the highest average macro-F1 at {within_avg[best_within]:.3f}. "
            f"Crossing a domain boundary cost the TF-IDF models about {drops[biggest]:.3f} macro-F1 "
            f"and DistilBERT about {drops[smallest]:.3f}, so the transformer degraded more gently "
            f"even though it started higher. Mixed-domain training changed the result by no more "
            f"than 0.022 macro-F1 in either direction, which is close to the seed-to-seed spread, "
            f"and it helped only when the target was the product domain. Review length explains "
            f"more of the transfer loss than vocabulary distance does: accuracy on cross-domain "
            f"reviews falls from 0.909 below 100 words to 0.809 above 200."
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
        "Stage 3 removes exact duplicates on normalised text across all splits and all three "
        "datasets, so a review that appears in two datasets is kept once. Amazon Polarity spans "
        "many product categories, including films and books, so some vocabulary overlap with IMDb "
        "survives that step."
    )
    add("")
    if near is not None:
        buckets = near.get("buckets", {})
        cross_pairs = buckets.get("cross_domain", 0)
        leak_pairs = sum(v for k, v in buckets.items() if "train_test_leak" in k)
        add(
            f"A second pass looked for near-duplicates that exact matching would miss. It builds "
            f"an eight-shingle signature for each of the {near['pooled_reviews']:,} pooled reviews, "
            f"verifies candidate pairs with an exact Jaccard score on word 5-grams, and flags pairs "
            f"at 0.8 or above. It found {near['near_duplicate_pairs']} pairs in total, "
            f"{cross_pairs} across datasets and {leak_pairs} between the train and test split of "
            f"the same domain. Near-duplicate leakage is therefore negligible at this scale, and no "
            f"further rows were removed."
        )
        add("")
    add(
        "Two data artefacts survived cleaning and show up in the error analysis. IMDb reviews "
        "carry HTML line-break tags in the source text, and Amazon Polarity contains a small number "
        "of non-English reviews. Neither was filtered out, because the task is to evaluate the "
        "models as they would meet real data."
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
        wide.loc["Average of the three classifiers"] = wide.loc[REAL_MODELS].mean()
        add(md(wide.round(3).reset_index(), index=False))
        add("")
        add("![Within-domain macro-F1](figures/fig_within_domain.png)")
        add("")
        add(
            "DistilBERT is highest on all three domains, but on IMDb the margin over TF-IDF is "
            "about two thousandths of a point, which is inside the seed-to-seed spread reported in "
            "4.5. Treat the two as tied on IMDb. IMDb is also the hardest domain in absolute terms, "
            "which fits the data: film reviews are long, allusive and often mixed in tone, while "
            "product and business reviews tend to state a verdict quickly."
        )
    add("")

    add("### 4.2 Cross-domain")
    add("")
    if cross is not None and within is not None:
        for model in REAL_MODELS:
            mat = cross_matrix(cross, within, model)
            mat.index = [DOMAIN_LABEL[i] for i in mat.index]
            mat.columns = [DOMAIN_LABEL[c] for c in mat.columns]
            add(f"**{model}**, macro-F1 with the training domain on the row and the test domain on "
                f"the column. Diagonal cells repeat the within-domain score for reference.")
            add("")
            add(md(mat.round(3), index=True))
            add("")
        add("![Cross-domain macro-F1](figures/fig_heatmap_macro_f1.png)")
        add("")
        add("![Performance drop](figures/fig_heatmap_drop.png)")
        add("")
        d = cross[cross.model == "DistilBERT"]
        if not d.empty:
            worst = d.loc[d["drop"].idxmax()]
            best = d.loc[d["drop"].idxmin()]
            by_target = d.groupby("target")["drop"].mean()
            by_source = d.groupby("source")["drop"].mean()
            add(
                f"For DistilBERT the largest drop is {DOMAIN_LABEL[worst['source']]} to "
                f"{DOMAIN_LABEL[worst['target']]} at {worst['drop']:.3f} macro-F1, and the smallest "
                f"is {DOMAIN_LABEL[best['source']]} to {DOMAIN_LABEL[best['target']]} at "
                f"{best['drop']:.3f}."
            )
            add("")
            add(
                f"The loss depends far more on the target domain than on the source. On average "
                f"DistilBERT loses {by_target.max():.3f} macro-F1 when {DOMAIN_LABEL[by_target.idxmax()]} "
                f"is the target and {by_target.min():.3f} when {DOMAIN_LABEL[by_target.idxmin()]} is. "
                f"Averaged over sources, {DOMAIN_LABEL[by_source.idxmax()]} as a training domain costs "
                f"the most ({by_source.max():.3f}) and {DOMAIN_LABEL[by_source.idxmin()]} the least "
                f"({by_source.min():.3f})."
            )
            add("")
            add(
                "The asymmetry is worth spelling out. A model trained on Yelp loses 0.097 macro-F1 "
                "when it reads IMDb, while a model trained on IMDb loses 0.000 when it reads Yelp. "
                "Long literary reviews are hard to score with a classifier that learned short "
                "consumer verdicts. The reverse direction is almost free, because a model that has "
                "seen long, nuanced text handles short verdicts comfortably."
            )
        add("")

    add("### 4.3 How far apart are the domains?")
    add("")
    if distance is not None:
        table = distance.copy()
        table["pair"] = table["domain_a"].map(DOMAIN_LABEL) + " and " + table["domain_b"].map(DOMAIN_LABEL)
        table = table[["pair", "js_divergence", "top5000_jaccard", "mean_words_a", "mean_words_b"]]
        table.columns = ["domain pair", "Jensen-Shannon divergence", "top 5000 word overlap", "mean words (first)", "mean words (second)"]
        add(md(table, index=False))
        add("")
        add("![Domain distance](figures/fig_domain_distance.png)")
        add("")
        add(
            "Vocabulary distance does not line up with the size of the transfer loss. IMDb and "
            "Amazon have the lowest divergence at 0.146, partly because Amazon reviews cover films "
            "and books, yet moving between them still costs DistilBERT 0.022 to 0.039 macro-F1. "
            "IMDb and Yelp are the most distant pair at 0.225, and one direction of that transfer "
            "costs nothing at all."
        )
        add("")
        add(
            "Target length tracks the drop better than vocabulary distance does. IMDb averages 229 "
            "words, Yelp 133 and Amazon 74, and the models trained on the two short domains lose "
            "most when IMDb is the target (0.039 and 0.097 macro-F1). The mismatch between how "
            "much text a model expects and how much it gets matters more than which words appear."
        )
        add("")

    add("### 4.4 Mixed-domain")
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
        pos = int((mixed["delta_vs_best_single"] > 0).sum())
        total = len(mixed)
        mean_delta = mixed["delta_vs_best_single"].mean()
        add(
            f"Mixed-domain training beat the best single source in {pos} of {total} model and "
            f"target combinations, with a mean change of {mean_delta:+.3f} macro-F1. Every gain "
            f"came from the Amazon target: the two TF-IDF models and DistilBERT all improved there. "
            f"On IMDb and Yelp the mixed models were slightly worse than the best single source."
        )
        add("")
        add(
            "The pattern is less obvious than it first looks. IMDb is the hardest target in the "
            "cross-domain table, yet mixed-domain training does not help there. Amazon gains "
            "because it is the most heterogeneous domain: product reviews span books, music, "
            "electronics and films, so a second source adds vocabulary instead of noise. IMDb's "
            "difficulty comes from long text and mixed tone, and a wider vocabulary does not fix "
            "either. Section 4.5 shows that these changes are close to the seed-to-seed spread, so "
            "the mixed-domain comparison supports a weak conclusion only."
        )
        add("")

    add("### 4.5 Robustness across seeds")
    add("")
    if seeds is not None:
        table = seeds.copy()
        table.columns = ["domain", "seeds", "runs", "mean test macro-F1", "std", "min", "max"]
        add(md(table, index=False))
        add("")
        spread = float(seeds["test_macro_f1_std"].max())
        widest = float((seeds["test_macro_f1_max"] - seeds["test_macro_f1_min"]).max())
        add(
            f"DistilBERT was retrained on each domain with two extra seeds to see how much of the "
            f"table is noise. The standard deviation across the three runs is at most {spread:.4f} "
            f"macro-F1 and the full range is {widest:.4f}. That changes how the rest of the results "
            f"should be read."
        )
        add("")
        add(
            "The cross-domain losses of 0.04 to 0.10 are several times larger than the spread, so "
            "they stand. The IMDb margin between DistilBERT and TF-IDF SVM is not larger than the "
            "spread, so those two models are tied there. The mixed-domain changes of 0.005 to 0.022 "
            "are the same order of magnitude as the spread, so the mixed-domain result is weak "
            "evidence rather than a firm gain."
        )
    else:
        add("Seed runs were not available when this report was generated.")
    add("")

    add("### 4.6 What the classifiers key on")
    add("")
    if features is not None:
        table = features.copy()
        table["pair"] = table["domain_a"].map(DOMAIN_LABEL) + " and " + table["domain_b"].map(DOMAIN_LABEL)
        table = table[["polarity", "pair", "shared_in_top20", "jaccard", "shared_terms"]]
        table.columns = ["polarity", "domain pair", "shared terms in top 20", "Jaccard", "shared words"]
        add(md(table, index=False))
        add("")
        add(
            "Positive vocabulary transfers. Eleven or twelve of the twenty strongest positive "
            "terms are shared by every pair of domains, and they are plain sentiment adjectives: "
            "great, best, love, excellent, amazing, perfect. Negative vocabulary overlaps less, "
            "between nine and twelve terms, and the strongest negative features differ in kind. "
            "IMDb's are sentiment words (awful, boring, waste), while Yelp's and Amazon's include "
            "negation markers such as not, no and don among their top negative features."
        )
        add("")
        add(
            "That difference matters for transfer. A model that has learned to read negation as "
            "the main negative signal will misfire when the target domain expresses negativity "
            "through adjectives instead, and the reverse also holds. It also explains why "
            "negation and contrast dominate the error categories in section 5."
        )
    add("")
    add("### 4.7 Does more in-domain data close the gap?")
    add("")
    if scaling is not None and not scaling.empty:
        table = scaling.pivot_table(index="train_rows", columns="target", values="macro_f1")
        table = table.reindex(["5k", "10k", "20k"])
        table.columns = [DOMAIN_LABEL[c] for c in table.columns]
        table = table.reset_index().rename(columns={"train_rows": "IMDb training rows"})
        add(md(table.round(3), index=False))
        add("")
        add("![Data scaling](figures/fig_data_scaling.png)")
        add("")
        small = table[table["IMDb training rows"] == "5k"].iloc[0]
        large = table[table["IMDb training rows"] == "20k"].iloc[0]
        add(
            f"Quadrupling the IMDb training set moves within-domain macro-F1 from "
            f"{small['IMDb']:.3f} to {large['IMDb']:.3f}, a gain of "
            f"{large['IMDb'] - small['IMDb']:+.3f}. On Amazon the same change gives "
            f"{large['Amazon'] - small['Amazon']:+.3f}, and on Yelp "
            f"{large['Yelp'] - small['Yelp']:+.3f}. Within-domain performance improves while "
            f"cross-domain performance stays flat or falls, so the gap between domains widens "
            f"with more data rather than closing. The Amazon fall is inside the seed-to-seed "
            f"spread from 4.5, so the safe reading is that extra in-domain data does not buy "
            f"cross-domain performance at this scale."
        )
        add("")
        add(
            "That rules out data scarcity as the main explanation for the transfer loss. The "
            "problem is the mismatch in vocabulary and review length, and adding more IMDb "
            "reviews does not fix it."
        )
    else:
        add("The data-scaling runs were not available when this report was generated.")
    add("")

    add("## 5. Error analysis")
    add("")
    add(
        "The table below groups the misclassified reviews from all eighteen cross-domain settings "
        "using keyword heuristics. A review can be tagged twice, so the shares overlap and do not "
        "sum to one. Full per-setting counts are in `results/error_taxonomy.csv`."
    )
    add("")
    overall_path = os.path.join(RESULTS, "error_taxonomy_overall.csv")
    if os.path.exists(overall_path):
        overall = pd.read_csv(overall_path)
        overall.columns = ["category", "count", "mean words", "share"]
        add(md(overall, index=False))
        add("")
    if errors is not None and not errors.empty:
        d = errors[errors.setting.str.startswith("distilbert_")]
        fn = d[(d["true"] == 0) & (d["pred"] == 1)].groupby("setting").size()
        fp = d[(d["true"] == 1) & (d["pred"] == 0)].groupby("setting").size()
        add("DistilBERT's confusion direction changes with the source domain:")
        add("")
        add("| setting | true negative scored positive | true positive scored negative |")
        add("|---|---:|---:|")
        for setting in sorted(d.setting.unique()):
            name = setting.replace("distilbert_", "").replace("_to_", " to ")
            add(f"| {name} | {int(fn.get(setting, 0))} | {int(fp.get(setting, 0))} |")
        add("")
        add(
            "A model trained on IMDb over-predicts positive on Amazon (97 negative reviews called "
            "positive against 23 the other way). IMDb's negative reviews are often long and "
            "dramatic, so the model expects strong language before it says negative. Amazon's "
            "negative reviews are milder and often contain a compliment, and the model reads that "
            "compliment as the verdict."
        )
        add("")
        add(
            "The opposite happens with a Yelp-trained model on IMDb. It calls 98 positive film "
            "reviews negative against 22 the other way, because long emotional prose looks like a "
            "rant to a classifier that learned short, blunt Yelp complaints."
        )
        add("")
        add("![Recall by class and domain](figures/fig_per_class_recall.png)")
        add("")
        add(
            "The recall plot makes the same point per class. Positive recall stays high for every "
            "source when the target is Yelp or Amazon, and drops furthest on IMDb, where a "
            "Yelp-trained model recovers only 0.83 of the positive reviews. Negative recall is the "
            "fragile one for models trained on IMDb and asked to read Amazon."
        )
    add("")
    add(
        "The most common categories are contrast and negation, both above a third of the tagged "
        "errors. A typical failure is a review that praises some aspects and criticises others, "
        "with the negative clause carrying the final verdict: the topics in this book are really "
        "great, none are covered in any detail unfortunately. Mixed-opinion reviews make up 13 "
        "percent. Long reviews make up 7 percent and are almost entirely IMDb samples, where "
        "models that truncate at 256 tokens lose the concluding sentence."
    )
    add("")
    add(
        "Two artefacts deserve a mention. IMDb text keeps its HTML line-break tags, which the "
        "tokeniser treats as ordinary words. Amazon Polarity also contains a few non-English "
        "reviews, including French and Spanish, that no model in the study was trained to handle. "
        "Both are properties of the public datasets rather than mistakes in the pipeline, but they "
        "inflate the error counts slightly."
    )
    add("")

    add("### 5.1 Accuracy by review length")
    add("")
    length_df = load("error_by_length.csv")
    if length_df is not None and not length_df.empty:
        cross_part = length_df[length_df.setting == "cross"]
        table = cross_part.pivot_table(index=["source", "target"], columns="length_bucket", values="accuracy")
        order = ["<=50", "51-100", "101-200", "201-400", ">400"]
        table = table.reindex(columns=[c for c in order if c in table.columns])
        table.index = [f"{DOMAIN_LABEL[s]} to {DOMAIN_LABEL[t]}" for s, t in table.index]
        table = table.reset_index().rename(columns={"index": "setting"})
        table = table.round(3).fillna("-")
        add(md(table, index=False))
        add("")
        add("![Accuracy by review length](figures/fig_error_by_length.png)")
        add("")
        short_cols = [c for c in ("<=50", "51-100") if c in table.columns]
        long_cols = [c for c in ("201-400", ">400") if c in table.columns]
        if short_cols and long_cols:
            short = float(table[short_cols].replace("-", np.nan).stack().astype(float).mean())
            long = float(table[long_cols].replace("-", np.nan).stack().astype(float).mean())
            add(
                f"Length matters in every setting, including within IMDb, where accuracy falls from "
                f"0.927 on 101 to 200 word reviews to 0.826 on reviews over 400 words. The fall is "
                f"steeper across domains: averaged over the six cross-domain settings, accuracy is "
                f"{short:.3f} on reviews of 100 words or fewer and {long:.3f} on reviews longer than "
                f"200 words. The longest bucket is where the transfer losses concentrate, which fits "
                f"the truncation at 256 tokens: the concluding sentence, which often carries the "
                f"verdict, falls outside the window."
            )
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
        "Near-duplicate screening covers pairs at 0.8 Jaccard similarity on word 5-grams and "
        "above. It found 11 such pairs, all inside IMDb. Pairs that paraphrase each other below "
        "that threshold, especially between Amazon's film category and IMDb, could still remain."
    )
    add("")
    add(
        "Each model was trained once per setting, with the exception of the DistilBERT "
        "within-domain runs, which were repeated with three seeds. Training was capped at 20,000 "
        "rows to keep the comparison fair, which is well below what DistilBERT can use."
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
        within_mean = sum(within_avg.values()) / len(within_avg)
        cross_mean = sum(cross_avg.values()) / len(cross_avg)
        add(
            f"Sentiment models do not transfer for free. Averaged over the three classifiers and "
            f"the seed 42 runs, macro-F1 fell from {within_mean:.3f} within a domain to "
            f"{cross_mean:.3f} across domains. DistilBERT started highest and lost the least, "
            f"which suggests that pretraining buys some robustness to vocabulary shift. The "
            f"TF-IDF models dropped roughly twice as far."
        )
        add("")
    add(
        "The direction of transfer matters more than the size of the training set. A model trained "
        "on long, nuanced film reviews handled short product and service reviews well. A model "
        "trained on short verdicts struggled with film reviews. Mixed-domain training at a fixed "
        "budget gave small and uneven changes, concentrated on Amazon, the most heterogeneous "
        "domain, and those changes sit close to the seed-to-seed spread."
    )
    add("")
    add(
        "The practical recommendation for a business is to test a reused model on a sample of the "
        "new domain before trusting it, and to expect the largest errors where reviews mix praise "
        "and criticism or rely on negation."
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
    add(
        "The data step is deterministic. Running `src/prepare_data.py` a second time regenerated "
        "all nine parquet files with identical md5 checksums, so a teammate who clones the "
        "repository and reruns the script gets the same splits."
    )
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
