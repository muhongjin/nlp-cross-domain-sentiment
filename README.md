# Cross-Domain Sentiment Analysis

Evaluating how well sentiment classifiers generalise across review domains, and whether
mixed-domain training closes the gap.

**Course:** Natural Language Processing — group project (25%)
**Deadline:** 15 November 2026, 23:59

## Research question

> How does moving from within-domain to cross-domain evaluation affect the sentiment
> classification performance of traditional machine learning and transformer models,
> and can mixed-domain training reduce this performance gap?

Supporting questions:

- How do TF-IDF + Logistic Regression and TF-IDF + SVM compare with fine-tuned DistilBERT
  within each domain?
- How much does each model's performance change when tested on a different domain?
- Does training on two domains improve performance on an unseen third domain, compared
  with training on either source domain alone?
- What types of errors explain the observed differences?

## Datasets

| Dataset | Domain | Train | Test | Positive class | Licence |
|---|---|---:|---:|---|---|
| [IMDb](https://huggingface.co/datasets/stanfordnlp/imdb) | Movie reviews | 25,000 | 25,000 | rating ≥ 7/10 | `other` — non-commercial research use |
| [Yelp Polarity](https://huggingface.co/datasets/fancyzhx/yelp_polarity) | Business & service reviews | 560,000 | 38,000 | 3–4 stars (5 stars dropped) | not declared; Yelp Dataset Challenge terms |
| [Amazon Polarity](https://huggingface.co/datasets/fancyzhx/amazon_polarity) | Product reviews | 3,600,000 | 400,000 | 4–5 stars | `apache-2.0` (HF packaging) |

All three are subsampled to a common size — **20,000 train / 2,000 validation / 10,000 test
per domain** — so that differences in performance are not explained by training-set size.

### Known limitations

- **Label construction differs across datasets.** Yelp's positive class is weaker (3–4 stars)
  than IMDb's (≥ 7/10) and Amazon's (4–5 stars). This is treated as a confound, not ignored.
- **Domains are not orthogonal.** Amazon Polarity spans multiple categories including
  Movies & TV and Books, which overlaps semantically with IMDb. Near-duplicate screening
  is applied across datasets and the remaining overlap is reported.
- **Review length differs.** IMDb reviews are much longer than Yelp/Amazon reviews; input
  length is fixed at 256 tokens and length distributions are reported.

### Citation

If you use this code, cite the original dataset papers:

- Maas et al. (2011). *Learning Word Vectors for Sentiment Analysis.* ACL-HLT 2011.
- Zhang, Zhao & LeCun (2015). *Character-level Convolutional Networks for Text Classification.* NIPS 2015.
- McAuley & Leskovec (2013). *Hidden Factors and Hidden Topics.* RecSys 2013.

The HuggingFace repositories `fancyzhx/yelp_polarity` and `fancyzhx/amazon_polarity` are
unofficial mirrors of the Zhang et al. benchmark data.

## Repository structure

```
data/
  raw/            # downloaded source data, not tracked
  processed/      # aligned parquet splits, not tracked (regenerate with src/prepare_data.py)
src/
  prepare_data.py       # Stage 3  — load, align, subsample, dedupe, export splits
  baselines.py          # Stage 4  — majority class, TF-IDF + LR, TF-IDF + SVM
  train_transformer.py  # Stage 5  — fine-tune DistilBERT
  tune.py               # Stage 6  — validation-only hyperparameter search
  run_within.py         # Stage 7  — within-domain experiments
  run_cross.py          # Stage 8  — cross-domain experiments
  run_mixed.py          # Stage 9  — mixed-domain experiments
  evaluate.py           # Stage 10 — metrics and tables
  visualize.py          # Stage 10 — figures
  error_analysis.ipynb  # Stage 11 — error taxonomy
configs/          # hyperparameter configs
results/          # csv / json experiment outputs
figures/          # figures used in the report and video
models/           # checkpoints, not tracked
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Experiment matrix

| Setting | Train | Evaluate | Models |
|---|---|---|---|
| Within-domain | one domain | same domain | 4 |
| Cross-domain | one domain | the other two | 4 |
| Mixed-domain | two domains | the third | 4 |

Cross-domain evaluation reuses the within-domain checkpoints — no retraining per direction.
Total transformer fine-tunes: 3 (within) + 3 (mixed) = 6.

## Status

| Stage | Description | Owner | Status |
|---|---|---|---|
| 1 | Define the problem | — | done (report draft) |
| 2 | Select three datasets | — | done |
| 3 | Explore and prepare data | — | not started |
| 4 | Build baseline models | — | not started |
| 5 | Build the transformer model | — | not started |
| 6 | Tune the models | — | not started |
| 7 | Run within-domain experiments | — | not started |
| 8 | Run cross-domain experiments | — | not started |
| 9 | Test an improvement (mixed-domain) | — | not started |
| 10 | Evaluate and visualise | — | not started |
| 11 | Analyse errors | — | not started |
| 12 | Build a demonstration (Streamlit) | — | not started |
| 13 | Prepare presentation and video | — | not started |
| 14 | Package the submission | — | not started |

## Group contributions

See `REFLECTION.md` (added before submission) for the signed contribution breakdown.
