# Cross-domain sentiment analysis

We trained sentiment classifiers on three kinds of reviews and then tested them on the wrong
one. That's the whole project.

A model that learned from film reviews has to read restaurant complaints. We wanted to know how
much accuracy it loses when that happens, and whether training on two domains helps it cope.

## What the numbers say

| Model | Within a domain | Across domains | Drop |
|---|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.901 | 0.838 | 0.063 |
| TF-IDF + Linear SVM | 0.904 | 0.840 | 0.065 |
| DistilBERT | 0.925 | 0.892 | 0.034 |

DistilBERT is the strongest model and also the one that suffers least. Least still means 0.034
macro-F1 on average. The two TF-IDF models lose about twice that.

The direction of a transfer matters more than the vocabulary distance between domains. A model
trained on Yelp scores 0.851 on IMDb. Flip the pair around and it costs 0.000. Long film
reviews are hard to score if you learned from short ones. Short reviews are easy if you learned
from long ones.

Mixing two domains at a fixed budget didn't reliably help. Three of nine combinations beat the
best single source, all with Amazon as the target, and the gains were 0.007 to 0.016. Retraining
the same model with a different seed moves the score by up to 0.013, so those gains are inside
the noise.

More training data doesn't fix the gap either. Going from 5,000 to 20,000 IMDb rows raised the
within-domain score by 0.009 and left the cross-domain scores flat or slightly lower.

## The three datasets

| Dataset | Domain | Train | Test | Positive class | Licence |
|---|---|---:|---:|---|---|
| [IMDb](https://huggingface.co/datasets/stanfordnlp/imdb) | movie reviews | 25,000 | 25,000 | rating 7 and above | `other`, non-commercial research use |
| [Yelp Polarity](https://huggingface.co/datasets/fancyzhx/yelp_polarity) | business and service reviews | 560,000 | 38,000 | 3 to 4 stars | not declared; Yelp Dataset Challenge terms |
| [Amazon Polarity](https://huggingface.co/datasets/fancyzhx/amazon_polarity) | product reviews | 3,600,000 | 400,000 | 4 to 5 stars | `apache-2.0` (HF packaging) |

We cut each one down to 20,000 training rows, 2,000 validation rows and 10,000 test rows, split
50/50. Equal sizes matter here. Without them a model that saw ten times more training text
would look better for the wrong reason.

Some things to know before trusting any of the numbers. Yelp builds its positive class from 3
and 4 star reviews and throws the 5 star ones away, so its "positive" is milder than IMDb's
(7 of 10) and Amazon's (4 of 5 stars). The three domains aren't equally far apart either: Yelp
and Amazon are both consumer verdicts on a purchase, IMDb is long-form criticism. And Amazon
Polarity spans every product category, films and books included, which means it shares some
vocabulary with IMDb.

Two quirks in the source data survived cleaning and are still there: IMDb reviews carry HTML
line-break tags, and a handful of Amazon reviews are not in English.

### Citing the data

- Maas et al. (2011). Learning Word Vectors for Sentiment Analysis. ACL-HLT.
- Zhang, Zhao and LeCun (2015). Character-level Convolutional Networks for Text Classification. NIPS.
- Sanh et al. (2019). DistilBERT, a distilled version of BERT. arXiv:1910.01108.
- McAuley and Leskovec (2013). Hidden Factors and Hidden Topics. RecSys.

`fancyzhx/yelp_polarity` and `fancyzhx/amazon_polarity` on HuggingFace are unofficial mirrors
of the Zhang et al. benchmark data. Cite the papers, not the mirrors.

## Code

```
src/
  prepare_data.py       loads, aligns, deduplicates and subsamples the three datasets
  baselines.py          majority class, TF-IDF + LR, TF-IDF + SVM
  tune.py               hyperparameter search on validation data
  train_transformer.py  fine-tunes DistilBERT for the within and mixed settings
  run_within.py         collects the within-domain results
  run_cross.py          cross-domain evaluation, plus the error dumps
  run_mixed.py          mixed-domain results and the comparison table
  evaluate.py           pivot tables, written to results/tables/
  visualize.py          every figure in figures/
  error_analysis.py     error taxonomy from the dumps
  error_by_length.py    accuracy by review length
  domain_distance.py    Jensen-Shannon divergence between domains
  near_duplicates.py    near-duplicate screening over train and test
  feature_analysis.py   strongest positive and negative terms per domain
  data_scaling.py       DistilBERT on 5k, 10k and 20k IMDb rows
  run_seeds.py          multi-seed summary
  common.py             paths, constants, metrics
  transformer_utils.py  tokenisation and inference helpers
tests/                  pytest suite for the metric and text helpers
scripts/
  setup_env.sh          recreates the conda environment
  run_all.sh            runs the experiments in order
configs/final.yaml      final hyperparameters
results/                CSV and JSON outputs
figures/                figures used in the report
REPORT.md               the written report
```

The group plan numbers the work in stages 1 to 14. This repository covers 3 to 11, and the
scripts above run in that order.

## Running it

Needs conda, a GPU, and the versions pinned in `requirements.txt`. Torch has to come from the
CUDA index first, which is what `setup_env.sh` does.

```bash
bash scripts/setup_env.sh
python -m pytest tests -q
export CUDA_VISIBLE_DEVICES=1
bash scripts/run_all.sh
```

The full sweep is about 90 minutes on one RTX 3090. Sixteen DistilBERT fine-tunes make up
almost all of it: 2 for the learning rate search, 3 within-domain, 3 mixed-domain, 2 for the
data-scaling run and 6 for the seed check. Everything else finishes in seconds.

The pins aren't decoration. transformers 4.46.3 works with torch 2.5.1, while the 5.x releases need torch 2.6 or newer,
and the server driver caps us at CUDA 12.2.

## Figures

| File | What is in it |
|---|---|
| `figures/fig_within_domain.png` | within-domain macro-F1 by model and domain |
| `figures/fig_heatmap_macro_f1.png` | source by target macro-F1 for all three models |
| `figures/fig_heatmap_drop.png` | DistilBERT's performance drop when transferring |
| `figures/fig_within_vs_cross.png` | average within-domain against cross-domain score |
| `figures/fig_mixed_vs_single.png` | mixed-domain against the best single source |
| `figures/fig_confusion_matrices.png` | DistilBERT cross-domain confusion matrices |
| `figures/fig_per_class_recall.png` | DistilBERT recall on positive and negative reviews |
| `figures/fig_domain_distance.png` | Jensen-Shannon divergence between the three domains |
| `figures/fig_data_scaling.png` | DistilBERT trained on 5k, 10k and 20k IMDb rows |
| `figures/fig_error_by_length.png` | accuracy by review length |
| `figures/fig_length_distribution.png` | review length by domain |

The video, slides and signed contribution breakdown are handed in separately. They're not in this repository.
