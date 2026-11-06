# Cross-domain sentiment analysis

Group project for the NLP course. We train sentiment classifiers on movie, business and
product reviews and measure how much accuracy they lose when the test reviews come from a
different domain.

## Research question

How does moving from within-domain to cross-domain evaluation affect the sentiment
classification performance of traditional machine learning and transformer models, and can
mixed-domain training reduce that performance gap?

## What we found

Four models (majority class, TF-IDF + Logistic Regression, TF-IDF + Linear SVM, fine-tuned
DistilBERT) were trained on three review domains of 20,000 rows each and evaluated in three
settings.

| Setting | What it does |
|---|---|
| Within-domain | train and test on the same domain |
| Cross-domain | train on one domain, test on the other two, no retraining |
| Mixed-domain | train on two domains, test on the third, training size held constant |

Headline numbers, from `REPORT.md`:

| Model | Within-domain macro-F1 | Cross-domain macro-F1 | Average drop |
|---|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.901 | 0.838 | 0.063 |
| TF-IDF + Linear SVM | 0.904 | 0.840 | 0.065 |
| DistilBERT | 0.925 | 0.892 | 0.034 |

DistilBERT starts highest and degrades least. Transfer is asymmetric: a Yelp-trained model
loses 0.097 macro-F1 on IMDb, while an IMDb-trained model loses 0.000 on Yelp.

Two checks limit how far these numbers can be pushed. Near-duplicate screening over the 90,000
training and test reviews found 11 pairs, all inside IMDb, none across datasets and none
between a train and test split. Retraining DistilBERT with two extra seeds moves the test score
by up to 0.013 macro-F1, which is enough that the 0.002 IMDb margin between DistilBERT and
TF-IDF SVM should be read as a tie, and enough that the mixed-domain changes (0.005 to 0.022)
are not conclusive.

Vocabulary distance between domains does not explain the losses. IMDb and Amazon have the
lowest Jensen-Shannon divergence (0.146) yet still cost each other 0.022 to 0.039 macro-F1.
Review length tracks the drop more closely: IMDb averages 229 words, Yelp 133 and Amazon 74,
and the models trained on the short domains suffer most when IMDb is the target.

A data-scaling run trains DistilBERT on 5,000, 10,000 and 20,000 IMDb rows. Within-domain
macro-F1 rises from 0.893 to 0.902 while the cross-domain scores stay flat or fall, so more
in-domain data does not close the gap at this scale.

## Datasets

| Dataset | Domain | Train | Test | Positive class | Licence |
|---|---|---:|---:|---|---|
| [IMDb](https://huggingface.co/datasets/stanfordnlp/imdb) | movie reviews | 25,000 | 25,000 | rating 7 and above | `other`, non-commercial research use |
| [Yelp Polarity](https://huggingface.co/datasets/fancyzhx/yelp_polarity) | business and service reviews | 560,000 | 38,000 | 3 to 4 stars | not declared; Yelp Dataset Challenge terms |
| [Amazon Polarity](https://huggingface.co/datasets/fancyzhx/amazon_polarity) | product reviews | 3,600,000 | 400,000 | 4 to 5 stars | `apache-2.0` (HF packaging) |

Each domain is subsampled to 20,000 train, 2,000 validation and 10,000 test rows, balanced
50/50.

### Known limitations

- The datasets build the positive class differently. Yelp's positive class is weaker (3 to 4
  stars) than IMDb's (7 of 10) and Amazon's (4 of 5 stars).
- The domains are not equally distant. Yelp and Amazon both collect consumer verdicts, while
  IMDb collects long-form film criticism.
- Amazon Polarity spans many categories including films and books, so some vocabulary overlaps
  with IMDb.
- IMDb text keeps HTML line-break tags, and Amazon Polarity contains a few non-English
  reviews. Both were left in place.

### Citation

- Maas et al. (2011). Learning Word Vectors for Sentiment Analysis. ACL-HLT.
- Zhang, Zhao and LeCun (2015). Character-level Convolutional Networks for Text Classification. NIPS.
- Sanh et al. (2019). DistilBERT, a distilled version of BERT. arXiv:1910.01108.
- McAuley and Leskovec (2013). Hidden Factors and Hidden Topics. RecSys.

The HuggingFace repositories `fancyzhx/yelp_polarity` and `fancyzhx/amazon_polarity` are
unofficial mirrors of the Zhang et al. benchmark data.

## Repository layout

```
src/
  prepare_data.py       load, align, deduplicate and subsample the three datasets
  baselines.py          majority class, TF-IDF + LR, TF-IDF + SVM
  tune.py               hyperparameter search on validation data
  train_transformer.py  fine-tune DistilBERT (within and mixed settings)
  run_within.py         collect within-domain results
  run_cross.py          cross-domain evaluation and error dumps
  run_mixed.py          mixed-domain results and comparison
  evaluate.py           pivot tables written to results/tables/
  visualize.py          every figure in figures/
  error_analysis.py     error taxonomy from the dumps
  error_by_length.py    accuracy by review length
  domain_distance.py    Jensen-Shannon divergence between domains
  near_duplicates.py    near-duplicate screening over train and test
  feature_analysis.py   strongest positive and negative terms per domain
  data_scaling.py       DistilBERT trained on 5k, 10k and 20k IMDb rows
  run_seeds.py          multi-seed summary
  common.py             paths, constants, metrics
  transformer_utils.py  tokenisation and inference helpers
tests/                  pytest suite for the metric and text helpers
scripts/
  setup_env.sh          recreate the conda environment
  run_all.sh            reproduce the experiments in order
configs/final.yaml      final hyperparameters
results/                CSV and JSON outputs
figures/                figures used in the report
REPORT.md               written report
```

The group plan numbers the work in stages 1 to 14. This repository covers stages 3 to 11; the
script names above follow that order.

## Setup

On the GPU server, with conda available:

```bash
bash scripts/setup_env.sh
```

Pinned versions: torch 2.5.1+cu121, transformers 4.46.3, datasets 3.2.0, Python 3.10. The pins
matter: newer transformers releases need torch 2.6 or later.

## Tests

```bash
python -m pytest tests -q
```

The suite covers metric computation, the text normalisation and duplicate keys used when
building the splits, and the shingle and similarity helpers in the near-duplicate check.

## Reproduce

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nlp
export CUDA_VISIBLE_DEVICES=1
bash scripts/run_all.sh
```

The full run takes about 90 minutes on a single RTX 3090, dominated by 16 DistilBERT
fine-tunes: 2 for the learning-rate search, 3 within-domain, 3 mixed-domain, 2 for the
data-scaling ablation and 6 for the seed check. Everything else takes seconds.

## Figures

| Figure | What it shows |
|---|---|
| `figures/fig_within_domain.png` | within-domain macro-F1 by model and domain |
| `figures/fig_heatmap_macro_f1.png` | source by target macro-F1 for all three models |
| `figures/fig_heatmap_drop.png` | DistilBERT performance drop when transferring |
| `figures/fig_within_vs_cross.png` | average within-domain against cross-domain score |
| `figures/fig_mixed_vs_single.png` | mixed-domain against the best single source |
| `figures/fig_confusion_matrices.png` | DistilBERT cross-domain confusion matrices |
| `figures/fig_per_class_recall.png` | DistilBERT recall on positive and negative reviews |
| `figures/fig_domain_distance.png` | Jensen-Shannon divergence between the three domains |
| `figures/fig_data_scaling.png` | DistilBERT trained on 5k, 10k and 20k IMDb rows |
| `figures/fig_error_by_length.png` | accuracy by review length |
| `figures/fig_length_distribution.png` | review length by domain |

## Group contributions

The signed contribution breakdown is submitted separately as a PDF, as the course requires.
