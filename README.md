# Cross-domain sentiment analysis

Evaluating how well sentiment classifiers generalise across review domains, and whether
mixed-domain training closes the gap.

**Course:** Natural Language Processing, group project (25%)
**Deadline:** 15 November 2026

## Research question

> How does moving from within-domain to cross-domain evaluation affect the sentiment
> classification performance of traditional machine learning and transformer models,
> and can mixed-domain training reduce this performance gap?

## What the experiments found

Four models (majority class, TF-IDF + Logistic Regression, TF-IDF + Linear SVM, fine-tuned
DistilBERT) were trained on three review domains of 20,000 rows each and evaluated in three
settings.

| Setting | What it does |
|---|---|
| Within-domain | train and test on the same domain |
| Cross-domain | train on one domain, test on the other two, no retraining |
| Mixed-domain | train on two domains, test on the third, training size held constant |

Headline numbers from `REPORT.md`:

| Model | Within-domain macro-F1 | Cross-domain macro-F1 | Average drop |
|---|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.901 | 0.838 | 0.063 |
| TF-IDF + Linear SVM | 0.904 | 0.840 | 0.065 |
| DistilBERT | 0.925 | 0.892 | 0.034 |

DistilBERT starts highest and degrades least. Transfer is asymmetric: a Yelp-trained model
loses 0.097 macro-F1 on IMDb, while an IMDb-trained model loses 0.000 on Yelp. Mixed-domain
training changed scores by no more than 0.022 macro-F1 either way, and every gain landed on
the Amazon target.

Two extra checks support the numbers. Near-duplicate screening over all 90,000 pooled
reviews found 11 near-duplicate pairs, none across datasets and none between a train and
test split, so the comparison is not inflated by leakage. Retraining DistilBERT with two
additional seeds gives a standard deviation of at most 0.007 macro-F1 (full range 0.013),
which is small next to the cross-domain gaps but large enough that the 0.002 IMDb margin
between DistilBERT and TF-IDF SVM should be read as a tie.

Vocabulary distance between domains does not explain the losses. IMDb and Amazon have the
lowest Jensen-Shannon divergence (0.146) yet still cost each other 0.022 to 0.039 macro-F1.
Review length tracks the drop more closely: IMDb averages 229 words, Yelp 133 and Amazon 74,
and the short-domain models suffer most when IMDb is the target.

A data-scaling run trains DistilBERT on 5,000, 10,000 and 20,000 IMDb rows. Within-domain
macro-F1 rises from 0.893 to 0.902, while the cross-domain scores stay flat or fall, so more
in-domain data does not close the gap. The data pipeline is deterministic: rerunning
`src/prepare_data.py` regenerates all nine parquet files with identical md5 checksums.

## Datasets

| Dataset | Domain | Train | Test | Positive class | Licence |
|---|---|---:|---:|---|---|
| [IMDb](https://huggingface.co/datasets/stanfordnlp/imdb) | Movie reviews | 25,000 | 25,000 | rating 7 and above | `other`, non-commercial research use |
| [Yelp Polarity](https://huggingface.co/datasets/fancyzhx/yelp_polarity) | Business and service reviews | 560,000 | 38,000 | 3 to 4 stars | not declared; Yelp Dataset Challenge terms |
| [Amazon Polarity](https://huggingface.co/datasets/fancyzhx/amazon_polarity) | Product reviews | 3,600,000 | 400,000 | 4 to 5 stars | `apache-2.0` (HF packaging) |

Each domain is subsampled to 20,000 train, 2,000 validation and 10,000 test rows, balanced
50/50, so that a larger training pool cannot explain a performance difference.

### Known limitations

- The datasets build the positive class differently. Yelp's positive class is weaker
  (3 to 4 stars) than IMDb's (7 of 10) and Amazon's (4 of 5 stars).
- The domains are not equally distant. Yelp and Amazon both collect consumer verdicts,
  while IMDb collects long-form film criticism.
- Amazon Polarity spans many categories including films and books, so some vocabulary
  overlaps with IMDb. Exact duplicates are removed; near-duplicates are not.
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
  prepare_data.py       Stage 3  load, align, deduplicate, subsample
  baselines.py          Stage 4  majority class, TF-IDF + LR, TF-IDF + SVM
  train_transformer.py  Stages 5 and 9  fine-tune DistilBERT
  tune.py               Stage 6  validation-only hyperparameter search
  run_within.py         Stage 7  collect within-domain results
  run_cross.py          Stage 8  cross-domain evaluation
  run_mixed.py          Stage 9  mixed-domain training and comparison
  evaluate.py           Stage 10 result tables
  visualize.py          Stage 10 figures
  error_analysis.py     Stage 11 error taxonomy
  domain_distance.py    vocabulary and length distance between domains
  near_duplicates.py    near-duplicate screening over the pooled corpus
  run_seeds.py          multi-seed robustness summary
  build_report.py       assemble REPORT.md from the result files
scripts/
  setup_env.sh          recreate the conda environment
  run_all.sh            reproduce every stage in order
configs/                final hyperparameters
results/                CSV and JSON outputs
figures/                figures used in the report
REPORT.md               written report
AGENTS.md               project rules and progress log
```

## Setup

On the GPU server, with conda available:

```bash
bash scripts/setup_env.sh
```

Pinned versions: torch 2.5.1+cu121, transformers 4.46.3, datasets 3.2.0, Python 3.10.

## Reproduce

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nlp
export CUDA_VISIBLE_DEVICES=1
bash scripts/run_all.sh
```

The full run takes about 90 minutes on a single RTX 3090, dominated by 14 DistilBERT
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
| `figures/fig_length_distribution.png` | review length by domain |

## Group contributions

See `REFLECTION.md`, added before submission, for the signed contribution breakdown.
