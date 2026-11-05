# Cross-domain sentiment analysis: evaluating generalisation across review domains

NLP group project. Three review domains, four models, three experimental settings.

## Abstract

We compare a majority-class baseline, two TF-IDF classifiers and a fine-tuned DistilBERT model on movie, business and product reviews. Inside a domain, DistilBERT reached the highest average macro-F1 at 0.925. Crossing a domain boundary cost the TF-IDF models about 0.065 macro-F1 and DistilBERT about 0.034, so the transformer degraded more gently even though it started higher. Mixed-domain training changed the result by no more than 0.022 macro-F1 in either direction, which is close to the seed-to-seed spread, and it helped only when the target was the product domain.

## 1. Introduction

Online reviews are one of the few places where a business can read what customers actually think, at a volume nobody can read by hand. Sentiment classifiers promise to do that reading automatically, but a classifier learns the vocabulary of the reviews it was trained on. A model trained on film criticism meets a different vocabulary when it reads restaurant complaints. The word unpredictable can praise a plot and describe a defect in a power bank.

This project asks how much performance a sentiment model loses when it crosses a domain boundary, and whether training on more than one domain reduces that loss.

**Research question.** How does moving from within-domain to cross-domain evaluation affect the sentiment classification performance of traditional machine learning and transformer models, and can mixed-domain training reduce this performance gap?

## 2. Data

Three English review datasets, each reduced to a common size so that a larger training pool cannot explain a performance difference.

| Dataset | Domain | Train | Validation | Test | Positive class |
|---|---|---:|---:|---:|---|
| IMDb | movie reviews | 20,000 | 2,000 | 10,000 | rating 7 and above |
| Yelp Polarity | business and service reviews | 20,000 | 2,000 | 10,000 | 3 to 4 stars |
| Amazon Polarity | product reviews | 20,000 | 2,000 | 10,000 | 4 to 5 stars |

Length statistics for the test splits:

| domain   |   rows |   positive_rate |   mean_words |   median_words |   p95_words |
|:---------|-------:|----------------:|-------------:|---------------:|------------:|
| IMDb     |  10000 |             0.5 |        228.8 |            172 |      584.05 |
| Yelp     |  10000 |             0.5 |        133.1 |             98 |      371    |
| Amazon   |  10000 |             0.5 |         74.3 |             66 |      156    |

The three datasets build their labels in different ways. Yelp treats 3 and 4 stars as positive and drops 5-star reviews, so its positive class is milder than IMDb's (7 out of 10) and Amazon's (4 out of 5 stars). This is a property of the source data, not something this project changed, and it matters when comparing absolute scores across domains.

Duplicates are removed on normalised text across all splits and all three datasets, so a review that appears in two datasets is kept once. Screening covers exact matches only. Amazon Polarity spans many product categories, including films and books, so some vocabulary overlap with IMDb survives the screening.

A second pass looked for near-duplicates that exact matching would miss. It builds an eight-shingle signature for each of the 90,000 pooled reviews, verifies candidate pairs with an exact Jaccard score on word 5-grams, and flags pairs at 0.8 or above. It found 11 pairs in total, 0 across datasets and 0 between the train and test split of the same domain. Near-duplicate leakage is therefore negligible at this scale, and no further rows were removed.

Two data artefacts survived cleaning and show up in the error analysis. IMDb reviews carry HTML line-break tags in the source text, and Amazon Polarity contains a small number of non-English reviews. Neither was filtered out, because the task is to evaluate the models as they would meet real data.

## 3. Method

Four models, all trained on the same splits:

| Model | Details |
|---|---|
| Majority class | Predicts the most frequent training label. Floor for comparison. |
| TF-IDF + Logistic Regression | Word 1-2 grams, sublinear term frequency, C tuned per domain on validation. |
| TF-IDF + Linear SVM | Same features, C tuned per domain on validation. |
| DistilBERT | distilbert-base-uncased, max length 256, batch 32, 3 epochs, learning rate 2e-5, best epoch chosen on validation. |

The TF-IDF vectoriser is fitted on training text only. Hyperparameters come from a small grid scored on each domain's validation split. The DistilBERT learning rate was chosen on IMDb validation and applied to every domain. Test splits took no part in training or in hyperparameter selection.

Three experimental settings:

1. Within-domain: train and test on the same domain.
2. Cross-domain: train on one domain, test on each of the other two. No retraining, the within-domain checkpoint is reused.
3. Mixed-domain: train on two domains, test on the third. Training size stays at 20,000 rows (10,000 per source domain) so the comparison isolates domain diversity rather than data volume.

## 4. Results

### 4.1 Within-domain

| model                            |   IMDb |   Yelp |   Amazon |
|:---------------------------------|-------:|-------:|---------:|
| Majority class                   |  0.333 |  0.333 |    0.333 |
| TF-IDF + LR                      |  0.898 |  0.925 |    0.88  |
| TF-IDF + SVM                     |  0.9   |  0.932 |    0.881 |
| DistilBERT                       |  0.902 |  0.949 |    0.925 |
| Average of the three classifiers |  0.9   |  0.935 |    0.895 |

![Within-domain macro-F1](figures/fig_within_domain.png)

DistilBERT is highest on all three domains, but on IMDb the margin over TF-IDF is about two thousandths of a point, which is inside the seed-to-seed spread reported in 4.5. Treat the two as tied on IMDb. IMDb is also the hardest domain in absolute terms, which fits the data: film reviews are long, allusive and often mixed in tone, while product and business reviews tend to state a verdict quickly.

### 4.2 Cross-domain

**TF-IDF + LR**, macro-F1 with the training domain on the row and the test domain on the column. Diagonal cells repeat the within-domain score for reference.

|        |   IMDb |   Yelp |   Amazon |
|:-------|-------:|-------:|---------:|
| IMDb   |  0.898 |  0.855 |    0.82  |
| Yelp   |  0.789 |  0.925 |    0.828 |
| Amazon |  0.851 |  0.883 |    0.88  |

**TF-IDF + SVM**, macro-F1 with the training domain on the row and the test domain on the column. Diagonal cells repeat the within-domain score for reference.

|        |   IMDb |   Yelp |   Amazon |
|:-------|-------:|-------:|---------:|
| IMDb   |  0.9   |  0.847 |    0.816 |
| Yelp   |  0.79  |  0.932 |    0.833 |
| Amazon |  0.864 |  0.888 |    0.881 |

**DistilBERT**, macro-F1 with the training domain on the row and the test domain on the column. Diagonal cells repeat the within-domain score for reference.

|        |   IMDb |   Yelp |   Amazon |
|:-------|-------:|-------:|---------:|
| IMDb   |  0.902 |  0.902 |    0.88  |
| Yelp   |  0.851 |  0.949 |    0.906 |
| Amazon |  0.887 |  0.925 |    0.925 |

![Cross-domain macro-F1](figures/fig_heatmap_macro_f1.png)

![Performance drop](figures/fig_heatmap_drop.png)

For DistilBERT the largest drop is Yelp to IMDb at 0.097 macro-F1, and the smallest is IMDb to Yelp at 0.000.

The loss depends far more on the target domain than on the source. On average DistilBERT loses 0.068 macro-F1 when IMDb is the target and 0.000 when Yelp is. Averaged over sources, Yelp as a training domain costs the most (0.070) and IMDb the least (0.011).

The asymmetry is worth spelling out. A model trained on Yelp loses 0.097 macro-F1 when it reads IMDb, while a model trained on IMDb loses 0.000 when it reads Yelp. Long literary reviews are hard to score with a classifier that learned short consumer verdicts. The reverse direction is almost free, because a model that has seen long, nuanced text handles short verdicts comfortably.

### 4.3 How far apart are the domains?

| domain pair     |   Jensen-Shannon divergence |   top 5000 word overlap |   mean words (first) |   mean words (second) |
|:----------------|----------------------------:|------------------------:|---------------------:|----------------------:|
| IMDb and Yelp   |                      0.2248 |                  0.3459 |                228.8 |                 133.1 |
| IMDb and Amazon |                      0.1455 |                  0.5142 |                228.8 |                  74.3 |
| Yelp and Amazon |                      0.1785 |                  0.4128 |                133.1 |                  74.3 |

![Domain distance](figures/fig_domain_distance.png)

Vocabulary distance does not line up with the size of the transfer loss. IMDb and Amazon have the lowest divergence at 0.146, partly because Amazon reviews cover films and books, yet moving between them still costs DistilBERT 0.022 to 0.039 macro-F1. IMDb and Yelp are the most distant pair at 0.225, and one direction of that transfer costs nothing at all.

Target length tracks the drop better than vocabulary distance does. IMDb averages 229 words, Yelp 133 and Amazon 74, and the models trained on the two short domains lose most when IMDb is the target (0.039 and 0.097 macro-F1). The mismatch between how much text a model expects and how much it gets matters more than which words appear.

### 4.4 Mixed-domain

| model        | trained on    | tested on   |   train rows |   mixed macro-F1 |   best single source |   delta |
|:-------------|:--------------|:------------|-------------:|-----------------:|---------------------:|--------:|
| DistilBERT   | imdb + yelp   | Amazon      |        20000 |            0.913 |                0.906 |   0.007 |
| DistilBERT   | yelp + amazon | IMDb        |        20000 |            0.881 |                0.887 |  -0.006 |
| DistilBERT   | imdb + amazon | Yelp        |        20000 |            0.92  |                0.925 |  -0.005 |
| TF-IDF + LR  | imdb + yelp   | Amazon      |        20000 |            0.844 |                0.828 |   0.016 |
| TF-IDF + LR  | yelp + amazon | IMDb        |        20000 |            0.828 |                0.851 |  -0.022 |
| TF-IDF + LR  | imdb + amazon | Yelp        |        20000 |            0.874 |                0.883 |  -0.009 |
| TF-IDF + SVM | imdb + yelp   | Amazon      |        20000 |            0.847 |                0.833 |   0.014 |
| TF-IDF + SVM | yelp + amazon | IMDb        |        20000 |            0.853 |                0.864 |  -0.011 |
| TF-IDF + SVM | imdb + amazon | Yelp        |        20000 |            0.879 |                0.888 |  -0.009 |

![Mixed-domain against best single source](figures/fig_mixed_vs_single.png)

Mixed-domain training beat the best single source in 3 of 9 model and target combinations, with a mean change of -0.003 macro-F1. Every gain came from the Amazon target: the two TF-IDF models and DistilBERT all improved there. On IMDb and Yelp the mixed models were slightly worse than the best single source.

The pattern is less obvious than it first looks. IMDb is the hardest target in the cross-domain table, yet mixed-domain training does not help there. Amazon gains because it is the most heterogeneous domain: product reviews span books, music, electronics and films, so a second source adds vocabulary instead of noise. IMDb's difficulty comes from long text and mixed tone, and a wider vocabulary does not fix either. Section 4.5 shows that these changes are close to the seed-to-seed spread, so the mixed-domain comparison supports a weak conclusion only.

### 4.5 Robustness across seeds

| domain   | seeds      |   runs |   mean test macro-F1 |    std |    min |    max |
|:---------|:-----------|-------:|---------------------:|-------:|-------:|-------:|
| IMDb     | 42, 43, 44 |      3 |               0.9066 | 0.0043 | 0.9018 | 0.9103 |
| Yelp     | 42, 43, 44 |      3 |               0.9429 | 0.0065 | 0.9358 | 0.9486 |
| Amazon   | 42, 43, 44 |      3 |               0.9278 | 0.0021 | 0.9254 | 0.9294 |

DistilBERT was retrained on each domain with two extra seeds to see how much of the table is noise. The standard deviation across the three runs is at most 0.0065 macro-F1 and the full range is 0.0128. That changes how the rest of the results should be read.

The cross-domain losses of 0.04 to 0.10 are several times larger than the spread, so they stand. The IMDb margin between DistilBERT and TF-IDF SVM is not larger than the spread, so those two models are tied there. The mixed-domain changes of 0.005 to 0.022 are the same order of magnitude as the spread, so the mixed-domain result is weak evidence rather than a firm gain.

### 4.6 What the classifiers key on

| polarity   | domain pair     |   shared terms in top 20 |   Jaccard | shared words                                                                                      |
|:-----------|:----------------|-------------------------:|----------:|:--------------------------------------------------------------------------------------------------|
| positive   | IMDb and Yelp   |                       11 |     0.379 | also, amazing, best, excellent, favorite, fun, great, love, loved, perfect, the best              |
| positive   | IMDb and Amazon |                       12 |     0.429 | amazing, best, enjoyed, excellent, fun, great, is great, love, perfect, the best, well, wonderful |
| positive   | Yelp and Amazon |                       11 |     0.379 | amazing, and, awesome, best, excellent, fun, good, great, love, perfect, the best                 |
| negative   | IMDb and Yelp   |                        9 |     0.29  | awful, bad, horrible, no, nothing, poor, terrible, the worst, worst                               |
| negative   | IMDb and Amazon |                       12 |     0.429 | at all, bad, boring, horrible, no, nothing, poor, poorly, the worst, waste, waste of, worst       |
| negative   | Yelp and Amazon |                       10 |     0.333 | bad, don, horrible, money, no, not, nothing, poor, the worst, worst                               |

Positive vocabulary transfers. Eleven or twelve of the twenty strongest positive terms are shared by every pair of domains, and they are plain sentiment adjectives: great, best, love, excellent, amazing, perfect. Negative vocabulary overlaps less, between nine and twelve terms, and the strongest negative features differ in kind. IMDb's are sentiment words (awful, boring, waste), while Yelp's and Amazon's include negation markers such as not, no and don among their top negative features.

That difference matters for transfer. A model that has learned to read negation as the main negative signal will misfire when the target domain expresses negativity through adjectives instead, and the reverse also holds. It also explains why negation and contrast dominate the error categories in section 5.

## 5. Error analysis

The table below groups the misclassified reviews from all eighteen cross-domain settings using keyword heuristics. A review can be tagged twice, so the shares overlap and do not sum to one. Full per-setting counts are in `results/error_taxonomy.csv`.

| category          |   count |   mean words |   share |
|:------------------|--------:|-------------:|--------:|
| contrast          |    1538 |        201.1 |   0.37  |
| negation          |    1507 |        200   |   0.363 |
| mixed opinion     |     536 |        264.4 |   0.129 |
| long review       |     287 |        486.6 |   0.069 |
| other             |     257 |         49.7 |   0.062 |
| very short review |      31 |          8.3 |   0.007 |

DistilBERT's confusion direction changes with the source domain:

| setting | true negative scored positive | true positive scored negative |
|---|---:|---:|
| amazon to imdb | 59 | 61 |
| amazon to yelp | 58 | 62 |
| imdb to amazon | 97 | 23 |
| imdb to yelp | 74 | 46 |
| yelp to amazon | 48 | 72 |
| yelp to imdb | 22 | 98 |

A model trained on IMDb over-predicts positive on Amazon (97 negative reviews called positive against 23 the other way). IMDb's negative reviews are often long and dramatic, so the model expects strong language before it says negative. Amazon's negative reviews are milder and often contain a compliment, and the model reads that compliment as the verdict.

The opposite happens with a Yelp-trained model on IMDb. It calls 98 positive film reviews negative against 22 the other way, because long emotional prose looks like a rant to a classifier that learned short, blunt Yelp complaints.

![Recall by class and domain](figures/fig_per_class_recall.png)

The recall plot makes the same point per class. Positive recall stays high for every source when the target is Yelp or Amazon, and drops furthest on IMDb, where a Yelp-trained model recovers only 0.83 of the positive reviews. Negative recall is the fragile one for models trained on IMDb and asked to read Amazon.

The most common categories are contrast and negation, both above a third of the tagged errors. A typical failure is a review that praises some aspects and criticises others, with the negative clause carrying the final verdict: the topics in this book are really great, none are covered in any detail unfortunately. Mixed-opinion reviews make up 13 percent. Long reviews make up 7 percent and are almost entirely IMDb samples, where models that truncate at 256 tokens lose the concluding sentence.

Two artefacts deserve a mention. IMDb text keeps its HTML line-break tags, which the tokeniser treats as ordinary words. Amazon Polarity also contains a few non-English reviews, including French and Spanish, that no model in the study was trained to handle. Both are properties of the public datasets rather than mistakes in the pipeline, but they inflate the error counts slightly.

## 6. Limitations

The three datasets use different rules to build their positive class, so part of the cross-domain drop may come from label definition rather than from domain shift. Yelp's positive class sits at 3 to 4 stars, which is weaker than the other two.

The domains are not equally distant. Yelp and Amazon both collect consumer verdicts on a purchase, while IMDb collects long-form film criticism. Treating the three as three points on the same line would be wrong.

Duplicate screening covers exact normalised matches. Near-duplicates with different wording, especially between Amazon's film category and IMDb, are not removed.

Each model was trained once per setting with a single seed, so small differences between models should be read with caution. Training was capped at 20,000 rows to keep the comparison fair, which is well below what DistilBERT can use.

Mixed-domain training holds the total training size constant. That answers whether domain diversity helps at fixed budget, not whether adding more data from a second domain helps in absolute terms.

## 7. Conclusion

Sentiment models do not transfer for free. Averaged over the three classifiers, macro-F1 fell from 0.910 within a domain to 0.856 across domains. DistilBERT started highest and lost the least, which suggests that pretraining buys some robustness to vocabulary shift. The TF-IDF models dropped roughly twice as far.

The direction of transfer matters more than the size of the training set. A model trained on long, nuanced film reviews handled short product and service reviews well. A model trained on short verdicts struggled with film reviews. Mixed-domain training at a fixed budget gave small and uneven changes, concentrated on Amazon, the most heterogeneous domain, and those changes sit close to the seed-to-seed spread.

The practical recommendation for a business is to test a reused model on a sample of the new domain before trusting it, and to expect the largest errors where reviews mix praise and criticism or rely on negation.

## 8. Reproducibility

| Step | Command |
|---|---|
| Environment | `bash scripts/setup_env.sh` |
| Everything | `bash scripts/run_all.sh` |
| Data only | `python src/prepare_data.py` |

Seed 42 is fixed for sampling, splitting and training. Hyperparameters live in `configs/final.yaml`. Every training run writes a JSON record with its settings and metrics.

## References

- Maas, Daly, Pham, Huang, Ng and Potts (2011). Learning word vectors for sentiment analysis. ACL-HLT.
- Zhang, Zhao and LeCun (2015). Character-level convolutional networks for text classification. NIPS.
- Sanh, Debut, Chaumond and Wolf (2019). DistilBERT, a distilled version of BERT. arXiv:1910.01108.
- McAuley and Leskovec (2013). Hidden factors and hidden topics. RecSys.
