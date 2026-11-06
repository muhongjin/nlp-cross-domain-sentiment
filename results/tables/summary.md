# Result tables

## Within-domain macro-F1

| model                            | imdb   | yelp   | amazon   |
|:---------------------------------|:-------|:-------|:---------|
| Majority class                   | 0.333  | 0.333  | 0.333    |
| TF-IDF + LR                      | 0.898  | 0.925  | 0.879    |
| TF-IDF + SVM                     | 0.900  | 0.932  | 0.881    |
| DistilBERT                       | 0.902  | 0.949  | 0.925    |
| Average of the three classifiers | 0.900  | 0.935  | 0.895    |

## Cross-domain results

| setting   | model        | source   | target   | accuracy   | macro_f1   | within_macro_f1   | drop   |
|:----------|:-------------|:---------|:---------|:-----------|:-----------|:------------------|:-------|
| cross     | DistilBERT   | amazon   | imdb     | 0.887      | 0.887      | 0.925             | 0.039  |
| cross     | DistilBERT   | amazon   | yelp     | 0.925      | 0.925      | 0.925             | 0.001  |
| cross     | DistilBERT   | imdb     | amazon   | 0.880      | 0.880      | 0.902             | 0.022  |
| cross     | DistilBERT   | imdb     | yelp     | 0.902      | 0.901      | 0.902             | 0.000  |
| cross     | DistilBERT   | yelp     | amazon   | 0.906      | 0.906      | 0.949             | 0.043  |
| cross     | DistilBERT   | yelp     | imdb     | 0.852      | 0.851      | 0.949             | 0.097  |
| cross     | TF-IDF + LR  | amazon   | imdb     | 0.851      | 0.851      | 0.879             | 0.029  |
| cross     | TF-IDF + LR  | amazon   | yelp     | 0.883      | 0.883      | 0.879             | -0.004 |
| cross     | TF-IDF + LR  | imdb     | amazon   | 0.821      | 0.820      | 0.898             | 0.078  |
| cross     | TF-IDF + LR  | imdb     | yelp     | 0.855      | 0.855      | 0.898             | 0.044  |
| cross     | TF-IDF + LR  | yelp     | amazon   | 0.829      | 0.828      | 0.925             | 0.096  |
| cross     | TF-IDF + LR  | yelp     | imdb     | 0.792      | 0.789      | 0.925             | 0.136  |
| cross     | TF-IDF + SVM | amazon   | imdb     | 0.864      | 0.864      | 0.881             | 0.017  |
| cross     | TF-IDF + SVM | amazon   | yelp     | 0.888      | 0.888      | 0.881             | -0.007 |
| cross     | TF-IDF + SVM | imdb     | amazon   | 0.816      | 0.816      | 0.900             | 0.084  |
| cross     | TF-IDF + SVM | imdb     | yelp     | 0.848      | 0.847      | 0.900             | 0.052  |
| cross     | TF-IDF + SVM | yelp     | amazon   | 0.834      | 0.833      | 0.932             | 0.100  |
| cross     | TF-IDF + SVM | yelp     | imdb     | 0.793      | 0.790      | 0.932             | 0.143  |

### DistilBERT: macro-F1 by source and target

| source   | imdb   | yelp   | amazon   |
|:---------|:-------|:-------|:---------|
| imdb     | -      | 0.901  | 0.880    |
| yelp     | 0.851  | -      | 0.906    |
| amazon   | 0.887  | 0.925  | -        |

### TF-IDF + LR: macro-F1 by source and target

| source   | imdb   | yelp   | amazon   |
|:---------|:-------|:-------|:---------|
| imdb     | -      | 0.855  | 0.820    |
| yelp     | 0.789  | -      | 0.828    |
| amazon   | 0.851  | 0.883  | -        |

### TF-IDF + SVM: macro-F1 by source and target

| source   | imdb   | yelp   | amazon   |
|:---------|:-------|:-------|:---------|
| imdb     | -      | 0.847  | 0.816    |
| yelp     | 0.790  | -      | 0.833    |
| amazon   | 0.864  | 0.888  | -        |

## Mixed-domain results

| setting   | model        | train_domains   | target   | train_size   | accuracy   | macro_f1   | best_single_source_f1   | delta_vs_best_single   |
|:----------|:-------------|:----------------|:---------|:-------------|:-----------|:-----------|:------------------------|:-----------------------|
| mixed     | DistilBERT   | imdb+yelp       | amazon   | 20000        | 0.913      | 0.913      | 0.906                   | 0.007                  |
| mixed     | DistilBERT   | yelp+amazon     | imdb     | 20000        | 0.881      | 0.881      | 0.887                   | -0.006                 |
| mixed     | DistilBERT   | imdb+amazon     | yelp     | 20000        | 0.920      | 0.920      | 0.925                   | -0.005                 |
| mixed     | TF-IDF + LR  | imdb+yelp       | amazon   | 20000        | 0.844      | 0.844      | 0.828                   | 0.016                  |
| mixed     | TF-IDF + LR  | yelp+amazon     | imdb     | 20000        | 0.829      | 0.828      | 0.851                   | -0.022                 |
| mixed     | TF-IDF + LR  | imdb+amazon     | yelp     | 20000        | 0.874      | 0.874      | 0.883                   | -0.009                 |
| mixed     | TF-IDF + SVM | imdb+yelp       | amazon   | 20000        | 0.847      | 0.847      | 0.833                   | 0.014                  |
| mixed     | TF-IDF + SVM | yelp+amazon     | imdb     | 20000        | 0.853      | 0.853      | 0.864                   | -0.011                 |
| mixed     | TF-IDF + SVM | imdb+amazon     | yelp     | 20000        | 0.879      | 0.879      | 0.888                   | -0.009                 |

## Per-class metrics, DistilBERT cross-domain

| source   | target   | positive_precision   | positive_recall   | negative_precision   | negative_recall   |
|:---------|:---------|:---------------------|:------------------|:---------------------|:------------------|
| amazon   | imdb     | 0.896                | 0.875             | 0.878                | 0.898             |
| amazon   | yelp     | 0.925                | 0.924             | 0.924                | 0.925             |
| imdb     | amazon   | 0.834                | 0.950             | 0.942                | 0.811             |
| imdb     | yelp     | 0.878                | 0.932             | 0.928                | 0.871             |
| yelp     | amazon   | 0.918                | 0.891             | 0.894                | 0.921             |
| yelp     | imdb     | 0.921                | 0.770             | 0.802                | 0.934             |

