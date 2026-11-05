# Result tables

## Within-domain macro-F1

| model          |   imdb |   yelp |   amazon |
|:---------------|-------:|-------:|---------:|
| DistilBERT     | 0.9018 | 0.9486 |   0.9254 |
| Majority class | 0.3333 | 0.3333 |   0.3333 |
| TF-IDF + LR    | 0.8985 | 0.9247 |   0.8795 |
| TF-IDF + SVM   | 0.8998 | 0.9324 |   0.8811 |
| Average        | 0.7584 | 0.7847 |   0.7548 |

## Cross-domain results

| setting   | model        | source   | target   |   accuracy |   macro_f1 |   within_macro_f1 |    drop |
|:----------|:-------------|:---------|:---------|-----------:|-----------:|------------------:|--------:|
| cross     | DistilBERT   | amazon   | imdb     |     0.8866 |     0.8866 |            0.9254 |  0.0388 |
| cross     | DistilBERT   | amazon   | yelp     |     0.9249 |     0.9249 |            0.9254 |  0.0005 |
| cross     | DistilBERT   | imdb     | amazon   |     0.8805 |     0.8799 |            0.9018 |  0.0219 |
| cross     | DistilBERT   | imdb     | yelp     |     0.9016 |     0.9015 |            0.9018 |  0.0003 |
| cross     | DistilBERT   | yelp     | amazon   |     0.906  |     0.906  |            0.9486 |  0.0426 |
| cross     | DistilBERT   | yelp     | imdb     |     0.8522 |     0.8512 |            0.9486 |  0.0974 |
| cross     | TF-IDF + LR  | amazon   | imdb     |     0.8508 |     0.8506 |            0.8795 |  0.0289 |
| cross     | TF-IDF + LR  | amazon   | yelp     |     0.8834 |     0.8834 |            0.8795 | -0.0039 |
| cross     | TF-IDF + LR  | imdb     | amazon   |     0.8212 |     0.8202 |            0.8985 |  0.0783 |
| cross     | TF-IDF + LR  | imdb     | yelp     |     0.8551 |     0.8549 |            0.8985 |  0.0436 |
| cross     | TF-IDF + LR  | yelp     | amazon   |     0.8288 |     0.8283 |            0.9247 |  0.0964 |
| cross     | TF-IDF + LR  | yelp     | imdb     |     0.7921 |     0.7891 |            0.9247 |  0.1356 |
| cross     | TF-IDF + SVM | amazon   | imdb     |     0.8643 |     0.8642 |            0.8811 |  0.0169 |
| cross     | TF-IDF + SVM | amazon   | yelp     |     0.8882 |     0.8882 |            0.8811 | -0.0071 |
| cross     | TF-IDF + SVM | imdb     | amazon   |     0.8159 |     0.8155 |            0.8998 |  0.0843 |
| cross     | TF-IDF + SVM | imdb     | yelp     |     0.8477 |     0.8474 |            0.8998 |  0.0524 |
| cross     | TF-IDF + SVM | yelp     | amazon   |     0.8335 |     0.8329 |            0.9324 |  0.0995 |
| cross     | TF-IDF + SVM | yelp     | imdb     |     0.7931 |     0.7896 |            0.9324 |  0.1428 |

### DistilBERT: macro-F1 by source and target

| source   |     imdb |     yelp |   amazon |
|:---------|---------:|---------:|---------:|
| imdb     | nan      |   0.9015 |   0.8799 |
| yelp     |   0.8512 | nan      |   0.906  |
| amazon   |   0.8866 |   0.9249 | nan      |

### TF-IDF + LR: macro-F1 by source and target

| source   |     imdb |     yelp |   amazon |
|:---------|---------:|---------:|---------:|
| imdb     | nan      |   0.8549 |   0.8202 |
| yelp     |   0.7891 | nan      |   0.8283 |
| amazon   |   0.8506 |   0.8834 | nan      |

### TF-IDF + SVM: macro-F1 by source and target

| source   |     imdb |     yelp |   amazon |
|:---------|---------:|---------:|---------:|
| imdb     | nan      |   0.8474 |   0.8155 |
| yelp     |   0.7896 | nan      |   0.8329 |
| amazon   |   0.8642 |   0.8882 | nan      |

## Mixed-domain results

| setting   | model        | train_domains   | target   |   train_size |   accuracy |   macro_f1 |   best_single_source_f1 |   delta_vs_best_single |
|:----------|:-------------|:----------------|:---------|-------------:|-----------:|-----------:|------------------------:|-----------------------:|
| mixed     | DistilBERT   | imdb+yelp       | amazon   |        20000 |     0.9132 |     0.9132 |                  0.906  |                 0.0072 |
| mixed     | DistilBERT   | yelp+amazon     | imdb     |        20000 |     0.8807 |     0.8807 |                  0.8866 |                -0.0059 |
| mixed     | DistilBERT   | imdb+amazon     | yelp     |        20000 |     0.9201 |     0.9201 |                  0.9249 |                -0.0048 |
| mixed     | TF-IDF + LR  | imdb+yelp       | amazon   |        20000 |     0.844  |     0.844  |                  0.8283 |                 0.0157 |
| mixed     | TF-IDF + LR  | yelp+amazon     | imdb     |        20000 |     0.8289 |     0.8284 |                  0.8506 |                -0.0222 |
| mixed     | TF-IDF + LR  | imdb+amazon     | yelp     |        20000 |     0.8743 |     0.8743 |                  0.8834 |                -0.0091 |
| mixed     | TF-IDF + SVM | imdb+yelp       | amazon   |        20000 |     0.8468 |     0.8468 |                  0.8329 |                 0.0139 |
| mixed     | TF-IDF + SVM | yelp+amazon     | imdb     |        20000 |     0.8532 |     0.8531 |                  0.8642 |                -0.0111 |
| mixed     | TF-IDF + SVM | imdb+amazon     | yelp     |        20000 |     0.8793 |     0.8793 |                  0.8882 |                -0.0089 |

## Per-class metrics, DistilBERT cross-domain

| source   | target   |   positive_precision |   positive_recall |   negative_precision |   negative_recall |
|:---------|:---------|---------------------:|------------------:|---------------------:|------------------:|
| amazon   | imdb     |               0.8958 |            0.875  |               0.8778 |            0.8982 |
| amazon   | yelp     |               0.9253 |            0.9244 |               0.9245 |            0.9254 |
| imdb     | amazon   |               0.8338 |            0.9504 |               0.9423 |            0.8106 |
| imdb     | yelp     |               0.8783 |            0.9324 |               0.928  |            0.8708 |
| yelp     | amazon   |               0.9184 |            0.8912 |               0.8943 |            0.9208 |
| yelp     | imdb     |               0.9215 |            0.77   |               0.8025 |            0.9344 |

