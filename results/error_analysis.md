# Error analysis

Categories come from keyword heuristics applied to the misclassified reviews, so they overlap: one review can be tagged as both negation and contrast. Counts below are therefore shares of tagged errors, not a partition.

## Overall error categories

| categories        |   count |   mean_words |   share |
|:------------------|--------:|-------------:|--------:|
| contrast          |    1538 |        201.1 |   0.37  |
| negation          |    1507 |        200   |   0.363 |
| mixed opinion     |     536 |        264.4 |   0.129 |
| long review       |     287 |        486.6 |   0.069 |
| other             |     257 |         49.7 |   0.062 |
| very short review |      31 |          8.3 |   0.007 |

## Per-setting breakdown

| setting                   | categories        |   count |   share_of_errors |
|:--------------------------|:------------------|--------:|------------------:|
| distilbert_amazon_to_imdb | contrast          |     104 |             0.347 |
| distilbert_amazon_to_imdb | long review       |      44 |             0.147 |
| distilbert_amazon_to_imdb | mixed opinion     |      49 |             0.163 |
| distilbert_amazon_to_imdb | negation          |      99 |             0.33  |
| distilbert_amazon_to_imdb | other             |       4 |             0.013 |
| distilbert_amazon_to_yelp | contrast          |      81 |             0.343 |
| distilbert_amazon_to_yelp | long review       |      24 |             0.102 |
| distilbert_amazon_to_yelp | mixed opinion     |      28 |             0.119 |
| distilbert_amazon_to_yelp | negation          |      83 |             0.352 |
| distilbert_amazon_to_yelp | other             |      16 |             0.068 |
| distilbert_amazon_to_yelp | very short review |       4 |             0.017 |
| distilbert_imdb_to_amazon | contrast          |      75 |             0.401 |
| distilbert_imdb_to_amazon | mixed opinion     |      18 |             0.096 |
| distilbert_imdb_to_amazon | negation          |      71 |             0.38  |
| distilbert_imdb_to_amazon | other             |      23 |             0.123 |
| distilbert_imdb_to_yelp   | contrast          |      81 |             0.335 |
| distilbert_imdb_to_yelp   | long review       |      24 |             0.099 |
| distilbert_imdb_to_yelp   | mixed opinion     |      30 |             0.124 |
| distilbert_imdb_to_yelp   | negation          |      85 |             0.351 |
| distilbert_imdb_to_yelp   | other             |      16 |             0.066 |
| distilbert_imdb_to_yelp   | very short review |       6 |             0.025 |
| distilbert_yelp_to_amazon | contrast          |      68 |             0.41  |
| distilbert_yelp_to_amazon | mixed opinion     |      13 |             0.078 |
| distilbert_yelp_to_amazon | negation          |      56 |             0.337 |
| distilbert_yelp_to_amazon | other             |      29 |             0.175 |
| distilbert_yelp_to_imdb   | contrast          |     108 |             0.369 |
| distilbert_yelp_to_imdb   | long review       |      42 |             0.143 |
| distilbert_yelp_to_imdb   | mixed opinion     |      46 |             0.157 |
| distilbert_yelp_to_imdb   | negation          |      93 |             0.317 |
| distilbert_yelp_to_imdb   | other             |       4 |             0.014 |
| tfidf_lr_amazon_to_imdb   | contrast          |     106 |             0.367 |
| tfidf_lr_amazon_to_imdb   | long review       |      27 |             0.093 |
| tfidf_lr_amazon_to_imdb   | mixed opinion     |      53 |             0.183 |
| tfidf_lr_amazon_to_imdb   | negation          |     100 |             0.346 |
| tfidf_lr_amazon_to_imdb   | other             |       3 |             0.01  |
| tfidf_lr_amazon_to_yelp   | contrast          |      85 |             0.395 |
| tfidf_lr_amazon_to_yelp   | long review       |       8 |             0.037 |
| tfidf_lr_amazon_to_yelp   | mixed opinion     |      27 |             0.126 |
| tfidf_lr_amazon_to_yelp   | negation          |      80 |             0.372 |
| tfidf_lr_amazon_to_yelp   | other             |      10 |             0.047 |
| tfidf_lr_amazon_to_yelp   | very short review |       5 |             0.023 |
| tfidf_lr_imdb_to_amazon   | contrast          |      67 |             0.376 |
| tfidf_lr_imdb_to_amazon   | mixed opinion     |      12 |             0.067 |
| tfidf_lr_imdb_to_amazon   | negation          |      72 |             0.404 |
| tfidf_lr_imdb_to_amazon   | other             |      27 |             0.152 |
| tfidf_lr_imdb_to_yelp     | contrast          |      87 |             0.37  |
| tfidf_lr_imdb_to_yelp     | long review       |      12 |             0.051 |
| tfidf_lr_imdb_to_yelp     | mixed opinion     |      32 |             0.136 |
| tfidf_lr_imdb_to_yelp     | negation          |      89 |             0.379 |
| tfidf_lr_imdb_to_yelp     | other             |       9 |             0.038 |
| tfidf_lr_imdb_to_yelp     | very short review |       6 |             0.026 |
| tfidf_lr_yelp_to_amazon   | contrast          |      62 |             0.369 |
| tfidf_lr_yelp_to_amazon   | mixed opinion     |      11 |             0.065 |
| tfidf_lr_yelp_to_amazon   | negation          |      66 |             0.393 |
| tfidf_lr_yelp_to_amazon   | other             |      29 |             0.173 |
| tfidf_lr_yelp_to_imdb     | contrast          |     105 |             0.362 |
| tfidf_lr_yelp_to_imdb     | long review       |      30 |             0.103 |
| tfidf_lr_yelp_to_imdb     | mixed opinion     |      47 |             0.162 |
| tfidf_lr_yelp_to_imdb     | negation          |     104 |             0.359 |
| tfidf_lr_yelp_to_imdb     | other             |       4 |             0.014 |
| tfidf_svm_amazon_to_imdb  | contrast          |     105 |             0.376 |
| tfidf_svm_amazon_to_imdb  | long review       |      24 |             0.086 |
| tfidf_svm_amazon_to_imdb  | mixed opinion     |      49 |             0.176 |
| tfidf_svm_amazon_to_imdb  | negation          |      99 |             0.355 |
| tfidf_svm_amazon_to_imdb  | other             |       2 |             0.007 |
| tfidf_svm_amazon_to_yelp  | contrast          |      85 |             0.397 |
| tfidf_svm_amazon_to_yelp  | long review       |      11 |             0.051 |
| tfidf_svm_amazon_to_yelp  | mixed opinion     |      23 |             0.107 |
| tfidf_svm_amazon_to_yelp  | negation          |      79 |             0.369 |
| tfidf_svm_amazon_to_yelp  | other             |      11 |             0.051 |
| tfidf_svm_amazon_to_yelp  | very short review |       5 |             0.023 |
| tfidf_svm_imdb_to_amazon  | contrast          |      65 |             0.376 |
| tfidf_svm_imdb_to_amazon  | mixed opinion     |      10 |             0.058 |
| tfidf_svm_imdb_to_amazon  | negation          |      71 |             0.41  |
| tfidf_svm_imdb_to_amazon  | other             |      27 |             0.156 |
| tfidf_svm_imdb_to_yelp    | contrast          |      91 |             0.378 |
| tfidf_svm_imdb_to_yelp    | long review       |      13 |             0.054 |
| tfidf_svm_imdb_to_yelp    | mixed opinion     |      32 |             0.133 |
| tfidf_svm_imdb_to_yelp    | negation          |      90 |             0.373 |
| tfidf_svm_imdb_to_yelp    | other             |      10 |             0.041 |
| tfidf_svm_imdb_to_yelp    | very short review |       5 |             0.021 |
| tfidf_svm_yelp_to_amazon  | contrast          |      58 |             0.349 |
| tfidf_svm_yelp_to_amazon  | mixed opinion     |      11 |             0.066 |
| tfidf_svm_yelp_to_amazon  | negation          |      67 |             0.404 |
| tfidf_svm_yelp_to_amazon  | other             |      30 |             0.181 |
| tfidf_svm_yelp_to_imdb    | contrast          |     105 |             0.37  |
| tfidf_svm_yelp_to_imdb    | long review       |      28 |             0.099 |
| tfidf_svm_yelp_to_imdb    | mixed opinion     |      45 |             0.158 |
| tfidf_svm_yelp_to_imdb    | negation          |     103 |             0.363 |
| tfidf_svm_yelp_to_imdb    | other             |       3 |             0.011 |

## Confusion direction

| setting                   |   true |   pred |   count |
|:--------------------------|-------:|-------:|--------:|
| distilbert_amazon_to_imdb |      0 |      1 |      59 |
| distilbert_amazon_to_imdb |      1 |      0 |      61 |
| distilbert_amazon_to_yelp |      0 |      1 |      58 |
| distilbert_amazon_to_yelp |      1 |      0 |      62 |
| distilbert_imdb_to_amazon |      0 |      1 |      97 |
| distilbert_imdb_to_amazon |      1 |      0 |      23 |
| distilbert_imdb_to_yelp   |      0 |      1 |      74 |
| distilbert_imdb_to_yelp   |      1 |      0 |      46 |
| distilbert_yelp_to_amazon |      0 |      1 |      48 |
| distilbert_yelp_to_amazon |      1 |      0 |      72 |
| distilbert_yelp_to_imdb   |      0 |      1 |      22 |
| distilbert_yelp_to_imdb   |      1 |      0 |      98 |
| tfidf_lr_amazon_to_imdb   |      0 |      1 |      41 |
| tfidf_lr_amazon_to_imdb   |      1 |      0 |      79 |
| tfidf_lr_amazon_to_yelp   |      0 |      1 |      47 |
| tfidf_lr_amazon_to_yelp   |      1 |      0 |      73 |
| tfidf_lr_imdb_to_amazon   |      0 |      1 |      79 |
| tfidf_lr_imdb_to_amazon   |      1 |      0 |      41 |
| tfidf_lr_imdb_to_yelp     |      0 |      1 |      72 |
| tfidf_lr_imdb_to_yelp     |      1 |      0 |      48 |
| tfidf_lr_yelp_to_amazon   |      0 |      1 |      39 |
| tfidf_lr_yelp_to_amazon   |      1 |      0 |      81 |
| tfidf_lr_yelp_to_imdb     |      0 |      1 |      29 |
| tfidf_lr_yelp_to_imdb     |      1 |      0 |      91 |
| tfidf_svm_amazon_to_imdb  |      0 |      1 |      47 |
| tfidf_svm_amazon_to_imdb  |      1 |      0 |      73 |
| tfidf_svm_amazon_to_yelp  |      0 |      1 |      56 |
| tfidf_svm_amazon_to_yelp  |      1 |      0 |      64 |
| tfidf_svm_imdb_to_amazon  |      0 |      1 |      74 |
| tfidf_svm_imdb_to_amazon  |      1 |      0 |      46 |
| tfidf_svm_imdb_to_yelp    |      0 |      1 |      73 |
| tfidf_svm_imdb_to_yelp    |      1 |      0 |      47 |
| tfidf_svm_yelp_to_amazon  |      0 |      1 |      34 |
| tfidf_svm_yelp_to_amazon  |      1 |      0 |      86 |
| tfidf_svm_yelp_to_imdb    |      0 |      1 |      23 |
| tfidf_svm_yelp_to_imdb    |      1 |      0 |      97 |
