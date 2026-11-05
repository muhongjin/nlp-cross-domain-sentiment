#!/bin/bash
# Reproduce every stage in order on the GPU server.
# Usage: bash scripts/run_all.sh
set -e
cd "$(dirname "$0")/.."
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate nlp

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

python src/prepare_data.py
python src/baselines.py

# Stage 6: learning rate chosen on IMDb validation only.
python src/train_transformer.py --train_domains imdb --test_domain imdb --lr 2e-5 --tag tune_lr2e5
python src/train_transformer.py --train_domains imdb --test_domain imdb --lr 3e-5 --tag tune_lr3e5
python src/tune.py

LR=$(python -c "import yaml; print(yaml.safe_load(open('configs/final.yaml'))['distilbert']['learning_rate'])")
echo "selected learning rate: $LR"

# Stage 5 and 7: one checkpoint per domain.
for d in imdb yelp amazon; do
  python src/train_transformer.py --train_domains "$d" --test_domain "$d" --lr "$LR"
done
python src/run_within.py

# Stage 8: cross-domain evaluation reuses the checkpoints above.
python src/run_cross.py

# Stage 9: mixed-domain training, three runs, two source domains each.
python src/train_transformer.py --train_domains imdb,yelp --test_domain amazon --lr "$LR"
python src/train_transformer.py --train_domains imdb,amazon --test_domain yelp --lr "$LR"
python src/train_transformer.py --train_domains yelp,amazon --test_domain imdb --lr "$LR"
python src/run_mixed.py

# Stage 10 and 11.
python src/evaluate.py
python src/visualize.py
python src/error_analysis.py
