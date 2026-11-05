# AGENTS.md: project rules and progress

This file is the shared contract for the project. Both human team members and AI agents follow it.
Any change to this file gets announced in the group chat.

## 1. Server rules (MLDA lab, highest priority)

The lab workstation `gpu312` is shared. The policy, quoted from the account email:

> - Whenever possible, please be considerate to other users on the same workstation by following the simple tips below.
> - Only keep files you need in your account as the hard disk capacity is shared among all users on the same workstation.
> - Use only 1 to 2 GPU for your training sessions.
> - Clear your GPU memories after training.
> - System administrator will clear memory from GPU Cards if not utilised for 2 days or if more than 2 GPU cards are being utilised.

How the rules map to practice:

| Rule | Practice |
|---|---|
| At most 2 GPUs | Default `CUDA_VISIBLE_DEVICES=1`. Two cards only when running two jobs in parallel. Never three or four. |
| Free GPU memory after training | Let the Python process exit. Kill leftover jobs. Close Jupyter kernels. |
| Keep only needed files | Home quota is on a 7.3 T volume that is 94 percent full. Delete intermediate checkpoints and raw data when a stage finishes. Target under 10 GB for this project. |
| Idle memory is reclaimed after 2 days | Long jobs run inside `tmux`, and finished jobs get cleaned up. |
| Shared disk | Check `du -sh ~/*` and `df -h /media/mldadmin` before downloading anything large. |

Machine snapshot from 2026-11-05:

| Resource | Value |
|---|---|
| GPU | 4 x RTX 3090 24 GB, driver 535.113.01, CUDA 12.2. GPU 0 is usually busy with another user's jobs, so the project defaults to GPU 1. |
| CPU | Threadripper PRO 3955WX, 16 cores / 32 threads |
| RAM | 251 GiB |
| Home | `/media/mldadmin/home/<ntu-account>`, on the shared 7.3 T volume |
| Network | Direct access to HuggingFace, PyPI and GitHub |
| sudo | Not available. Everything installs under the home directory. |

The account expires at the end of December 2026. The lab does not back up data, so code lives in GitHub.

## 2. Environment

Conda lives at `~/miniconda3`, environment name `nlp`, Python 3.10.

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nlp
```

Pinned versions that work together on this machine:

| Package | Version |
|---|---|
| torch | 2.5.1+cu121 |
| transformers | 4.46.3 |
| datasets | 3.2.0 |
| huggingface-hub | 0.36.2 |

Newer transformers releases need torch 2.6 or later, so the versions are pinned on purpose.
Recreate the environment with `bash scripts/setup_env.sh` if it breaks.

## 3. Data rules

- One schema everywhere: `text` (string) and `label` (int, 0 = negative, 1 = positive).
- Amazon uses the `content` field. The `title` field is dropped so the three domains stay comparable.
- Fixed sizes per domain: 20,000 train, 2,000 validation, 10,000 test. All balanced at 50/50.
- Train and validation are drawn disjointly from the original training split.
- Seed is 42 everywhere. Splits are generated once and committed as parquet under `data/processed/`.
- Data and checkpoints are not committed. Team members regenerate data with `python src/prepare_data.py`.
- Label construction differs between datasets and is documented in `results/data_stats.md`.

## 4. Experiment rules

- Cross-domain evaluation reuses the within-domain checkpoints. Retraining per direction is not needed and would violate the GPU rule.
- Transformer fine-tunes in total: 2 tuning runs plus 3 within-domain plus 3 mixed-domain runs.
- The target domain never takes part in training or in hyperparameter selection.
- Test sets stay untouched until the final evaluation.
- Every run writes a JSON record with its hyperparameters and metrics.

## 5. Repo layout

```
src/          one script per stage
configs/      final hyperparameters
results/      CSV and JSON outputs, plus generated markdown
figures/      figures used in the report and video
data/         raw and processed data, not committed
models/       checkpoints, not committed
scripts/      helper shell scripts
```

## 6. Git rules

- One commit per stage, message prefix `feat:`, `fix:`, `docs:` or `chore:`.
- Do not commit `data/`, `models/`, `*.log` or files over 10 MB.
- Run `git status` before committing and check nothing large slipped in.

## 7. Progress

Last updated: 2026-11-05 14:40 SGT.

**Status: stages 3 to 11 are complete and pushed.** The repository holds 19 commits, 98
tracked files, 11 figures and the full result set. `REPORT.md` is the written deliverable.
Stage 12 and later were explicitly out of scope for this run.

| Stage | Description | Script | Output | Status |
|---|---|---|---|---|
| 1 | Define the problem | n/a | `README.md`, `REPORT.md` | done |
| 2 | Select three datasets | n/a | `README.md`, `results/data_stats.md` | done |
| 3 | Explore and prepare data | `src/prepare_data.py` | `data/processed/`, `results/data_stats.md` | done |
| 4 | Build baseline models | `src/baselines.py` | `results/baseline_within.csv` | done |
| 5 | Build the transformer model | `src/train_transformer.py` | `models/distilbert_*`, `results/transformer/` | done |
| 6 | Tune the models | `src/tune.py` | `configs/final.yaml`, `results/tuning_*.csv` | done |
| 7 | Within-domain experiments | `src/run_within.py` | `results/within_domain.csv` | done |
| 8 | Cross-domain experiments | `src/run_cross.py` | `results/cross_domain.csv` | done |
| 9 | Mixed-domain experiment | `src/run_mixed.py` | `results/mixed_domain.csv` | done |
| 10 | Evaluate and visualise | `src/evaluate.py`, `src/visualize.py` | `results/tables/`, `figures/` | done |
| 11 | Analyse errors | `src/error_analysis.py` | `results/error_analysis.md`, `results/error_taxonomy.csv` | done |
| 12 | Streamlit demonstration | n/a | n/a | out of scope for this run |
| 13 | Presentation and video | n/a | slides, video | pending |
| 14 | Package the submission | n/a | reflection, GAI declaration | pending |

Extra analysis inside Stage 3, 10 and 11:

| Script | Purpose | Output |
|---|---|---|
| `src/near_duplicates.py` | near-duplicate screening over all 90,000 pooled reviews | `results/near_duplicates.json` |
| `src/domain_distance.py` | Jensen-Shannon divergence and vocabulary overlap between domains | `results/domain_distance.csv`, `figures/fig_domain_distance.png` |
| `src/run_seeds.py` | multi-seed robustness for the DistilBERT within-domain runs | `results/seed_robustness.csv` |
| `src/feature_analysis.py` | strongest positive and negative terms per domain, and their overlap | `results/top_features.csv`, `results/feature_overlap.csv` |
| `src/data_scaling.py` | DistilBERT trained on 5k, 10k and 20k IMDb rows, tested on all three domains | `results/data_scaling.csv`, `figures/fig_data_scaling.png` |
| `src/error_by_length.py` | accuracy by review length for all DistilBERT settings | `results/error_by_length.csv`, `figures/fig_error_by_length.png` |

### Results so far

Within-domain macro-F1: DistilBERT 0.907 (IMDb), 0.943 (Yelp), 0.928 (Amazon), averaged over
three seeds. TF-IDF SVM 0.900, 0.932, 0.881. Logistic Regression 0.898, 0.925, 0.880.

Cross-domain: DistilBERT loses 0.097 going from Yelp to IMDb and 0.000 going from IMDb to
Yelp. The TF-IDF models lose roughly twice as much on average (0.063 to 0.065 against 0.034
for DistilBERT).

Mixed-domain: 3 of 9 combinations beat the best single source, mean change -0.003. All three
gains land on the Amazon target, and the sizes are close to the seed-to-seed spread, so the
result is weak evidence.

Seed spread: standard deviation across three seeds is 0.002 to 0.007 macro-F1, full range
0.013. The IMDb gap between DistilBERT and TF-IDF SVM (0.002) sits inside that spread.

Data checks: near-duplicate screening over 90,000 pooled reviews found 11 pairs, all inside
IMDb, none across datasets and none between train and test. Jensen-Shannon divergence is
lowest between IMDb and Amazon (0.146) and highest between IMDb and Yelp (0.225), which does
not line up with the transfer losses. Target review length lines up better.

Length and data: cross-domain accuracy is 0.909 on reviews of 100 words or fewer and 0.809 on
reviews over 200 words, and even within IMDb accuracy falls from 0.927 to 0.826 across those
buckets. Quadrupling the IMDb training set (5k to 20k rows) lifts within-domain macro-F1 by
0.009 but leaves the cross-domain scores flat or lower, so data scarcity is not the cause.
The data pipeline is deterministic: rerunning `src/prepare_data.py` regenerates all nine
parquet files with identical md5 checksums.

Full write-up: `REPORT.md`. Figures: `figures/`.

### Remaining work

- Record the 10 minute video and build the slides.
- Sign the group reflection document and the GAI declaration.
- Delete the checkpoints from the server after the demo, per the disk rule.

### Notes carried between stages

- IMDb has only 25,000 training reviews, so it caps the training size for every domain at 20,000.
- Yelp labels 3 to 4 stars as positive and drops 5-star reviews. Its positive class is weaker than the other two, which is a known confound.
- Amazon Polarity spans many product categories including Movies and TV, so part of its vocabulary overlaps with IMDb. Exact duplicates are removed. Near-duplicate screening over all 90,000 pooled reviews found 11 pairs, all inside IMDb, with none across datasets and none between train and test.
- GPU 0 is used by another account. Every training command sets `CUDA_VISIBLE_DEVICES` to 1 or 2.
- transformers 4.46.3 is pinned because newer releases need torch 2.6 or later and the server has torch 2.5.1.
- The DistilBERT learning rate tuning produced two runs that scored the same to four decimals (0.9090 on IMDb validation). The script picked 2e-5 and every domain model uses it.
- Vocabulary distance does not predict the transfer loss. IMDb and Amazon are the closest pair by Jensen-Shannon divergence (0.146) and still lose 0.022 to 0.039 macro-F1 between them. Target review length tracks the loss better.
- Stage 12 and later are outside the scope of this run. The work stops at Stage 11 plus the written report.
