# AGENTS.md — Cross-Domain Sentiment Analysis

本文件是项目的协作约定，**人类成员和 AI Agent 都必须遵守**。
修改本文件需在组内说明。

---

## 0. 服务器使用规则（MLDA Lab 强制，优先级最高）

> 原文（NTU MLDA workstation policy）：
>
> - Whenever possible, please be considerate to other users on the same workstation by
>   following the simple tips below.
> - Only keep files you need in your account as the hard disk capacity is shared among all
>   users on the same workstation.
> - Use only 1 to 2 GPU for your training sessions.
> - Clear your GPU memories after training.
> - System administrator will clear memory from GPU Cards if not utilised for 2 days or if
>   more than 2 GPU cards are being utilised.

### 0.1 硬性规则

| # | 规则 | 落地做法 |
|---|---|---|
| 1 | **训练最多用 1–2 张 GPU** | 默认 `CUDA_VISIBLE_DEVICES=1`；确需并行时最多 `1,2`，**禁止 `0,1,2,3`** |
| 2 | **训练结束立即释放显存** | 跑完 `kill` 掉进程，或确认 Python 进程已退出；Jupyter 用完关 kernel |
| 3 | **只保留必要文件** | 磁盘全机共享，当前仅剩约 480 GB（94% 已用）。中间产物、旧 checkpoint 用完即删 |
| 4 | **空闲显存超 2 天会被管理员清理** | 长训练用 `tmux` 挂住，别让进程处于"占着卡不干活"状态 |
| 5 | **占用超过 2 张卡会被管理员清理** | 同上，严格遵守第 1 条 |

### 0.2 资源现状（2026-11 探测）

| 资源 | 状态 | 使用建议 |
|---|---|---|
| GPU | 4× RTX 3090 24GB，驱动 535.113.01 / CUDA 12.2 | **GPU0 常被其他用户占用**，默认用 GPU1 |
| CPU | Threadripper PRO 3955WX，16C/32T | 数据预处理可放心用多进程 |
| 内存 | 251 GiB | 不是瓶颈 |
| `/media/mldadmin` | 7.3 T，已用 94%，剩 ~480 G | home 在此，**不要放大文件** |
| `/` | 879 G，已用 5% | 不归我们使用 |
| 网络 | HuggingFace / PyPI / GitHub 直连可用 | 无需镜像 |
| sudo | 无 | 所有安装走用户级（home 下） |

账号有效期至 **2026 年 12 月底**，实验室**不备份数据**——代码一律推 GitHub。

---

## 1. 项目速览

- **任务**：跨域情感分析。三个域（IMDb 电影 / Yelp 商户 / Amazon 商品）二分类。
- **模型**：多数类基线、TF-IDF + LR、TF-IDF + SVM、微调 DistilBERT。
- **实验**：域内（within）、跨域（cross）、混合域（mixed）。
- **主指标**：macro-F1（辅以 accuracy、各类 P/R/F1、混淆矩阵）。
- **仓库**：https://github.com/muhongjin/nlp-cross-domain-sentiment

---

## 2. 环境

```bash
# 一次性初始化（用户级，无需 sudo）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init bash && source ~/.bashrc
conda create -n nlp python=3.10 -y && conda activate nlp
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 验证
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

环境名固定为 **`nlp`**，Python **3.10**。不要用系统 Python 3.8。

---

## 3. 数据约定

- **统一 schema**：`text: str`, `label: int`（`0 = negative`, `1 = positive`）。所有脚本只认这两个字段。
- **Amazon 用 `content` 字段**，丢弃 `title`，保证三域公平。
- **统一规模**：每域 20,000 train / 2,000 validation / 10,000 test。
- **固定 seed**：`SEED = 42`，划分只生成一次，产物进 `data/processed/`。
- **标签映射写死并断言**，不要靠猜：
  ```python
  # IMDb: 0=neg, 1=pos | Yelp: 0=neg(1-2★), 1=pos(3-4★) | Amazon: 0=neg, 1=pos
  assert set(labels) == {0, 1}
  ```
- **数据与 checkpoint 不入库**（已在 `.gitignore`）。组员 clone 后跑 `src/prepare_data.py` 重新生成。

---

## 4. 实验规范

### 4.1 GPU

```bash
export CUDA_VISIBLE_DEVICES=1          # 默认
export CUDA_VISIBLE_DEVICES=1,2        # 仅当确需并行，且不超过 2 张
```

训练脚本结束前释放资源：

```python
import gc, torch
del model, optimizer
gc.collect()
torch.cuda.empty_cache()
```

跑完确认已释放：

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

### 4.2 长任务

一律用 `tmux`，禁止裸跑：

```bash
tmux new -s train
# Ctrl+B 然后 D 脱离；tmux attach -t train 回来
```

### 4.3 算力预算

- 单次 DistilBERT 微调约 5–8 分钟（20K 样本，`max_len=256`，3 epoch）。
- 总训练次数：**3 次域内 + 3 次混合域 = 6 次**。跨域实验复用已有 checkpoint，**不重新训练**。
- 不要按"每个跨域方向训一个模型"来做——那是 24 次，违反资源规则。

### 4.4 可复现

- 所有超参写进 `configs/final.yaml`。
- 每次实验记录：模型、域、seed、超参、指标。
- 测试集在最终评估前不得参与任何调参。

---

## 5. 仓库与协作

### 5.1 目录

```
data/processed/    # 对齐后的 parquet（不入库）
src/               # 每个 Stage 一个脚本
configs/           # 超参配置
results/           # csv / json 结果
figures/           # 进报告的图
models/            # checkpoint（不入库，用完清理）
```

### 5.2 Git

- 每人负责的 Stage 用独立分支，通过 PR 合并到 `main`。
- **不要直接改别人的脚本**，各自负责文件；需要改动先沟通。
- commit message 用 `feat:` / `fix:` / `docs:` / `chore:` 前缀。
- 不提交：`data/`、`models/`、`*.log`、`.env`、大文件（>10 MB）。
- 提交前 `git status` 确认没有误加数据文件。

### 5.3 结果文件

- 命名固定，避免互相覆盖：
  - `results/within_domain.csv`
  - `results/cross_domain.csv`
  - `results/mixed_domain.csv`
  - `results/error_taxonomy.csv`
- 重跑覆盖前先确认不是别人正在用的版本。

---

## 6. 磁盘卫生

- 训练产生的中间 checkpoint 只留**最好的一版**，其余删除。
- 数据集原始文件（`data/raw/`）用完可删，`prepare_data.py` 能重新下载。
- 定期检查：
  ```bash
  du -sh ~/* 2>/dev/null | sort -h
  df -h /media/mldadmin
  ```
- 本项目在服务器上的总占用建议控制在 **10 GB 以内**。

---

## 7. 禁止事项

- ❌ 同时占用 3 张及以上 GPU
- ❌ 训练结束后不释放显存 / 留着僵尸进程
- ❌ 把大数据集、模型权重提交进 git
- ❌ 使用系统 Python 3.8 跑实验
- ❌ 用测试集调参
- ❌ 在 `main` 分支上直接开发
- ❌ 长期保留不用的文件占用共享磁盘
- ❌ 把服务器密码写进任何文件或提交记录

---

## 8. 提交前检查清单

- [ ] 显存已释放（`nvidia-smi` 无自己的进程）
- [ ] 没有新增 >10 MB 的文件进 git
- [ ] 结果文件已写入 `results/`，命名符合约定
- [ ] 超参记录在 `configs/` 或结果文件里
- [ ] 代码用固定 seed 可复现
- [ ] 数据来自 `data/processed/`，不是各自重新下载的版本
