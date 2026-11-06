"""Fine-tune DistilBERT on one or two domains and evaluate on a target domain.

Within-domain:
    python src/train_transformer.py --train_domains imdb --test_domain imdb

Mixed-domain (train on two source domains, test on the third):
    python src/train_transformer.py --train_domains imdb,yelp --test_domain amazon

Validation always comes from the training domains, so the target domain never informs
hyperparameter choice.
"""
import argparse
import os
import platform
import time

import numpy as np
import pandas as pd
import torch

from common import (
    MAX_LEN,
    MIXED_TRAIN_TOTAL,
    MODELS,
    RESULTS,
    SEED,
    TRAIN_PER_CLASS,
    compute_metrics,
    ensure_dirs,
    load_split,
    set_seed,
    write_json,
)
from transformer_utils import MODEL_NAME, pick_eval_key, tokenize_frame

# Rows per domain in a within-domain run: 10,000 per class.
WITHIN_TRAIN_TOTAL = TRAIN_PER_CLASS * 2


def balanced_head(df, rows):
    """First `rows // 2` examples of each class, shuffled."""
    n = rows // 2
    return pd.concat([df[df.label == 0].head(n), df[df.label == 1].head(n)])


def build_training_frame(train_domains, per_domain):
    parts = [balanced_head(load_split(d, "train"), per_domain) for d in train_domains]
    return pd.concat(parts).sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def build_validation_frame(val_domains, per_domain):
    parts = [balanced_head(load_split(d, "val"), per_domain) for d in val_domains]
    return pd.concat(parts).sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_domains", default="imdb")
    parser.add_argument("--test_domain", default="imdb")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=MAX_LEN)
    parser.add_argument("--tag", default="")
    parser.add_argument(
        "--val_rows",
        type=int,
        default=1000,
        help="validation rows for a single source domain; mixed runs scale this per domain",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--train_size",
        type=int,
        default=0,
        help="total balanced training rows; 0 uses the default for the setting",
    )
    args = parser.parse_args()

    train_domains = [d.strip() for d in args.train_domains.split(",") if d.strip()]
    test_domain = args.test_domain
    is_mixed = len(train_domains) > 1
    tag = args.tag or ("_".join(train_domains) if is_mixed else train_domains[0])

    ensure_dirs()
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} train={train_domains} test={test_domain}")

    import transformers
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    per_domain = MIXED_TRAIN_TOTAL // len(train_domains) if is_mixed else WITHIN_TRAIN_TOTAL
    if args.train_size:
        per_domain = args.train_size // len(train_domains)
    train_df = build_training_frame(train_domains, per_domain)
    val_df = build_validation_frame(train_domains, args.val_rows * len(train_domains))
    test_df = load_split(test_domain, "test")
    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    train_ds = tokenize_frame(tokenizer, train_df, args.max_len)
    val_ds = tokenize_frame(tokenizer, val_df, args.max_len)
    test_ds = tokenize_frame(tokenizer, test_df, args.max_len)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    def hf_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": float((preds == labels).mean()),
            "macro_f1": float(compute_metrics(labels.tolist(), preds.tolist())["macro_f1"]),
        }

    out_dir = os.path.join(MODELS, f"distilbert_{tag}")
    eval_key = pick_eval_key()
    train_kwargs = dict(
        output_dir=out_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64,
        num_train_epochs=args.epochs,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=2,
    )
    train_kwargs[eval_key] = "epoch"

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**train_kwargs),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=hf_metrics,
    )

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    val_pred = np.argmax(trainer.predict(val_ds).predictions, axis=-1)
    test_pred = np.argmax(trainer.predict(test_ds).predictions, axis=-1)
    val_metrics = compute_metrics(val_df["label"].tolist(), val_pred.tolist())
    test_metrics = compute_metrics(test_df["label"].tolist(), test_pred.tolist())

    result = {
        "model": "distilbert",
        "train_domains": train_domains,
        "test_domain": test_domain,
        "setting": "mixed" if is_mixed else "within",
        "tag": tag,
        "seed": args.seed,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_len": args.max_len,
        "train_size": int(len(train_df)),
        "val_size": int(len(val_df)),
        "train_time_s": round(train_time, 1),
        "val": val_metrics,
        "test": test_metrics,
        "provenance": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "host": platform.node(),
        },
        "history": trainer.state.log_history,
    }
    write_json(os.path.join(RESULTS, "transformer", f"{tag}_to_{test_domain}.json"), result)
    print(
        f"[{tag} -> {test_domain}] val macro_f1={val_metrics['macro_f1']:.4f} "
        f"test acc={test_metrics['accuracy']:.4f} test macro_f1={test_metrics['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
