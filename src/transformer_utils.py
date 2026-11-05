"""Shared DistilBERT helpers for training and inference."""
import inspect
import os

import numpy as np
import torch

MODEL_NAME = "distilbert-base-uncased"


def pick_eval_key():
    """TrainingArguments renamed evaluation_strategy to eval_strategy in transformers 4.46."""
    params = inspect.signature(__import__("transformers").TrainingArguments.__init__).parameters
    return "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"


def tokenize_frame(tokenizer, frame, max_len):
    from datasets import Dataset

    ds = Dataset.from_dict({"text": frame["text"].tolist(), "label": frame["label"].tolist()})

    def _tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_len)

    return ds.map(_tok, batched=True, remove_columns=["text"])


def predict_texts(model_dir, texts, max_len=256, batch_size=64, device=None):
    """Run inference with a saved checkpoint and return predicted labels."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    preds = []
    texts = list(texts)
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = tokenizer(
                batch, truncation=True, max_length=max_len, padding=True, return_tensors="pt"
            ).to(device)
            logits = model(**enc).logits
            preds.extend(torch.argmax(logits, dim=-1).cpu().numpy().tolist())
    return np.array(preds)
