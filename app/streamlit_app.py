"""Stage 12: Streamlit demo for the cross-domain sentiment models.

Run on the GPU server:
    streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0

On a laptop, forward the port first:
    ssh -NfL localhost:8501:localhost:8501 <user>@gpu312.dynip.ntu.edu.sg
"""
import os
import sys

import pandas as pd
import streamlit as st
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from transformer_utils import MODEL_NAME, predict_texts  # noqa: E402

DOMAINS = ["imdb", "yelp", "amazon"]
DOMAIN_LABEL = {
    "imdb": "IMDb (movie reviews)",
    "yelp": "Yelp (business reviews)",
    "amazon": "Amazon (product reviews)",
}

st.set_page_config(page_title="Cross-domain sentiment", layout="wide")
st.title("Cross-domain sentiment analysis")
st.caption(
    "Fine-tuned DistilBERT checkpoints from the NLP group project. "
    "Pick the domain a model was trained on, then type a review."
)


@st.cache_resource
def load_model(domain):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    path = os.path.join(ROOT, "models", f"distilbert_{domain}")
    if not os.path.isdir(path):
        return None, None
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).eval()
    return tokenizer, model


def score(tokenizer, model, text, max_len=256):
    enc = tokenizer([text], truncation=True, max_length=max_len, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[0].tolist()
    return probs


def results_table():
    path = os.path.join(ROOT, "results", "within_domain.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df.pivot_table(index="model", columns="test_domain", values="macro_f1").round(3)
    return None


col_left, col_right = st.columns([3, 2])

with col_left:
    domain = st.selectbox("Model trained on", DOMAINS, format_func=lambda d: DOMAIN_LABEL[d])
    text = st.text_area(
        "Review text",
        "The film looked gorgeous but the plot made no sense and the ending was a mess.",
        height=160,
    )
    if st.button("Predict", type="primary"):
        tokenizer, model = load_model(domain)
        if tokenizer is None:
            st.error(f"No checkpoint found at models/distilbert_{domain}. Train it first.")
        else:
            probs = score(tokenizer, model, text)
            label = "positive" if probs[1] >= probs[0] else "negative"
            st.metric("Prediction", label)
            st.write("confidence", f"{max(probs):.3f}")
            st.bar_chart(pd.DataFrame({"probability": probs}, index=["negative", "positive"]))

with col_right:
    st.subheader("Within-domain macro-F1")
    table = results_table()
    if table is not None:
        st.dataframe(table, use_container_width=True)
    else:
        st.info("Run the experiments first: results/within_domain.csv is missing.")

st.divider()
st.subheader("What the model is good at, and where it breaks")
st.markdown(
    "A checkpoint trained on one domain is usually weakest when it meets reviews from another "
    "domain. The cross-domain tables in `results/tables/` show how large that gap is for each "
    "pair of domains, and `results/error_analysis.md` lists the kinds of reviews that go wrong."
)
