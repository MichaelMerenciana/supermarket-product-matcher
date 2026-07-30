"""
Streamlit app used for manually labeling product pairs.

The generated labels were used as ground truth for evaluating the product
matching approach. This file is only a data annotation tool and is not directly part
of the final matching pipeline.
"""

import streamlit as st
import pandas as pd
import numpy as np

path = "data/results/labeled_pairs.csv"

SEED = 42
np.random.seed(SEED)

df = pd.read_csv("data/results/pairs.csv")
labels = pd.read_csv("data/results/labeled_pairs.csv")

N = len(df)

draws = 600
high_draws = int(1/4 * draws)
mid_draws = int(2/4 * draws)
low_draws = int(1/4 * draws)

split1 = int(1/3 * N)
split2 = int(2/3 * N)

idxs_high = np.random.choice(np.arange(0, split1), size=high_draws, replace=False)
idxs_mid = np.random.choice(np.arange(split1, split2), size=mid_draws, replace=False)
idxs_low = np.random.choice(np.arange(split2, N), size=low_draws, replace=False)
draw_idxs = np.concatenate([idxs_high, idxs_mid, idxs_low])

if "idxs" not in st.session_state:
    st.session_state.idxs = draw_idxs

if "matches" not in st.session_state:
    st.session_state.matches = df

matches = st.session_state.matches
matches["label"] = labels["label"] # load already labeled pairs (if any)

if "input" not in st.session_state:
    st.session_state.input = 0

def label_match(value, draw_idxs: np.ndarray):
    product_idx = draw_idxs[st.session_state.input]
    matches.loc[product_idx, "label"] = value
    save_matches(matches, path)
    st.session_state.input += 1 # go to next drawn match

def save_matches(matches: pd.DataFrame, path):
    matches[["product_id_dirk", "product_id_ah", "name_dirk", "name_ah", "label"]].to_csv(path, index=False)

idx = st.number_input(
    "Random Draw index",
    key="input",
    min_value=0,
    max_value=draws  - 1
)

row = matches.iloc[draw_idxs[idx]]

data = [
    ["Name", row["href"], row["name_dirk"], row["name_ah"]],
    ["Embedding", row["embedding_score"], "", ""],
    ["Fuzz", row["name_fuzz_score"], "", ""],
    ["Jacc", row["name_jacc_score"], "", ""],
    ["Quantity", row["quantity_score"], row["quantity_dirk"], row["quantity_ah"]],
    ["Price", row["price_score"], row["price_dirk"], row["price_ah"]],
    ["Final", row["score"], "", ""],
]

st.table(data)

col1, col2 = st.columns(2)

if col1.button("Accept", on_click=label_match, args=(1, draw_idxs)):
    st.success("Accepted")

if col2.button("Reject", on_click=label_match, args=(0, draw_idxs)):
    st.error("Rejected")

if st.button("Save", on_click=save_matches, args=(matches, path)):
    st.info(f"Saved to: {path}")