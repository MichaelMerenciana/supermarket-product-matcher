import numpy as np
import pandas as pd
import pickle
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from supermarkt.feature_engineering import tokenize

df_dirk = pd.read_csv("data/processed/dirk.csv")
df_ah = pd.read_csv("data/processed/ah.csv")

COSINE_WEIGHT = 0.35
FUZZY_WEIGHT = 0.25
JACCARD_WEIGHT = 0.10
QUANTITY_WEIGHT = 0.25
PRICE_WEIGHT = 0.05

MIN_SIMILARITY_THRESHOLD = 0.6

# region Entity Resolution

# Use Jaccard similarity with tokens
def jaccard_similarity(t1: set[str], t2: set[str]) -> float:
    return len(t1 & t2) / len(t1 | t2)

def name_fuzz_similarity(x1: str, x2: str):
    return fuzz.token_sort_ratio(x1, x2) / 100

def quantity_similarity(x1: float, x2: float, p = 2):
    # Punish slight differences more than linearly
    rel_diff = abs(x1 - x2) / max(x1, x2)
    return (1 - rel_diff) ** p

def price_similarity(x1: float, x2: float): # relative price difference
    rel_diff = abs(x1 - x2) / max(x1, x2)
    return 1 - rel_diff

def embedding_similarity(x1: np.ndarray, x2: np.ndarray):
    return cosine_similarity([x1], [x2])[0][0]

def matching_score(row, cos_sims: np.ndarray) -> pd.Series:
    """
    Compute matching features and a weighted similarity score for a product pair.

    The final score combines multiple similarity measures:
    - cosine similarity between product embeddings (0.35)
    - fuzzy name similarity (0.25)
    - token Jaccard similarity (0.10)
    - quantity similarity (0.25)
    - price similarity (0.05)
    """

    cosine_sim = cos_sims[row["product_id_dirk"], row["product_id_ah"]]
    name_fuzz_score = name_fuzz_similarity(row["name_only_dirk"], row["name_only_ah"])
    name_jaccard_score = jaccard_similarity(set(row["name_tokens_dirk"]), set(row["name_tokens_ah"]))
    quantity_score = quantity_similarity(row["quantity_dirk"], row["quantity_ah"])
    price_score = price_similarity(row["price_dirk"], row["price_ah"])

    score = (
        COSINE_WEIGHT * cosine_sim +
        FUZZY_WEIGHT * name_fuzz_score +
        JACCARD_WEIGHT * name_jaccard_score +
        QUANTITY_WEIGHT * quantity_score +
        PRICE_WEIGHT * price_score
    )

    return pd.Series({
        "embedding_score": cosine_sim,
        "name_fuzz_score": name_fuzz_score,
        "name_jacc_score": name_jaccard_score,
        "quantity_score": quantity_score,
        "price_score": price_score,
        "score": score,
    })

def generate_candidates(df_dirk: pd.DataFrame, df_ah: pd.DataFrame):
    """
    Generate product pair candidates using precomputed block keys.

    Products from both supermarkets are matched on shared block keys to
    reduce the number of possible comparisons before applying the full
    matching model.
    """
    df_dirk["name_tokens"] = df_dirk["name_only"].map(lambda x: set(tokenize(x)))
    df_ah["name_tokens"] = df_ah["name_only"].map(lambda x: set(tokenize(x)))

    candidates = df_dirk.merge(
        df_ah,
        on=["block_key"],
        suffixes=("_dirk", "_ah")
    )

    return candidates

def add_score_columns(df: pd.DataFrame):
    df["embedding_score"] = 0.0
    df["name_fuzz_score"] = 0.0
    df["name_jacc_score"] = 0.0
    df["quantity_score"] = 0.0
    df["price_score"] = 0.0

# endregion

# region Embedding

def embed_products(df: pd.DataFrame, model: SentenceTransformer):
    """
    Embed the text representations of the product using a multilingual Sentence-BERT model.

    Returns:
    2D Tensor with the product embeddings.
    """

    assert "name" in df.columns
    assert "quantity" in df.columns
    assert "unit" in df.columns
    assert "ingredients" in df.columns

    products = (
        df[["name", "quantity", "unit", "ingredients"]]
        .fillna("")
        .apply(
            lambda row: (
                f"Name: {row['name']}; "
                f"Quantity: {row['quantity']}{row['unit']}; "
                f"Ingredients: {row['ingredients']}"
            ),
            axis=1
        ).tolist()
    )

    embeddings = model.encode(products, show_progress_bar=True)

    return embeddings

def embed_write(model: SentenceTransformer, supermarket = "dirk"):
    """
    Embed the products of a supermarket and store them in pickle.

    The processed product data is loaded, duplicate product entries created
    by previous explode operations are removed,
    
    and the resulting product
    representations are embedded and stored as a pickle file.
    """

    in_path = f"data/processed/{supermarket}.csv"
    out_path = f"data/processed/{supermarket}_embeddings.pickle"

    df = pd.read_csv(in_path)

    # Undo explode: keep one product per id
    df = df.drop_duplicates(subset="product_id", keep="first").reset_index(drop=True)
    df.to_csv(f"data/processed/{supermarket}_unexploded.csv")

    print(f"Embedding products for {supermarket}")
    embeddings = embed_products(df, model)

    with open(out_path, "wb") as f:
        pickle.dump(embeddings, f)

def embed_read(supermarket = "dirk") -> np.ndarray:
    out_path = f"data/processed/{supermarket}_embeddings.pickle"

    with open(out_path, "rb") as f:
        embeddings = pickle.load(f)

    return embeddings

model = SentenceTransformer("intfloat/multilingual-e5-base")

embed_write(model, supermarket="dirk")
embed_write(model, supermarket="ah")

embeddings_dirk = embed_read("dirk")
embeddings_ah = embed_read("ah")

df_dirk_unexploded = pd.read_csv("data/processed/dirk_unexploded.csv")
df_ah_unexploded = pd.read_csv("data/processed/ah_unexploded.csv")

cosine_scores = embeddings_dirk @ embeddings_ah.T

candidates = generate_candidates(df_dirk, df_ah)
print("Candidates:", len(candidates))

add_score_columns(candidates)

candidates[[
    "embedding_score",
    "name_fuzz_score",
    "name_jacc_score",
    "quantity_score",
    "price_score",
    "score",
]] = candidates.apply(
    lambda row: matching_score(row, cosine_scores),
    axis=1,
)


pairs = candidates[candidates["score"] > MIN_SIMILARITY_THRESHOLD]
high_score_cand = pairs[["product_id_dirk", "product_id_ah", "href", "name_only_dirk", "name_only_ah", "price_dirk", "price_ah", "quantity_dirk", "quantity_ah", "score"]]
high_score_cand.to_csv("data/results/high_score_cand.csv", index=False)
print("High-scoring Candidates:", len(pairs))

best_matches = (
    pairs.sort_values("score", ascending=False)
         .drop_duplicates(subset="href", keep="first")
)

print("Total Matches:", len(best_matches))

best_matches.to_csv("data/results/pairs.csv", index=False)