import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from itertools import combinations

# Info from: https://www.ah.nl/informatie/eigen-merken and https://www.dirk.nl/meer/inspiratie/a-merk-vs-huismerk
OWN_BRAND_WORDS = {
    "1", "de", "beste",
    "vleeschmeesters",
    "met",
    "ah",
    "terra",
    "perla",
    "delicata",
    "zaanse","hoeve",
    "liefde", "&", "passie"
    "zaanlander",
    "brouwers",
    "streeckgenoten"
}

GENERIC_WORDS = {
    "met",
    "en",
    "op",
    "bij",
    "het",
    "voor",
    "van",
    "groot",
    "grote",
    "klein",
    "kleine",
    "verpakt",
    "zak",
    "vers",
    "verse",
    "los",
    "extra",
    "mini",
}

STOPWORDS = OWN_BRAND_WORDS | GENERIC_WORDS

def get_idf_dict(df_dirk: pd.DataFrame, df_ah: pd.DataFrame) -> dict:
    """
    Creates an IDF dictionary for each word in the combined product name dataset.
    """
    # Document: a product name (e.g. campina Bio halfvolle melk)
    # Corpus: all dirk + ah product names
    corpus = pd.concat([df_dirk["name_only"], df_ah["name_only"]])

    vectorizer = TfidfVectorizer(tokenizer=tokenize, min_df=3, max_df=0.2)
    _ = vectorizer.fit_transform(corpus)

    names = vectorizer.get_feature_names_out()
    idfs = vectorizer.idf_

    idf_dict = dict(zip(names, idfs))
    
    return idf_dict

def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"\w+", text.lower())
    return [t for t in tokens if t not in STOPWORDS]

def create_name_blocks(
    x: str,
    idf_dict: dict,
    max_single_tokens: int = 4,
    max_pair_tokens: int = 5
):
    """
    Create name blocks used as keys for candidate generation.

    Tokens with higher IDF values are prioritized because they usually are more
    informative for matching product names.
    """
    tokens = tokenize(x)

    tokens = [
        t for t in tokens
        if t in idf_dict
        and len(t) > 2
    ]

    # Important words first
    tokens.sort(key=lambda t: idf_dict[t], reverse=True)

    blocks = []

    # single rare tokens
    for t in tokens[:max_single_tokens]:
        blocks.append(t)

    # pairs of rare tokens
    for a, b in combinations(tokens[:max_pair_tokens], 2):
        blocks.append(f"{a}_{b}")

    return blocks

def add_features(df: pd.DataFrame, idf_dict: dict, supermarket = "dirk"):
    """
    Adds engineered features used for downstream matching and EDA
    """
    assert "price" in df.columns
    assert "quantity" in df.columns
    assert "name_only" in df.columns

    df = df.copy()

    # Price per unit
    df["price_per_unit"] = df["price"] / df["quantity"]
    df["price_per_unit"].head()

    # Vegan/Veggie
    MEATS = ["kip", "rund", "varken", "lam", "vlees", "VIS"]
    DAIRY = ["MELK", "BOTER", "YOGHURT", "ROOM", "honing", "wei", "wei-eiwit"]
    EGG_EXACT = r"\bEI\b" # Egg needs special handling (bc of contains)
    EGG_CONTAINS = ["EIGEEL", "EIPOEDER"]
    contains_pattern = "|".join(MEATS + DAIRY + EGG_CONTAINS)
    egg_pattern = EGG_EXACT

    ingredients_clean = df["ingredients"].str.split("Allergenen:").str[0]

    df["veggie"] = ~ingredients_clean.str.contains("|".join(MEATS), na=False)
    df["vegan"] = ~(
        ingredients_clean.str.contains(contains_pattern, na=False, regex=True, case=False)
        | ingredients_clean.str.contains(egg_pattern, na=False, regex=True)
    )
    df[["name", "ingredients", "veggie", "vegan"]].head()

    # Own brand
    if supermarket == "dirk":
        df["own_brand"] = df["name"].str.contains("1 de Beste", case=False)
    elif supermarket == "ah":
        ah_own_brand_pattern = r"AH|Perla|Delicata|De Zaanse Hoeve|Zaanlander|Brouwers"
        df["own_brand"] = df["name"].str.contains(ah_own_brand_pattern, case=False)
        
    # protein/euro
    weight_mask = (df["unit"] == "kg") | (df["unit"] == "l")

    df.loc[weight_mask, "protein_total_g"] = (
        df.loc[weight_mask, "protein_g"] *
        df.loc[weight_mask, "quantity"] * 10 # since protein per 100g
    )

    df.loc[weight_mask, "protein_per_euro"] = (
        df.loc[weight_mask, "protein_total_g"] /
        df.loc[weight_mask, "price"]
    )

    # relative kcal contribution per macronutrient
    df["frac_kcal_carbs"] = df["carbs_g"] * 4 / df["kcal"]
    df["frac_kcal_fat"] = df["fat_g"] * 9 / df["kcal"]
    df["frac_kcal_protein"] = df["protein_g"] * 4 / df["kcal"]

    # Create product_id column
    df.reset_index(drop=True, inplace=True) # remove old index after removing non-food items
    df.insert(0, "product_id", df.index)

    # Block key for candidate generation
    df["block_tokens"] = df.apply(
        lambda x: create_name_blocks(x["name_only"], idf_dict),
        axis=1
    )

    df["block_key"] = df.apply(
        lambda x: [
            f"{x['own_brand']}_{b}"
            for b in x["block_tokens"]
        ],
        axis=1
    )

    df = df.explode("block_key") # turn the [block_key] into separate columns

    return df