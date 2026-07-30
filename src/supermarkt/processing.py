import pandas as pd
import numpy as np
import re
import os
from supermarkt.feature_engineering import add_features, get_idf_dict

UNIT_MAP = {
    "kilo": "kg",
    "gr": "g",
    "gram": "g",
    "liter": "l",
    "stuks": "stuk",
    "bos": "stuk",
    "pakket": "stuk",
    "cups": "stuk",
    "zakjes": "stuk",
    "plakjes": "stuk",
    "dozen": "stuk",
    "Tros": "stuk",
    "pakket": "stuk",
    "min": "stuk",
    "personen": "stuk",
    "tabletten": "stuk"
}

# region DIRK

path_dirk = "data/raw/dirk.csv"
df_dirk = pd.read_csv(path_dirk)
df_dirk = df_dirk.drop("nutrients", axis=1)

# Cleaning units

df_dirk = df_dirk[df_dirk["href"].str.contains("http")] # drop any rows without links
df_dirk = df_dirk[df_dirk["portion"].str.contains(" ", na=False)].copy()

df_dirk["unit"] = df_dirk["portion"].str.split().str[1]

# Regex to capture the name only
pattern_name = r"\s*(?:\d+(?:[.,]\d+)?\s*(?:kg|kilo|g|gr|gram|ml|liter|l|stuk|stuks|st\.?|pak|pakket|bos|cups?|zakjes?)|\d+\s*-?\s*pack)\s*$"

df_dirk["name_only"] = df_dirk["name"].str.replace(pattern_name, "", case=False, regex=True)

# Regex to capture the quantity and unit
pattern = r"(\d+(?:[.,]\d+)?)\s*(kg|kilo|g|gr|gram|ml|liter|l|stuk|stuks|st\.?|pak|pakket|bos|cups?|zakjes?)"

df_dirk[["quantity", "unit"]] = (
    df_dirk["portion"]
      .str.lower()
      .str.extract(pattern)
)

# Normalize decimal commas
df_dirk["quantity"] = (
    df_dirk["quantity"]
      .str.replace(",", ".")
      .astype(float)
)

# Normalize units
df_dirk["unit"] = df_dirk["unit"].replace(UNIT_MAP)

# Convert units to kg and l
mask = df_dirk["unit"].isin(["g", "ml"])
df_dirk.loc[mask, "quantity"] /= 1000

# replace g and ml units
df_dirk["unit"] = df_dirk["unit"].replace({
    "g": "kg",
    "ml": "l",
})

# Calculate real quantity
# some 'portion' still contains int x int unit
pattern = r"(\d+)\s*x\s*(\d+(?:[.,]\d+)?)\s*([a-zA-Z]+)" # amount x portion unit
mask = df_dirk["portion"].str.contains(pattern, na=False)
extracted = df_dirk.loc[mask, "portion"].str.extract(pattern)

amount = extracted[0].astype(int)
size = extracted[1].str.replace(",", ".", regex=False).astype(float)
unit = extracted[2].str.lower()
factor = (
    unit.map({
        "ml": 1 / 1000,
        "cl": 1 / 100,
        "l": 1,
        "liter": 1,
        "g": 1 / 1000,
        "gram": 1 / 1000,
        "gr": 1 / 1000,
        "kg": 1,
    }).fillna(1)
)
df_dirk.loc[mask, "quantity"] = amount * size * factor

# Make floats
numeric_cols = [
    "price", "kcal", "quantity",
    "carbs_g", "fat_g", "protein_g", "salt_g"
]

df_dirk[numeric_cols] = df_dirk[numeric_cols].astype(float)

# endregion

# region AH

path_ah = "data/raw/AH.csv"

COL_MAP = {
    "prod_desc": "name",
    "quantity_value": "quantity",
    "quantity_unit": "unit",
    "D2026_03_14": "price"
}

COLS = [
    "prod_desc",
    "D2026_03_14", # latest date for price
    "quantity_value", 
    "quantity_unit",
    "ingredients",
    "labels",
    "nutrients"
]

df_ah = pd.read_csv(path_ah)
df_ah = df_ah[COLS]
df_ah.rename(columns=COL_MAP, inplace=True)

# Drop non-food items (nutrients = NaN)
df_ah.dropna(subset=["nutrients"], inplace=True)

# Parsing nutrients

def parse_nutrients(s: str):
    nutrients = {
        "kcal": np.nan,
        "carbs_g": np.nan,
        "fat_g": np.nan,
        "protein_g": np.nan,
        "salt_g": np.nan,
        "saturated_fat_g": np.nan,
        "sugars_g": np.nan,
        "fiber_g": np.nan,
    }

    translation_map = {
        "Vet": "fat_g",
        "waarvan verzadigd": "saturated_fat_g",
        "Koolhydraten": "carbs_g",
        "waarvan suikers": "sugars_g",
        "Voedingsvezel": "fiber_g",
        "Eiwitten": "protein_g",
        "Zout": "salt_g",
    }

    split_outer = s.split("\\", maxsplit=2) # '\\'
    if len(split_outer) < 2:
        return nutrients
    
    # Parse kcal
    energy = split_outer[1]
    kcal_pattern = r"(\d+)\s*kcal" # r"\((\d+)\s*kcal\)"
    m_kcal = re.search(kcal_pattern, energy)

    if m_kcal:
        kcal = int(m_kcal.group(1))
        nutrients["kcal"] = kcal
    else:
        # print("Kcal regex failed for:", repr(energy))
        pass

    # Parse nutrients
    if len(split_outer) > 2:
        split_inner = split_outer[2].split("g\\")

        pattern = r"(.+?)([\d.,]+)$"
        for p in split_inner[:-1]: # skip the part after 'Zout'
            m = re.match(pattern, p.strip())

            if not m:
                # print("Regex failed for:", repr(p))
                continue

            nutrient, value = m.groups()
            nutrient_translated = translation_map.get(nutrient)
            if nutrient_translated:
                nutrients[nutrient_translated] = float(value.replace(",", "."))
    
    return nutrients

nutrient_columns = df_ah["nutrients"].apply(parse_nutrients).apply(pd.Series)
df_ah = pd.concat([df_ah, nutrient_columns], axis=1)
df_ah.drop(columns="nutrients", inplace=True)

# Normalize units
df_ah["unit"] = df_ah["unit"].replace(UNIT_MAP)

# Convert quantities based on units
df_ah.loc[df_ah["unit"] == "g", "quantity"] /= 1000
df_ah.loc[df_ah["unit"] == "ml", "quantity"] /= 1000
df_ah.loc[df_ah["unit"] == "cl", "quantity"] /= 100

# replace g, ml and cl units
df_ah["unit"] = df_ah["unit"].replace({
    "g": "kg",
    "ml": "l",
    "cl": "l",
})

# Regex to capture the name only
pattern_name = r"\s*(?:\d+(?:[.,]\d+)?\s*(?:kg|kilo|g|gr|gram|ml|liter|l|stuk|stuks|st\.?|pak|pakket|bos|cups?|zakjes?)|\d+\s*-?\s*pack)\s*$"

df_ah["name_only"] = df_ah["name"].str.replace(pattern_name, "", case=False, regex=True)

# endregion

# region Feature Engineering

idf_dict = get_idf_dict(df_dirk, df_ah)

df_dirk = add_features(df_dirk, idf_dict, supermarket="dirk")
df_ah = add_features(df_ah, idf_dict, supermarket="ah")

# Storing
dir = "data/processed"
os.makedirs(dir, exist_ok=True)

out_path_dirk = "data/processed/dirk.csv"
df_dirk.to_csv(out_path_dirk, index=False)
print(f"Saved to {out_path_dirk}")

out_path_ah = "data/processed/ah.csv"
df_ah.to_csv(out_path_ah, index=False)
print(f"Saved to {out_path_ah}")
