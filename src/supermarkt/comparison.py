import pandas as pd

def price_comparison(pairs: pd.DataFrame , own_brand=False):
    if own_brand:
        print("===== Price comparison (Own Brand) =====")
        pairs = pairs[pairs["own_brand_dirk"] == 1]
    else:
        print("===== Price comparison (General) =====")

    ah_cheaper = (pairs["price_ah"] < pairs["price_dirk"]).sum()
    dirk_cheaper = (pairs["price_dirk"] < pairs["price_ah"]).sum()
    equal = (pairs["price_ah"] == pairs["price_dirk"]).sum()

    pairs = pairs[pairs["rel_price_diff"].abs() < 0.4] # skip outliers (promotions)
    print(f"Number of non-outliers: {len(pairs)}")

    avg_rel_diff = pairs["rel_price_diff"].mean() # < 0 AH is cheaper, > 0 AH is more expensive

    if avg_rel_diff > 0:
        print(f"Dirk is {avg_rel_diff:.1%} cheaper on average")
    else:
        print(f"AH is {-avg_rel_diff:.1%} cheaper on average")

    total_ah = pairs["price_ah"].sum()
    total_dirk = pairs["price_dirk"].sum()

    print(f"Total Basket Price (AH): €{total_ah:.2f}")
    print(f"Total Basket Price (Dirk): €{total_dirk:.2f}")

    return ah_cheaper, dirk_cheaper, equal

# region Correct Pairs

# Write Correct Pairs
labeled_path = "data/results/labeled_pairs.csv"
pairs_path = "data/results/pairs.csv"

part_labeled_pairs = pd.read_csv(labeled_path)
pairs = pd.read_csv(pairs_path)
df = part_labeled_pairs.merge(pairs)

labeled_pairs = df.dropna(subset=["label"])

correct_pairs = labeled_pairs[labeled_pairs["label"] == 1]

correct_pairs["abs_price_diff"] = correct_pairs["price_ah"] - correct_pairs["price_dirk"]
correct_pairs["rel_price_diff"] = (
    correct_pairs["abs_price_diff"]
    / correct_pairs[["price_ah", "price_dirk"]].max(axis=1)
)

correct_pairs.to_csv("data/results/correct_pairs.csv", index=False)
# correct_pairs = pd.read_csv("data/results/correct_pairs.csv")

print("\n========== Correct Labelled Pairs ==========\n")

# Average price dirk vs ah
ah_cheaper, dirk_cheaper, equal = price_comparison(pairs=correct_pairs, own_brand=False)

# Average price dirk vs ah (Own Brand)
ah_cheaper_own, dirk_cheaper_own, equal_own = price_comparison(pairs=correct_pairs, own_brand=True)

# 'Wins' per supermarket
wins_df = pd.DataFrame(
    [
        [dirk_cheaper, equal, ah_cheaper],
        [dirk_cheaper_own, equal_own, ah_cheaper_own]
    ],
    columns=["Cheaper Dirk", "Equal", "Cheaper AH"],
    index=["Overall", "Own Brand"]
)

print("\nComparison:")
print(wins_df)

# endregion

# region Predicted Pairs

print("\n========== Predicted Pairs ==========\n")
pairs = pd.read_csv("data/results/pairs.csv")
predicted_pairs = pairs[pairs["score"] > 0.78] # 0.78 from f1-optimization result
predicted_pairs["abs_price_diff"] = predicted_pairs["price_ah"] - predicted_pairs["price_dirk"]
predicted_pairs["rel_price_diff"] = (
    predicted_pairs["abs_price_diff"]
    / predicted_pairs[["price_ah", "price_dirk"]].max(axis=1)
)

# Average price dirk vs ah
ah_cheaper, dirk_cheaper, equal = price_comparison(pairs=predicted_pairs, own_brand=False)

# Average price dirk vs ah (Own Brand)
ah_cheaper_own, dirk_cheaper_own, equal_own = price_comparison(pairs=predicted_pairs, own_brand=True)

# 'Wins' per supermarket
wins_df = pd.DataFrame(
    [
        [dirk_cheaper, equal, ah_cheaper],
        [dirk_cheaper_own, equal_own, ah_cheaper_own]
    ],
    columns=["Cheaper Dirk", "Equal", "Cheaper AH"],
    index=["Overall", "Own Brand"]
)

print("\nComparison:")
print(wins_df)

# endregion
