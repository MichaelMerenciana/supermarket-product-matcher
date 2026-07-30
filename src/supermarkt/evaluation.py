import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, precision_score, recall_score

labeled_path = "data/results/labeled_pairs.csv"
pairs_path = "data/results/pairs.csv"

part_labeled_pairs = pd.read_csv(labeled_path)
pairs = pd.read_csv(pairs_path)
df = part_labeled_pairs.merge(pairs)

labeled_pairs = df.dropna(subset=["label"])

# Dealing with a float (similarity) to a label (binary) classification problem -> use ROC-AUC and PR-AUC

roc_auc = roc_auc_score(labeled_pairs["label"], labeled_pairs["score"])
print(f"ROC-AUC Score: {roc_auc:.4f}")

precision, recall, thresholds = precision_recall_curve(labeled_pairs["label"], labeled_pairs["score"])
pr_auc = auc(recall, precision)
print(f"PR-AUC Score: {pr_auc:.4f}")

# Making a custom threshold
thresholds = np.arange(0, 1, 0.01)

results = []

for threshold in thresholds:
    predictions = labeled_pairs["score"] >= threshold
    
    results.append({
        "threshold": threshold,
        "f1": f1_score(labeled_pairs["label"], predictions),
        "precision": precision_score(labeled_pairs["label"], predictions),
        "recall": recall_score(labeled_pairs["label"], predictions),
    })

results_df = pd.DataFrame(results)

best = results_df.loc[results_df["f1"].idxmax()]

print("\nBest Custom Threshold:")
print(best)