# evaluate_bias_template.py

"""
Template for bias metrics AFTER you have:
- test_predictions.csv from train_fer.py
- demographics.csv from your FairFace script

You should merge them into one CSV, e.g.:
    merged.csv with columns:
        dataset_index, true_label, pred_label,
        apparent_gender, apparent_race
"""

import pandas as pd
import numpy as np
from sklearn.metrics import recall_score
from math import log

def normalized_std(proportions):
    """Representational bias: normalized standard deviation of group proportions."""
    proportions = np.array(proportions, dtype=float)
    return proportions.std() / (proportions.mean() + 1e-8)


def compute_representational_bias(df, group_col):
    group_counts = df[group_col].value_counts(normalize=True).sort_index()
    return normalized_std(group_counts.values), group_counts


def compute_group_recall(df, group_col, label_col="true_label", pred_col="pred_label"):
    """
    Recall per group (overall, not per emotion class).
    """
    recalls = {}
    for g, gdf in df.groupby(group_col):
        recalls[g] = recall_score(gdf[label_col], gdf[pred_col], average="macro")
    return recalls


def compute_class_group_recall(df, group_col, num_classes=7):
    """
    Recall for each (class, group) pair.
    Returns a nested dict: recall[class][group] = value
    """
    result = {}
    for c in range(num_classes):
        result[c] = {}
        cdf = df[df["true_label"] == c]
        for g, gdf in cdf.groupby(group_col):
            if len(gdf) == 0:
                result[c][g] = np.nan
            else:
                y_true = (gdf["true_label"] == c).astype(int)
                y_pred = (gdf["pred_label"] == c).astype(int)
                result[c][g] = recall_score(y_true, y_pred)
    return result


def compute_disparity_score(class_group_recall):
    """
    Simple disparity metric: average (max - min) recall across groups, over classes.
    Ignores NaNs.
    """
    diffs = []
    for c, recs in class_group_recall.items():
        vals = [v for v in recs.values() if not np.isnan(v)]
        if len(vals) > 1:
            diffs.append(max(vals) - min(vals))
    if len(diffs) == 0:
        return np.nan
    return float(np.mean(diffs))


def main():
    merged_path = "outputs/merged_predictions_with_demographics.csv"
    df = pd.read_csv(merged_path)
    
    print("=== Representational bias: gender ===")
    rep_bias_gender, gender_counts = compute_representational_bias(df, "apparent_gender")
    print("Normalized std:", rep_bias_gender)
    print("Group proportions:\n", gender_counts)

    print("\n=== Group recall: gender ===")
    gender_recalls = compute_group_recall(df, "apparent_gender")
    for g, r in gender_recalls.items():
        print(f"{g}: recall={r:.4f}")

    print("\n=== Class-group recall & disparity: gender ===")
    num_classes = df["true_label"].nunique()
    class_group_recalls = compute_class_group_recall(df, "apparent_gender", num_classes=num_classes)
    disparity_gender = compute_disparity_score(class_group_recalls)
    print("Disparity (avg max-min across classes):", disparity_gender)


if __name__ == "__main__":
    main()
