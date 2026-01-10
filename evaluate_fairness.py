import pandas as pd
from sklearn.metrics import accuracy_score

EMOTION_NAMES = [
    "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"
]

def overall_accuracy(df):
    return accuracy_score(df["true_label"], df["pred_label"])

def accuracy_by_group(df, group_col):
    groups = df[group_col].dropna().unique()
    stats = []
    for g in groups:
        sub = df[df[group_col] == g]
        acc = accuracy_score(sub["true_label"], sub["pred_label"])
        stats.append((g, len(sub), acc))
    return sorted(stats, key=lambda x: x[0])

def main():
    df = pd.read_csv("outputs/merged_predictions_with_demographics.csv")

    # Drop rows with missing demographics for group-based analysis
    df_gender = df.dropna(subset=["apparent_gender"])
    df_race = df.dropna(subset=["apparent_race"])

    print(f"Total test samples: {len(df)}")
    print(f"With gender labels: {len(df_gender)}")
    print(f"With race labels:   {len(df_race)}")

    # 1. Overall accuracy
    overall_acc = overall_accuracy(df)
    print(f"\nOverall test accuracy: {overall_acc:.4f}")

    # 2. Accuracy by apparent_gender
    print("\nAccuracy by apparent gender:")
    gender_stats = accuracy_by_group(df_gender, "apparent_gender")
    for g, n, acc in gender_stats:
        print(f"  {g:10s}  n={n:4d}  acc={acc:.4f}")

    if gender_stats:
        gender_disparity = max(acc for _, _, acc in gender_stats) - min(acc for _, _, acc in gender_stats)
        print(f"  -> gender accuracy disparity (max-min): {gender_disparity:.4f}")

    # 3. Accuracy by apparent_race
    print("\nAccuracy by apparent race:")
    race_stats = accuracy_by_group(df_race, "apparent_race")
    for r, n, acc in race_stats:
        print(f"  {r:15s}  n={n:4d}  acc={acc:.4f}")

    if race_stats:
        race_disparity = max(acc for _, _, acc in race_stats) - min(acc for _, _, acc in race_stats)
        print(f"  -> race accuracy disparity (max-min): {race_disparity:.4f}")

if __name__ == "__main__":
    main()
