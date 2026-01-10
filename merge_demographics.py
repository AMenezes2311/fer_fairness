import pandas as pd

def main():
    # 1. Load model predictions on FER2013 test set
    preds = pd.read_csv("outputs/test_predictions.csv")
    # preds currently has columns: index, true_label, pred_label
    # Remove probability columns if they exist
    prob_cols = [col for col in preds.columns if col.startswith("prob_")]
    preds = preds.drop(columns=prob_cols)

    # Make sure we have a 'dataset_index' column to merge on
    if "dataset_index" not in preds.columns:
        if "index" in preds.columns:
            preds = preds.rename(columns={"index": "dataset_index"})
        else:
            raise KeyError(
                "Neither 'dataset_index' nor 'index' found in predictions CSV. "
                "Available columns: " + ", ".join(preds.columns)
            )

    # 2. Load DeepFace demographics (test-only)
    demo = pd.read_csv("outputs/deepface_demographics.csv")

    # Ensure demographics also has the right key column
    if "dataset_index" not in demo.columns:
        if "index" in demo.columns:
            demo = demo.rename(columns={"index": "dataset_index"})
        else:
            raise KeyError(
                "Neither 'dataset_index' nor 'index' found in demographics CSV. "
                "Available columns: " + ", ".join(demo.columns)
            )

    demo["apparent_gender"] = demo["apparent_gender"].str.lower()
    demo["apparent_race"] = demo["apparent_race"].str.lower()

    # 3. Merge on dataset_index
    merged = preds.merge(
        demo[["dataset_index", "apparent_gender", "apparent_race"]],
        on="dataset_index",
        how="left"
    )

    out_path = "outputs/merged_predictions_with_demographics.csv"
    merged.to_csv(out_path, index=False)

    print(f"Saved merged data to {out_path}")
    print("Total rows:", len(merged))
    print("Missing gender:", merged["apparent_gender"].isna().sum())
    print("Missing race:", merged["apparent_race"].isna().sum())


if __name__ == "__main__":
    main()
