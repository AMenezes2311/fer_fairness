import os
import pandas as pd
from tqdm import tqdm
from deepface import DeepFace

INPUT_CSV = "data/fer_image_paths.csv"
OUTPUT_CSV = "outputs/deepface_demographics.csv"
BATCH_SIZE = 1000

def main():
    df = pd.read_csv(INPUT_CSV)

    os.makedirs("outputs", exist_ok=True)

    # Load existing progress if file exists
    if os.path.exists(OUTPUT_CSV):
        existing = pd.read_csv(OUTPUT_CSV)
        completed = set(existing["dataset_index"].values)
        print(f"Resuming: {len(completed)} already processed")
    else:
        existing = pd.DataFrame()
        completed = set()
        print("Starting fresh")

    df_todo = df[~df["dataset_index"].isin(completed)].copy()
    print("Remaining images:", len(df_todo))

    results = []

    for start in range(0, len(df_todo), BATCH_SIZE):
        batch = df_todo.iloc[start:start+BATCH_SIZE]
        print(f"\n--- Processing batch {start} to {start + len(batch)} ---")

        for _, row in tqdm(batch.iterrows(), total=len(batch)):
            img_path = row["img_path"]
            dataset_index = int(row["dataset_index"])

            try:
                analysis = DeepFace.analyze(
                    img_path=img_path,
                    actions=["gender", "race"],
                    enforce_detection=False,
                    detector_backend="retinaface"
                )

                if isinstance(analysis, list):
                    analysis = analysis[0]

                gender = analysis.get("dominant_gender", None)
                race = analysis.get("dominant_race", None)
                gender_prob = analysis.get("gender", {})
                race_prob = analysis.get("race", {})

            except Exception as e:
                print(f"Error on {img_path}: {e}")
                gender = None
                race = None
                gender_prob = {}
                race_prob = {}

            results.append({
                "dataset_index": dataset_index,
                "img_path": img_path,
                "apparent_gender": gender,
                "apparent_race": race,
                "gender_scores": str(gender_prob),
                "race_scores": str(race_prob),
            })

        # Save batch results
        batch_df = pd.DataFrame(results)
        if existing.empty:
            combined = batch_df
        else:
            combined = pd.concat([existing, batch_df], ignore_index=True)

        combined.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved progress: {len(combined)} predictions")

        existing = combined.copy()
        results = []  # reset for next batch

    print("\n All batches complete")
    print(f"Final output: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
