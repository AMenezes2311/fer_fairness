# export_fer_images.py

import os
import numpy as np
import pandas as pd
from PIL import Image

def main():
    csv_path = "data/fer2013.csv"
    out_dir = "data/fer_images"
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    df["dataset_index"] = df.index  # stable ID

    image_paths = []

    for i, row in df.iterrows():
        pixels = np.fromstring(row["pixels"], dtype=np.uint8, sep=" ")
        img = pixels.reshape(48, 48)

        img = Image.fromarray(img)
        img = img.resize((224, 224))

        filename = f"{int(row['dataset_index'])}.png"
        filepath = os.path.join(out_dir, filename)
        img.save(filepath)

        abs_filepath = os.path.abspath(filepath) 

        image_paths.append(
            {
                "dataset_index": int(row["dataset_index"]),
                "Usage": row["Usage"],
                "img_path": abs_filepath,  
            }
        )

    paths_df = pd.DataFrame(image_paths)
    paths_df.to_csv("data/fer_image_paths.csv", index=False)
    print("Saved images to", out_dir)
    print("Saved mapping to data/fer_image_paths.csv")

if __name__ == "__main__":
    main()
