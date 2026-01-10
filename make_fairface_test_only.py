# make_fairface_test_only.py

import pandas as pd

df = pd.read_csv("data/fer_image_paths.csv")
test_df = df[df["Usage"] == "PrivateTest"].copy()
test_df.to_csv("data/fairface_test_only.csv", index=False)

print("Saved data/fairface_test_only.csv with", len(test_df), "rows")
