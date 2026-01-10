import pandas as pd

df = pd.read_csv("outputs/deepface_demographics.csv")

print("\n=== GENDER DISTRIBUTION ===")
gender_counts = df["apparent_gender"].value_counts()
gender_props = df["apparent_gender"].value_counts(normalize=True)

print(gender_counts)
print("\nProportions:\n", gender_props)

print("\n=== RACE DISTRIBUTION ===")
race_counts = df["apparent_race"].value_counts()
race_props = df["apparent_race"].value_counts(normalize=True)

print(race_counts)
print("\nProportions:\n", race_props)

# Optional: joint distribution
print("\n=== RACE × GENDER ===")
joint = pd.crosstab(df["apparent_race"], df["apparent_gender"], normalize="index")
print(joint)
