from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data/raw/SWOW-EN18/SWOW-EN.R100.20180827.csv"

# Preserve raw values so we can see how missing responses are encoded.
df = pd.read_csv(
    DATA_PATH,
    usecols=["participantID", "cue", "R1"],
    keep_default_na=False
)

print("DATA QUALITY CHECK")
print("=" * 50)

# 1. Missing and unusual response values
print("\nR1 value counts, including empty values:")
print(df["R1"].value_counts(dropna=False).head(15))

print("\nEmpty R1 responses:", (df["R1"] == "").sum())
print("Empty cues:", (df["cue"] == "").sum())

# 2. Whitespace and capitalization
print("\nR1 values with leading/trailing whitespace:",
      (df["R1"] != df["R1"].str.strip()).sum())

print("Cues with leading/trailing whitespace:",
      (df["cue"] != df["cue"].str.strip()).sum())

print("\nExample unusual R1 values:")
print(df.loc[
    (df["R1"] != df["R1"].str.strip()) | (df["R1"] == ""),
    "R1"
].head(10).tolist())

# 3. Duplicate participant-cue pairs
duplicates = df[
    df.duplicated(subset=["participantID", "cue"], keep=False)
].sort_values(["participantID", "cue"])

print("\nDuplicate participant-cue rows:")
print(duplicates.to_string(index=False))

# 4. Basic consistency checks
print("\nUnique participants:", df["participantID"].nunique())
print("Unique cues:", df["cue"].nunique())
print("Unique R1 responses:", df["R1"].nunique())

print("\nCheck complete.")