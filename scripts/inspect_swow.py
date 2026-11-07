
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
print("Looking for data in:", DATA_DIR.resolve())
print("Directory exists:", DATA_DIR.exists())

# Locate the balanced R100 dataset.
matches = list(DATA_DIR.rglob("SWOW-EN.R100*.csv"))

if not matches:
    raise FileNotFoundError(
        "Could not find SWOW-EN.R100 inside data/raw"
    )

r100_path = matches[0]

print("=" * 60)
print("SWOW-EN DATASET INSPECTION")
print("=" * 60)

# First inspect the available CSV files and their schemas.
for path in sorted(DATA_DIR.rglob("*.csv")):
    header = pd.read_csv(path, nrows=0)

    print(f"\nFile: {path.name}")
    print(f"Columns: {header.columns.tolist()}")

# Now load only the columns relevant to our investigation.
header = pd.read_csv(r100_path, nrows=0)
available = {column.lower(): column for column in header.columns}

required = ["cue", "r1", "r2", "r3"]

missing = [column for column in required if column not in available]

if missing:
    raise ValueError(f"Missing expected columns: {missing}")

selected = [
    available[column]
    for column in ["participantid", "cue", "r1", "r2", "r3"]
    if column in available
]

df = pd.read_csv(r100_path, usecols=selected)

cue_col = available["cue"]
r1_col = available["r1"]
participant_col = available.get("participantid")

print("\n" + "=" * 60)
print("R100 DATASET SUMMARY")
print("=" * 60)

print(f"Total rows: {len(df):,}")
print(f"Unique cues: {df[cue_col].nunique():,}")
print(f"Unique R1 responses: {df[r1_col].nunique():,}")
print(f"Missing R1 responses: {df[r1_col].isna().sum():,}")

if participant_col:
    print(f"Unique participants: {df[participant_col].nunique():,}")

    duplicate_pairs = df.duplicated(
        subset=[participant_col, cue_col]
    ).sum()

    print(f"Duplicate participant-cue pairs: {duplicate_pairs:,}")

    participants_per_cue = (
        df.groupby(cue_col)[participant_col].nunique()
    )

    print("\nParticipants per cue:")
    print(participants_per_cue.describe())

r1_per_cue = (
    df.groupby(cue_col)[r1_col]
    .count()
)

print("\nValid R1 responses per cue:")
print(r1_per_cue.describe())

print("\nExample associations (no participant metadata):")

print(
    df[[cue_col, r1_col]].head(10).to_string(index=False)
)

print("\nInspection complete!")
