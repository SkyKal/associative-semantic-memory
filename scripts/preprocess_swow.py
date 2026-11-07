
from pathlib import Path
import pandas as pd

RAW_PATH = Path(
    "data/raw/SWOW-EN18/SWOW-EN.R100.20180827.csv"
)

OUTPUT_PATH = Path(
    "data/processed/swow_r100_clean.csv"
)

COLUMNS = [
    "id", "participantID", "cue",
    "R1", "R2", "R3"
]

RESPONSE_COLUMNS = ["R1", "R2", "R3"]

def main():
    df = pd.read_csv(
        RAW_PATH,
        usecols=COLUMNS,
        dtype=str,
        keep_default_na=False,
        low_memory=False
    )

    # Remove incidental whitespace without changing word forms.
    for col in ["cue"] + RESPONSE_COLUMNS:
        df[col] = df[col].str.strip()

    # Identify exact missing markers.
    for col in RESPONSE_COLUMNS:
        df[f"{col}_available"] = df[col] != "NA"
        df.loc[df[col] == "NA", col] = ""

    # Basic integrity checks.
    print("=== PREPROCESSING SUMMARY ===")
    print(f"Rows: {len(df):,}")
    print(f"Participants: {df['participantID'].nunique():,}")
    print(f"Cues: {df['cue'].nunique():,}")

    print("\nMissing responses:")
    for col in RESPONSE_COLUMNS:
        print(
            f"{col}: {(~df[f'{col}_available']).sum():,}"
        )

    cue_counts = df.groupby("cue").size()

    print("\nRows per cue:")
    print(f"Minimum: {cue_counts.min()}")
    print(f"Maximum: {cue_counts.max()}")

    duplicates = df.duplicated(
        subset=["participantID", "cue"],
        keep=False
    )

    print(
        "\nRows involved in repeated participant-cue pairs:",
        duplicates.sum()
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved processed dataset to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
