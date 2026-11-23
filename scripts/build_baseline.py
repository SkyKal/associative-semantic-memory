
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. File paths and settings
# --------------------------------------------------

DATA_PATH = Path("data/processed/swow_r100_clean.csv")
SPLIT_PATH = Path("data/processed/participant_splits.csv")

OUTPUT_PATH = Path("data/processed/statistical_associations.csv")

TOP_K = 10
EXAMPLE_CUE = "Abel"


# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

def load_training_data():

    df = pd.read_csv(
        DATA_PATH,
        usecols=["participantID", "cue", "R1"],
        dtype=str,
        keep_default_na=False
    )

    splits = pd.read_csv(
        SPLIT_PATH,
        dtype=str,
        keep_default_na=False
    )

    # Attach each participant's split assignment.
    df = df.merge(
        splits,
        on="participantID",
        how="left",
        validate="many_to_one"
    )

    # Check that every observation has a split.
    assert df["split"].notna().all()

    # Select all training observations.
    assigned_train = df[df["split"] == "train"]

    # Keep only observations with a valid R1 response.
    train = assigned_train[
        assigned_train["R1"] != ""
    ].copy()

    # Identify participants with no valid R1 responses.
    valid_participants = set(train["participantID"])
    assigned_participants = set(
        assigned_train["participantID"]
    )

    no_valid_r1 = assigned_participants - valid_participants

    # Report dataset information.
    print("=== BASELINE TRAINING DATA ===")
    print(f"Total observations: {len(df):,}")
    print(f"Assigned training participants: "
          f"{len(assigned_participants):,}")
    print(f"Participants without valid R1: "
          f"{len(no_valid_r1):,}")
    print(f"Training observations: {len(train):,}")
    print(f"Training participants with valid R1: "
          f"{train['participantID'].nunique():,}")
    print(f"Unique training cues: {train['cue'].nunique():,}")
    print(f"Unique training responses: {train['R1'].nunique():,}")

    return train


# --------------------------------------------------
# 3. Calculate cue-response frequencies
# --------------------------------------------------

def calculate_association_counts(train):

    pair_counts = (
        train
        .groupby(["cue", "R1"])
        .size()
        .reset_index(name="count")
    )

    print("\n=== ASSOCIATION FREQUENCIES ===")
    print(
        pair_counts.head(15).to_string(index=False)
    )

    print(
        f"\nTotal unique cue-response pairs: "
        f"{len(pair_counts):,}"
    )

    return pair_counts


# --------------------------------------------------
# 4. Calculate conditional probabilities
# --------------------------------------------------

def calculate_probabilities(pair_counts):

    # Calculate total valid responses for each cue.
    pair_counts["cue_total"] = (
        pair_counts
        .groupby("cue")["count"]
        .transform("sum")
    )

    # Calculate P(response | cue).
    pair_counts["probability"] = (
        pair_counts["count"] / pair_counts["cue_total"]
    )

    return pair_counts


# --------------------------------------------------
# 5. Verify probability distributions
# --------------------------------------------------

def verify_probabilities(pair_counts):

    probability_sums = (
        pair_counts
        .groupby("cue")["probability"]
        .sum()
    )

    print("\n=== PROBABILITY CHECK ===")
    print(probability_sums.describe())

    # Every cue's response probabilities should sum to 1.
    assert np.allclose(probability_sums, 1.0)

    print("\nAll cue probability distributions sum to 1.")


# --------------------------------------------------
# 6. Retrieve associated words
# --------------------------------------------------

def retrieve(cue, pair_counts, top_k=10):

    candidates = pair_counts[
        pair_counts["cue"] == cue
    ]

    # Sort by probability, then count.
    ranked = candidates.sort_values(
        by=["probability", "count", "R1"],
        ascending=[False, False, True]
    )

    return (
        ranked[["R1", "count", "probability"]]
        .head(top_k)
        .rename(columns={"R1": "response"})
        .reset_index(drop=True)
    )


# --------------------------------------------------
# 7. Main execution
# --------------------------------------------------

def main():

    # Load training observations.
    train = load_training_data()

    # Calculate association frequencies.
    pair_counts = calculate_association_counts(train)

    # Calculate conditional probabilities.
    pair_counts = calculate_probabilities(pair_counts)

    # Verify the probabilities.
    verify_probabilities(pair_counts)

    # Save the statistical model's association table.
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    pair_counts.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(f"\nAssociation table saved to: {OUTPUT_PATH}")

    # Demonstrate retrieval.
    print(f"\n=== TOP {TOP_K} ASSOCIATIONS FOR {EXAMPLE_CUE!r} ===")

    predictions = retrieve(
        EXAMPLE_CUE,
        pair_counts,
        TOP_K
    )

    if predictions.empty:
        print("Cue not found in training data.")
    else:
        print(predictions.to_string(index=False))


if __name__ == "__main__":
    main()
