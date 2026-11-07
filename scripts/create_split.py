
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_PATH = Path("data/processed/swow_r100_clean.csv")
OUTPUT_PATH = Path("data/processed/participant_splits.csv")

RANDOM_STATE = 42


def main():
    df = pd.read_csv(
        INPUT_PATH,
        usecols=["participantID", "cue"],
        dtype=str,
        keep_default_na=False
    )

    participants = df["participantID"].drop_duplicates().to_numpy()

    # First separate 70% training from 30% remaining.
    train_ids, remaining_ids = train_test_split(
        participants,
        test_size=0.30,
        random_state=RANDOM_STATE
    )

    # Divide the remaining 30% equally.
    validation_ids, test_ids = train_test_split(
        remaining_ids,
        test_size=0.50,
        random_state=RANDOM_STATE
    )

    # Verify that participants do not overlap.
    assert set(train_ids).isdisjoint(validation_ids)
    assert set(train_ids).isdisjoint(test_ids)
    assert set(validation_ids).isdisjoint(test_ids)

    assignments = pd.DataFrame({
        "participantID": (
            list(train_ids)
            + list(validation_ids)
            + list(test_ids)
        ),
        "split": (
            ["train"] * len(train_ids)
            + ["validation"] * len(validation_ids)
            + ["test"] * len(test_ids)
        )
    })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(OUTPUT_PATH, index=False)

    print("=== PARTICIPANT SPLIT ===")
    print(f"Total participants: {len(participants):,}")

    for name, ids in [
        ("Training", train_ids),
        ("Validation", validation_ids),
        ("Testing", test_ids)
    ]:
        print(f"{name}: {len(ids):,} ({len(ids)/len(participants)*100:.2f}%)")

    # Check how participants are distributed across cues.
    split_map = dict(
        zip(assignments["participantID"], assignments["split"])
    )

    df["split"] = df["participantID"].map(split_map)

    cue_counts = (
        df.groupby(["cue", "split"])["participantID"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(columns=["train", "validation", "test"], fill_value=0)
    )

    print("\n=== PARTICIPANTS PER CUE ===")

    for split in ["train", "validation", "test"]:
        values = cue_counts[split]
        print(
            f"{split}: min={values.min()}, "
            f"median={values.median()}, "
            f"max={values.max()}"
        )

    print(f"\nSplit assignments saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
