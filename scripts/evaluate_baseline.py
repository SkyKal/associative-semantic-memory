
from pathlib import Path
import sys

import pandas as pd

# Make the project's src directory available for imports.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from semantic_memory.evaluation import (
    evaluate_ranked_retrieval,
    summarise_evaluation
)


# --------------------------------------------------
# 1. File paths and settings
# --------------------------------------------------

DATA_PATH = PROJECT_ROOT / "data/processed/swow_r100_clean.csv"

SPLIT_PATH = PROJECT_ROOT / "data/processed/participant_splits.csv"

ASSOCIATION_PATH = (
    PROJECT_ROOT / "data/processed/statistical_associations.csv"
)

REPORT_DIR = PROJECT_ROOT / "reports"

K_VALUES = (1, 5, 10)

RESPONSE_COLUMNS = ["R1", "R2", "R3"]


# --------------------------------------------------
# 2. Load held-out test data
# --------------------------------------------------

def load_test_data():

    df = pd.read_csv(
        DATA_PATH,
        usecols=[
            "participantID", "cue",
            "R1", "R2", "R3"
        ],
        dtype=str,
        keep_default_na=False
    )

    splits = pd.read_csv(
        SPLIT_PATH,
        dtype=str,
        keep_default_na=False
    )

    df = df.merge(
        splits,
        on="participantID",
        how="left",
        validate="many_to_one"
    )

    assert df["split"].notna().all()

    test = df[df["split"] == "test"].copy()

    print("=== TEST DATA ===")
    print(f"Test observations: {len(test):,}")
    print(f"Test participants: {test['participantID'].nunique():,}")
    print(f"Test cues: {test['cue'].nunique():,}")

    return test


# --------------------------------------------------
# 3. Load the statistical model
# --------------------------------------------------

def load_association_model():

    associations = pd.read_csv(
    ASSOCIATION_PATH,
    dtype={"cue": str, "R1": str},
    keep_default_na=False
    )

    # Ensure each cue-response pair appears only once.
    assert not associations.duplicated(
        subset=["cue", "R1"]
    ).any()

    # Reconstruct each cue's ranked candidate list.
    associations = associations.sort_values(
        by=["cue", "probability", "count", "R1"],
        ascending=[True, False, False, True]
    )

    ranked_by_cue = {
        cue: group["R1"].tolist()
        for cue, group in associations.groupby("cue")
    }

    print("\n=== STATISTICAL MODEL ===")
    print(f"Cues in model: {len(ranked_by_cue):,}")
    print(f"Association pairs: {len(associations):,}")

    return ranked_by_cue


# --------------------------------------------------
# 4. Evaluate R1, R2 and R3
# --------------------------------------------------

def evaluate_all_positions(test, ranked_by_cue):

    all_results = []

    for response_col in RESPONSE_COLUMNS:

        print(f"\nEvaluating {response_col}...")

        position_results = evaluate_ranked_retrieval(
            ranked_by_cue=ranked_by_cue,
            test_data=test,
            response_col=response_col,
            k_values=K_VALUES
        )

        position_results.insert(
            0,
            "response_position",
            response_col
        )

        all_results.append(position_results)

    return pd.concat(
        all_results,
        ignore_index=True
    )


# --------------------------------------------------
# 5. Save and display results
# --------------------------------------------------

def main():

    test = load_test_data()

    ranked_by_cue = load_association_model()

    per_cue_results = evaluate_all_positions(
        test,
        ranked_by_cue
    )

    summary = summarise_evaluation(
        per_cue_results
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    per_cue_path = (
        REPORT_DIR / "baseline_per_cue_metrics.csv"
    )

    summary_path = (
        REPORT_DIR / "baseline_summary.csv"
    )

    per_cue_results.to_csv(
        per_cue_path,
        index=False
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    print("\n=== BASELINE EVALUATION ===")

    print(
        summary.round(4).to_string(index=False)
    )

    print(f"\nPer-cue results saved to: {per_cue_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
