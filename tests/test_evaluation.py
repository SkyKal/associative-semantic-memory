
import numpy as np
import pandas as pd

from semantic_memory.evaluation import evaluate_ranked_retrieval


def test_modal_agreement_and_frequency_ndcg():
    test_data = pd.DataFrame({
        "cue": ["ocean"] * 15,
        "R1": (
            ["water"] * 6
            + ["blue"] * 4
            + ["waves"] * 3
            + ["fish"] * 2
        )
    })

    rankings = {
        "ocean": [
            "water", "blue", "waves", "fish"
        ]
    }

    result = evaluate_ranked_retrieval(
        rankings,
        test_data,
        "R1",
        k_values=(1, 5, 10)
    )

    row = result.iloc[0]

    assert row["top1_modal_agreement"] == 1.0
    assert row["modal_tie"] == False

    assert np.isclose(row["recall_at_5"], 1.0)
    assert np.isclose(row["weighted_recall_at_5"], 1.0)
    assert np.isclose(row["ndcg_at_10"], 1.0)


def test_modal_ties_are_handled_fairly():
    test_data = pd.DataFrame({
        "cue": ["ocean"] * 5,
        "R1": [
            "water", "water",
            "blue", "blue",
            "waves"
        ]
    })

    rankings = {
        "ocean": ["blue", "water", "waves"]
    }

    result = evaluate_ranked_retrieval(
        rankings,
        test_data,
        "R1",
        k_values=(1,)
    )

    row = result.iloc[0]

    # Blue is tied with water for the most frequent response.
    assert row["modal_tie"] == True
    assert row["n_modal_responses"] == 2
    assert row["top1_modal_agreement"] == 1.0


def test_relevance_and_modal_agreement_are_different():
    test_data = pd.DataFrame({
        "cue": ["ocean"] * 15,
        "R1": (
            ["water"] * 6
            + ["blue"] * 4
            + ["waves"] * 3
            + ["fish"] * 2
        )
    })

    rankings = {
        "ocean": ["fish", "waves", "blue", "water"]
    }

    result = evaluate_ranked_retrieval(
        rankings,
        test_data,
        "R1",
        k_values=(1,)
    )

    row = result.iloc[0]

    # Fish is a valid response, but not the modal response.
    assert row["precision_at_1"] == 1.0
    assert row["top1_modal_agreement"] == 0.0
