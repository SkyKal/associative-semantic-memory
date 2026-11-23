
import pandas as pd


def evaluate_ranked_retrieval(
    ranked_by_cue,
    test_data,
    response_col,
    k_values=(1, 5, 10)
):
    """
    Evaluate ranked predictions against held-out human responses.

    ranked_by_cue:
        Dictionary mapping each cue to its ranked response words.

    test_data:
        Test observations containing cue and response_col.

    response_col:
        One of R1, R2 or R3.

    k_values:
        Retrieval cutoffs at which to calculate precision and recall.

    Returns:
        DataFrame containing metrics for each cue.
    """

    if response_col not in test_data.columns:
        raise ValueError(f"Column {response_col} not found.")

    if any(k <= 0 for k in k_values):
        raise ValueError("All K values must be positive.")

    # Ignore unavailable responses.
    valid_data = test_data[
        test_data[response_col] != ""
    ]

    results = []

    # Evaluate each cue independently.
    for cue, group in valid_data.groupby("cue"):

        # Count how many times each response occurred.
        target_counts = group[response_col].value_counts()

        # Distinct responses considered relevant.
        relevant = set(target_counts.index)

        # Retrieve the model's complete ranking.
        ranked = list(ranked_by_cue.get(cue, []))

        # Ensure candidates are unique while preserving rank.
        ranked = list(dict.fromkeys(ranked))

        candidate_set = set(ranked)

        total_responses = int(target_counts.sum())
        total_unique = len(relevant)

        row = {
            "cue": cue,
            "n_test_responses": total_responses,
            "n_unique_test_responses": total_unique,
            "n_candidates": len(ranked)
        }

        # Calculate candidate coverage.
        available = relevant.intersection(candidate_set)

        row["candidate_coverage"] = (
            len(available) / total_unique
        )

        covered_frequency = sum(
            int(target_counts.get(response, 0))
            for response in available
        )

        row["weighted_candidate_coverage"] = (
            covered_frequency / total_responses
        )

        # Calculate MRR over the complete ranking.
        first_relevant_rank = next(
            (
                rank
                for rank, response in enumerate(ranked, start=1)
                if response in relevant
            ),
            None
        )

        row["mrr"] = (
            1 / first_relevant_rank
            if first_relevant_rank is not None
            else 0.0
        )

        # Calculate metrics at each K.
        for k in k_values:

            top_k = ranked[:k]
            retrieved = set(top_k)

            hits = relevant.intersection(retrieved)

            precision = len(hits) / k
            recall = len(hits) / total_unique

            retrieved_frequency = sum(
                int(target_counts.get(response, 0))
                for response in retrieved
            )

            weighted_recall = (
                retrieved_frequency / total_responses
            )

            row[f"precision_at_{k}"] = precision
            row[f"recall_at_{k}"] = recall
            row[f"weighted_recall_at_{k}"] = weighted_recall

        results.append(row)

    return pd.DataFrame(results)


def summarise_evaluation(per_cue_results):
    """
    Calculate macro-averaged evaluation metrics
    across cues for each response position.
    """

    metric_columns = [
        column
        for column in per_cue_results.columns
        if (
            column.startswith("precision_at_")
            or column.startswith("recall_at_")
            or column.startswith("weighted_recall_at_")
        )
    ]

    metric_columns += [
        "mrr",
        "candidate_coverage",
        "weighted_candidate_coverage"
    ]

    # Average each metric equally across cues.
    means = (
        per_cue_results
        .groupby("response_position")[metric_columns]
        .mean()
        .reset_index()
    )

    # Report the amount of test data supporting each result.
    support = (
        per_cue_results
        .groupby("response_position")
        .agg(
            n_cues=("cue", "nunique"),
            valid_test_responses=("n_test_responses", "sum")
        )
        .reset_index()
    )

    summary = support.merge(
        means,
        on="response_position"
    )

    order = ["R1", "R2", "R3"]

    summary["response_position"] = pd.Categorical(
        summary["response_position"],
        categories=order,
        ordered=True
    )

    return summary.sort_values("response_position")
