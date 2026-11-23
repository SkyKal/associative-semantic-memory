
import math

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
        Retrieval cutoffs for evaluation.
    """

    if response_col not in test_data.columns:
        raise ValueError(f"Column {response_col} not found.")

    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("K values must be positive.")

    # Exclude unavailable responses.
    valid_data = test_data[
        test_data[response_col] != ""
    ]

    results = []

    # Evaluate every cue separately.
    for cue, group in valid_data.groupby("cue"):

        # Human response frequencies in the test set.
        target_counts = group[response_col].value_counts()

        relevant = set(target_counts.index)

        # Model's ranked candidate list.
        ranked = list(ranked_by_cue.get(cue, []))

        # Remove duplicate predictions while preserving order.
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

        # --------------------------------------------------
        # 1. Candidate coverage
        # --------------------------------------------------

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

        # --------------------------------------------------
        # 2. Top-1 modal-response agreement
        # --------------------------------------------------

        maximum_frequency = target_counts.max()

        # There can be multiple equally frequent responses.
        modal_responses = set(
            target_counts[
                target_counts == maximum_frequency
            ].index
        )

        row["n_modal_responses"] = len(modal_responses)
        row["modal_tie"] = len(modal_responses) > 1

        top_prediction = ranked[0] if ranked else None

        row["top1_modal_agreement"] = float(
            top_prediction in modal_responses
        )

        # --------------------------------------------------
        # 3. Mean Reciprocal Rank
        # --------------------------------------------------

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

        # --------------------------------------------------
        # 4. Precision, recall and NDCG at different K
        # --------------------------------------------------

        for k in k_values:

            top_k = ranked[:k]
            retrieved = set(top_k)

            hits = relevant.intersection(retrieved)

            # Precision@K
            precision = len(hits) / k

            # Recall@K
            recall = len(hits) / total_unique

            # Frequency-weighted Recall@K
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

            # NDCG@K using human response frequencies
            dcg = sum(
                int(target_counts.get(response, 0))
                / math.log2(rank + 1)
                for rank, response in enumerate(top_k, start=1)
            )

            # Ideal ranking: most frequent human responses first.
            ideal_frequencies = sorted(
                target_counts.tolist(),
                reverse=True
            )[:k]

            idcg = sum(
                frequency / math.log2(rank + 1)
                for rank, frequency in enumerate(
                    ideal_frequencies,
                    start=1
                )
            )

            ndcg = dcg / idcg if idcg > 0 else 0.0

            row[f"ndcg_at_{k}"] = ndcg

        results.append(row)

    return pd.DataFrame(results)


def summarise_evaluation(per_cue_results):
    """
    Calculate macro-averaged evaluation metrics across cues.

    Also reports the proportion of cues whose most frequent
    test response has a tie.
    """

    metric_prefixes = (
        "precision_at_",
        "recall_at_",
        "weighted_recall_at_",
        "ndcg_at_"
    )

    metric_columns = [
        column
        for column in per_cue_results.columns
        if column.startswith(metric_prefixes)
    ]

    metric_columns += [
        "top1_modal_agreement",
        "mrr",
        "candidate_coverage",
        "weighted_candidate_coverage"
    ]

    # Each cue receives equal weight in the macro average.
    means = (
        per_cue_results
        .groupby("response_position")[metric_columns]
        .mean()
        .reset_index()
    )

    # Report evaluation support and frequency ties.
    support = (
        per_cue_results
        .groupby("response_position")
        .agg(
            n_cues=("cue", "nunique"),
            valid_test_responses=("n_test_responses", "sum"),
            tied_modal_cues=("modal_tie", "sum"),
            modal_tie_rate=("modal_tie", "mean")
        )
        .reset_index()
    )

    summary = support.merge(
        means,
        on="response_position"
    )

    # Preserve R1, R2, R3 ordering.
    order = ["R1", "R2", "R3"]

    summary["response_position"] = pd.Categorical(
        summary["response_position"],
        categories=order,
        ordered=True
    )

    return summary.sort_values("response_position")

