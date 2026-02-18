"""
metrics.py — Multi-Objective Evaluation Metrics
=================================================
Team member: Ishan Biswas (implemented), Nishant Suresh (owner)
Key functions: ndcg_at_k, mrr, catalog_coverage, compute_all_metrics

PURPOSE:
    Implements the three core metrics for the multi-objective evaluation:

    1. nDCG@12   — Normalized Discounted Cumulative Gain at rank 12
                   (matches the H&M Kaggle competition metric)
    2. MRR       — Mean Reciprocal Rank across all test users
    3. Catalog Coverage — Fraction of the total catalog that appears in
                          at least one user's top-12 recommendations

    These metrics are computed for both the Villain and the Hero, and
    the results feed into the Pareto trade-off study in `analytics/pareto/`.

USAGE:
    from src.utils.metrics import compute_all_metrics
    results = compute_all_metrics(predictions, ground_truth, catalog_size, k=12)
"""

import numpy as np


def ndcg_at_k(predicted, actual, k=12):
    """
    Compute nDCG@K for a single user.

    Args:
        predicted (list): Ranked list of predicted item IDs (length >= k).
        actual (int or list): Ground-truth item ID(s). If a single int,
                              treated as a 1-element list.
        k (int): Cutoff rank.

    Returns:
        float: nDCG@K score in [0, 1].
    """
    if isinstance(actual, (int, np.integer)):
        actual = [actual]
    actual_set = set(actual)

    # DCG: sum of 1/log2(rank+1) for each hit in top-K
    dcg = 0.0
    for i, item in enumerate(predicted[:k]):
        if item in actual_set:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because rank is 1-indexed

    # Ideal DCG: best possible with len(actual) relevant items
    ideal_hits = min(len(actual), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def mrr(predicted, actual):
    """
    Compute Reciprocal Rank for a single user.

    Args:
        predicted (list): Ranked list of predicted item IDs.
        actual (int or list): Ground-truth item ID(s).

    Returns:
        float: Reciprocal rank (0 if no hit in the list).
    """
    if isinstance(actual, (int, np.integer)):
        actual = [actual]
    actual_set = set(actual)

    for i, item in enumerate(predicted):
        if item in actual_set:
            return 1.0 / (i + 1)
    return 0.0


def catalog_coverage(all_predictions, catalog_size, k=12):
    """
    Compute catalog coverage across all users.

    Args:
        all_predictions (list[list]): Top-K predictions per user.
        catalog_size (int):           Total number of unique items (excl. PAD).
        k (int):                      Cutoff rank.

    Returns:
        float: Coverage ratio in [0, 1].
    """
    seen = set()
    for preds in all_predictions:
        seen.update(preds[:k])
    # Remove PAD (0) if it somehow appears
    seen.discard(0)
    return len(seen) / catalog_size if catalog_size > 0 else 0.0


def compute_all_metrics(predictions, ground_truth, catalog_size, k=12):
    """
    Compute all three metrics across a set of users and return as a dict.

    Args:
        predictions (list[list]): Top-K predictions per user.
        ground_truth (list):      Ground-truth item (one per user, int or list).
        catalog_size (int):       Total number of unique items (excl. PAD).
        k (int):                  Cutoff rank.

    Returns:
        dict: {"ndcg@12": float, "mrr": float, "catalog_coverage": float}
    """
    ndcg_scores = [
        ndcg_at_k(pred, gt, k) for pred, gt in zip(predictions, ground_truth)
    ]
    mrr_scores = [
        mrr(pred, gt) for pred, gt in zip(predictions, ground_truth)
    ]
    coverage = catalog_coverage(predictions, catalog_size, k)

    return {
        f"ndcg@{k}": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
        "mrr": float(np.mean(mrr_scores)) if mrr_scores else 0.0,
        "catalog_coverage": coverage,
    }
