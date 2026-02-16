"""
metrics.py — Multi-Objective Evaluation Metrics
=================================================
Team: Ishan, Elizabeth, Nishant

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
        predicted (list): Ranked list of predicted item IDs.
        actual (list):    Ground-truth item IDs.
        k (int):          Cutoff rank.

    Returns:
        float: nDCG@K score in [0, 1].
    """
    raise NotImplementedError("TODO: Implement nDCG@K")


def mrr(predicted, actual):
    """
    Compute Mean Reciprocal Rank for a single user.

    Args:
        predicted (list): Ranked list of predicted item IDs.
        actual (list):    Ground-truth item IDs.

    Returns:
        float: Reciprocal rank (0 if no hit).
    """
    raise NotImplementedError("TODO: Implement MRR")


def catalog_coverage(all_predictions, catalog_size, k=12):
    """
    Compute catalog coverage across all users.

    Args:
        all_predictions (list[list]): Top-K predictions per user.
        catalog_size (int):           Total number of unique items.
        k (int):                      Cutoff rank.

    Returns:
        float: Coverage ratio in [0, 1].
    """
    raise NotImplementedError("TODO: Implement catalog coverage")


def compute_all_metrics(predictions, ground_truth, catalog_size, k=12):
    """
    Compute all three metrics and return as a dict.

    Returns:
        dict: {"ndcg@12": float, "mrr": float, "catalog_coverage": float}
    """
    raise NotImplementedError("TODO: Aggregate all metrics")
