"""
metrics.py — Multi-Objective Evaluation Metrics
=================================================
Team: Elizabeth Coquillette (primary), Ishan Biswas (integration)

PURPOSE:
    Implements the core metrics for the multi-objective evaluation:

    1. nDCG@12   — Normalized Discounted Cumulative Gain at rank 12
                   (matches the H&M Kaggle competition metric)
    2. MRR       — Mean Reciprocal Rank across all test users
    3. Catalog Coverage — Fraction of the total catalog that appears in
                          at least one user's top-12 recommendations

    Extended metrics for long-tail analysis:
    4. popularity_logit_scores — smoothed popularity logits per item
    5. mean_tail_score_at_k — average tail score in recommendations
    6. tail_item_rate_at_k — fraction of recs that are tail items

    These metrics are computed for both the Villain and the Hero, and
    the results feed into the Pareto trade-off study in `analytics/pareto/`.

USAGE:
    from src.utils.metrics import compute_all_metrics
    results = compute_all_metrics(predictions, ground_truth, catalog_size, k=12)
"""

import numpy as np


def _to_relevant_set(actual):
    """Convert ground truth to a set, handling int scalars gracefully."""
    if actual is None:
        return set()
    if isinstance(actual, (int, np.integer)):
        return {actual}
    if isinstance(actual, (set, frozenset)):
        return set(actual)
    return set(actual)


def ndcg_at_k(predicted, actual, k=12):
    """
    Compute nDCG@K for a single user.

    Args:
        predicted (list): Ranked list of predicted item IDs.
        actual (int or list): Ground-truth item ID(s). If a single int,
                              treated as a 1-element set.
        k (int):          Cutoff rank.

    Returns:
        float: nDCG@K score in [0, 1].
    """
    if k <= 0:
        return 0.0
    relevant = _to_relevant_set(actual)
    if not relevant:
        return 0.0

    ranked = list(predicted)[:k]
    if not ranked:
        return 0.0

    dcg = 0.0
    for rank, item_id in enumerate(ranked, start=1):
        if item_id in relevant:
            dcg += 1.0 / np.log2(rank + 1.0)

    ideal_hits = min(len(relevant), k)
    idcg = float(np.sum(1.0 / np.log2(np.arange(2, ideal_hits + 2)))) if ideal_hits > 0 else 0.0
    if idcg == 0.0:
        return 0.0
    return float(dcg / idcg)


def mrr(predicted, actual):
    """
    Compute Reciprocal Rank for a single user.

    Args:
        predicted (list): Ranked list of predicted item IDs.
        actual (int or list): Ground-truth item ID(s).

    Returns:
        float: Reciprocal rank (0 if no hit in the list).
    """
    relevant = _to_relevant_set(actual)
    if not relevant:
        return 0.0

    for rank, item_id in enumerate(predicted, start=1):
        if item_id in relevant:
            return float(1.0 / rank)
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
    if catalog_size <= 0:
        return 0.0
    if k <= 0:
        return 0.0

    surfaced = set()
    for pred in all_predictions:
        surfaced.update(list(pred)[:k])
    # Remove PAD (0) if it somehow appears
    surfaced.discard(0)
    return float(len(surfaced) / catalog_size)


def popularity_logit_scores(item_sales_counts, alpha=0.5):
    """
    Compute smoothed popularity logits for each item:
        p_i = (n_i + alpha) / (N + alpha * M)
        logit_i = log(p_i / (1 - p_i))

    Args:
        item_sales_counts (dict): item_id -> sales count
        alpha (float): additive smoothing constant (>0)

    Returns:
        dict: item_id -> popularity_logit
    """
    if alpha <= 0:
        raise ValueError("alpha must be > 0")
    if not item_sales_counts:
        return {}

    counts = {item_id: float(max(0, c)) for item_id, c in item_sales_counts.items()}
    n_total = float(sum(counts.values()))
    m_items = float(len(counts))
    denom = n_total + alpha * m_items
    if denom <= 0:
        return {item_id: 0.0 for item_id in counts}

    out = {}
    for item_id, n_i in counts.items():
        p_i = (n_i + alpha) / denom
        p_i = float(np.clip(p_i, 1e-12, 1.0 - 1e-12))
        out[item_id] = float(np.log(p_i / (1.0 - p_i)))
    return out


def mean_tail_score_at_k(all_predictions, pop_logit_by_item, k=12):
    """
    Mean tail score over all recommended slots in top-K.
    tail_score = -popularity_logit
    """
    if k <= 0:
        return 0.0
    scores = []
    for pred in all_predictions:
        for item_id in list(pred)[:k]:
            scores.append(-float(pop_logit_by_item.get(item_id, 0.0)))
    if not scores:
        return 0.0
    return float(np.mean(scores))


def tail_item_rate_at_k(all_predictions, item_sales_counts, tail_threshold, k=12):
    """
    Fraction of recommended top-K slots where item sales < tail_threshold.
    """
    if k <= 0:
        return 0.0
    if tail_threshold <= 0:
        return 0.0
    total_slots = 0
    tail_slots = 0
    for pred in all_predictions:
        topk = list(pred)[:k]
        total_slots += len(topk)
        for item_id in topk:
            if item_sales_counts.get(item_id, 0) < tail_threshold:
                tail_slots += 1
    if total_slots == 0:
        return 0.0
    return float(tail_slots / total_slots)


def compute_all_metrics(predictions, ground_truth, catalog_size, k=12):
    """
    Compute all three core metrics across a set of users and return as a dict.

    Args:
        predictions (list[list]): Top-K predictions per user.
        ground_truth (list):      Ground-truth item(s) per user (int or list).
        catalog_size (int):       Total number of unique items (excl. PAD).
        k (int):                  Cutoff rank.

    Returns:
        dict: {"ndcg@k": float, "mrr": float, "catalog_coverage": float}
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("predictions and ground_truth must have the same length")

    ndcg_scores = [ndcg_at_k(p, a, k=k) for p, a in zip(predictions, ground_truth)]
    mrr_scores = [mrr(p, a) for p, a in zip(predictions, ground_truth)]

    return {
        f"ndcg@{k}": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
        "mrr": float(np.mean(mrr_scores)) if mrr_scores else 0.0,
        "catalog_coverage": catalog_coverage(predictions, catalog_size, k=k),
    }


def compute_multi_objective_metrics(
    predictions,
    ground_truth,
    catalog_size,
    item_sales_counts,
    k=12,
    alpha=0.5,
    tail_threshold=50,
):
    """
    Extended metric bundle for relevance + long-tail exposure.
    """
    results = compute_all_metrics(predictions, ground_truth, catalog_size, k=k)
    pop_logits = popularity_logit_scores(item_sales_counts, alpha=alpha)
    results["mean_tail_score@k"] = mean_tail_score_at_k(predictions, pop_logits, k=k)
    results["tail_item_rate@k"] = tail_item_rate_at_k(
        predictions, item_sales_counts, tail_threshold=tail_threshold, k=k
    )
    results["alpha"] = float(alpha)
    results["tail_threshold"] = int(tail_threshold)
    return results
