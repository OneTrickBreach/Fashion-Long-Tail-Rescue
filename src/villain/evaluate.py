"""
evaluate.py — Standalone Evaluation Script for Villain Baseline
================================================================
Team member: Ishan Biswas
Key functions: evaluate_villain

PURPOSE:
    Loads a trained Villain checkpoint and runs full evaluation on the
    test set without re-training.  Reports nDCG@12, MRR, Catalog Coverage,
    and per-popularity-bucket metrics (head / torso / tail breakdown).

    Also generates a long-tail analysis showing whether the Villain is
    biased toward popular items (spoiler: it is — that's the point).

USAGE:
    python -m src.villain.evaluate                   # uses villain_best.pt
    python -m src.villain.evaluate --checkpoint checkpoints/villain_latest.pt
"""

import os
import json
import argparse
import logging
import numpy as np
import pandas as pd
import torch

from src.utils.helpers import load_config, set_seed, get_device, setup_logging
from src.data.dataset import build_dataloaders
from src.villain.config import get_villain_config
from src.villain.model import VillainModel
from src.utils.metrics import ndcg_at_k, mrr, catalog_coverage, compute_all_metrics

logger = logging.getLogger("seeing-the-unseen")


@torch.no_grad()
def evaluate_villain(config: dict, checkpoint_path: str | None = None) -> dict:
    """
    Load a trained Villain model and evaluate on the test set.

    Computes:
        - Overall nDCG@12, MRR, Catalog Coverage
        - Per-popularity-bucket breakdown (head / torso / tail)
        - Pop-bias analysis: what fraction of recommendations go to each bucket

    Args:
        config:          Parsed config.yaml.
        checkpoint_path: Path to .pt checkpoint (default: villain_best.pt).

    Returns:
        dict: Full evaluation results.
    """
    setup_logging()
    set_seed(config["project"]["seed"])
    device = get_device(config["embedding"]["device"])
    villain_cfg = get_villain_config(config)
    eval_k = config["evaluation"]["k"]
    ckpt_dir = config["paths"]["checkpoints"]

    # ── DataLoaders ──────────────────────────────────────────
    logger.info("Building dataloaders …")
    _, _, test_loader, meta = build_dataloaders(config, mode="villain")
    num_items = meta["num_items"]
    catalog_size = num_items - 1
    idx_to_id = meta["idx_to_id"]

    # ── Load model ───────────────────────────────────────────
    if checkpoint_path is None:
        checkpoint_path = os.path.join(ckpt_dir, "villain_best.pt")
    logger.info(f"Loading model from {checkpoint_path} …")

    model = VillainModel(num_items=num_items, config=villain_cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(f"  Loaded checkpoint from epoch {ckpt['epoch'] + 1}")

    # ── Load article metadata for popularity analysis ────────
    sampled_dir = config["paths"]["sampled_data"]
    articles = pd.read_csv(os.path.join(sampled_dir, "articles_sampled.csv"))
    txn = pd.read_csv(os.path.join(sampled_dir, "transactions_sampled.csv"))

    # Build popularity buckets
    article_counts = txn["article_id"].value_counts()
    id_to_idx = meta["id_to_idx"]

    # Assign head/torso/tail per article
    ranks = article_counts.rank(pct=True)
    pop_bucket = {}
    for aid, pct in ranks.items():
        if aid in id_to_idx:
            idx = id_to_idx[aid]
            if pct >= 0.90:
                pop_bucket[idx] = "head"
            elif pct >= 0.50:
                pop_bucket[idx] = "torso"
            else:
                pop_bucket[idx] = "tail"

    # ── Run test evaluation ──────────────────────────────────
    logger.info("Running test evaluation …")
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    n_batches = 0
    criterion = torch.nn.CrossEntropyLoss()

    for batch in test_loader:
        item_seq = batch["item_seq"].to(device)
        positions = batch["positions"].to(device)
        seq_len = batch["seq_len"].to(device)
        targets = batch["target"].to(device)

        logits = model(item_seq, positions, seq_len)
        loss = criterion(logits, targets)
        total_loss += loss.item()
        n_batches += 1

        logits[:, 0] = float("-inf")
        _, top_indices = logits.topk(eval_k, dim=-1)

        all_predictions.extend(top_indices.cpu().tolist())
        all_targets.extend(targets.cpu().tolist())

    # ── Overall metrics ──────────────────────────────────────
    overall = compute_all_metrics(all_predictions, all_targets, catalog_size, k=eval_k)
    overall["avg_loss"] = total_loss / max(n_batches, 1)

    logger.info("─── Overall Test Metrics ───")
    for k_name, v in overall.items():
        logger.info(f"  {k_name}: {v:.4f}")

    # ── Per-bucket breakdown ─────────────────────────────────
    bucket_results = {"head": [], "torso": [], "tail": []}
    for pred, tgt in zip(all_predictions, all_targets):
        bucket = pop_bucket.get(tgt, "unknown")
        if bucket in bucket_results:
            bucket_results[bucket].append((pred, tgt))

    logger.info("─── Per-Bucket Test Metrics ───")
    bucket_metrics = {}
    for bucket_name in ["head", "torso", "tail"]:
        items = bucket_results[bucket_name]
        if not items:
            continue
        preds = [p for p, _ in items]
        tgts = [t for _, t in items]
        bm = compute_all_metrics(preds, tgts, catalog_size, k=eval_k)
        bm["num_samples"] = len(items)
        bucket_metrics[bucket_name] = bm
        logger.info(
            f"  {bucket_name:>5}: nDCG@{eval_k}={bm[f'ndcg@{eval_k}']:.4f}, "
            f"MRR={bm['mrr']:.4f}, n={len(items)}"
        )

    # ── Pop-bias analysis: where do recommendations go? ──────
    logger.info("─── Recommendation Bias Analysis ───")
    rec_buckets = {"head": 0, "torso": 0, "tail": 0, "unknown": 0}
    total_recs = 0
    for pred_list in all_predictions:
        for item_idx in pred_list:
            bucket = pop_bucket.get(item_idx, "unknown")
            rec_buckets[bucket] += 1
            total_recs += 1

    bias_analysis = {}
    for bucket_name in ["head", "torso", "tail"]:
        count = rec_buckets[bucket_name]
        pct = count / total_recs if total_recs > 0 else 0
        bias_analysis[bucket_name] = {"count": count, "pct": round(pct, 4)}
        logger.info(f"  {bucket_name:>5}: {count:,} recs ({pct:.1%} of all)")

    # ── Pop-bias vector analysis ─────────────────────────────
    logger.info("─── Learned Pop-Bias Vector ───")
    pop_bias = model.pop_bias.detach().cpu().numpy()
    # Skip PAD at index 0
    pop_bias_items = pop_bias[1:]
    logger.info(f"  Mean bias:   {pop_bias_items.mean():.4f}")
    logger.info(f"  Std bias:    {pop_bias_items.std():.4f}")
    logger.info(f"  Min bias:    {pop_bias_items.min():.4f}")
    logger.info(f"  Max bias:    {pop_bias_items.max():.4f}")

    # Bias by bucket
    bias_by_bucket = {"head": [], "torso": [], "tail": []}
    for item_idx, bucket in pop_bucket.items():
        if item_idx > 0 and item_idx < len(pop_bias):
            bias_by_bucket[bucket].append(pop_bias[item_idx])

    for bucket_name in ["head", "torso", "tail"]:
        vals = bias_by_bucket[bucket_name]
        if vals:
            logger.info(
                f"  {bucket_name:>5} avg bias: {np.mean(vals):.4f} "
                f"(n={len(vals)})"
            )

    # ── Compile full results ─────────────────────────────────
    results = {
        "overall": overall,
        "per_bucket": bucket_metrics,
        "recommendation_bias": bias_analysis,
        "pop_bias_stats": {
            "mean": float(pop_bias_items.mean()),
            "std": float(pop_bias_items.std()),
            "min": float(pop_bias_items.min()),
            "max": float(pop_bias_items.max()),
        },
        "checkpoint": checkpoint_path,
        "checkpoint_epoch": ckpt["epoch"] + 1,
    }

    # ── Save ─────────────────────────────────────────────────
    output_dir = config["paths"]["outputs"]
    os.makedirs(output_dir, exist_ok=True)
    eval_path = os.path.join(output_dir, "villain_eval_full.json")
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Full evaluation saved → {eval_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Villain baseline")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint file (default: checkpoints/villain_best.pt)",
    )
    args = parser.parse_args()
    cfg = load_config()
    evaluate_villain(cfg, checkpoint_path=args.checkpoint)
