"""
evaluate.py — Standalone Evaluation Script for Hero Model
=========================================================
Team member: Ishan Biswas
Key functions: evaluate_hero

PURPOSE:
    Loads a trained Hero checkpoint and runs full evaluation on the
    test set to report nDCG@12, MRR, Catalog Coverage, and 
    per-popularity-bucket metrics (head / torso / tail breakdown).
    
    This helps measure the "Tail-item recommendation rate."
"""

import os
import json
import argparse
import logging
import numpy as np
import pandas as pd
import torch

from src.utils.helpers import load_config, set_seed, get_device, setup_logging
from src.data.dataset import build_dataloaders, build_id_maps, load_multimodal_embeddings
from src.hero.model import HeroModel
from src.utils.metrics import compute_all_metrics

logger = logging.getLogger("seeing-the-unseen")


@torch.no_grad()
def evaluate_hero(config: dict, checkpoint_path: str | None = None) -> dict:
    setup_logging()
    set_seed(config["project"]["seed"])
    device = get_device(config["embedding"]["device"])
    eval_k = config["evaluation"]["k"]
    ckpt_dir = config["paths"]["checkpoints"]

    # ── DataLoaders ──────────────────────────────────────────
    logger.info("Building dataloaders …")
    _, _, test_loader, meta = build_dataloaders(config, mode="hero")
    num_items = meta["num_items"]
    catalog_size = num_items - 1
    idx_to_id = meta["idx_to_id"]
    id_to_idx = meta["id_to_idx"]

    # Load visual embeddings
    emb_path = os.path.join(config["paths"]["embeddings"], "multimodal_embeddings.pt")
    visual_embeddings = load_multimodal_embeddings(emb_path, id_to_idx, num_items).to(device)

    # ── Load model ───────────────────────────────────────────
    if checkpoint_path is None:
        checkpoint_path = os.path.join(ckpt_dir, "hero_best.pt")
    logger.info(f"Loading model from {checkpoint_path} …")

    model = HeroModel(num_items=num_items, config=config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(f"  Loaded checkpoint from epoch {ckpt['epoch'] + 1}")

    # ── Load article metadata for popularity analysis ────────
    sampled_dir = config["paths"]["sampled_data"]
    txn = pd.read_csv(os.path.join(sampled_dir, "transactions_sampled.csv"))

    # Build popularity buckets based on training transactions
    article_counts = txn["article_id"].value_counts()

    # Assign head/torso/tail per article
    ranks = article_counts.rank(pct=True)
    pop_bucket = {}
    for aid, pct in ranks.items():
        if aid in id_to_idx:
            idx = id_to_idx[aid]
            # Tail definition: bottom 50%
            if pct >= 0.90:
                pop_bucket[idx] = "head"
            elif pct >= 0.50:
                pop_bucket[idx] = "torso"
            else:
                pop_bucket[idx] = "tail"
                
    # In config.yaml, long_tail_threshold is 10. Let's strictly use < 10 interactions.
    threshold = config["sampling"].get("long_tail_threshold", 10)
    strict_tail_buckets = {}
    for aid, count in article_counts.items():
        if aid in id_to_idx:
            idx = id_to_idx[aid]
            if count < threshold:
                strict_tail_buckets[idx] = "tail"
            elif count < 50:
                strict_tail_buckets[idx] = "torso"
            else:
                strict_tail_buckets[idx] = "head"
    pop_bucket = strict_tail_buckets # Override with strict counts

    # ── Run test evaluation ──────────────────────────────────
    logger.info("Running test evaluation …")
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    n_batches = 0
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    # Precompute visual catalog injection for fair comparison
    hero_catalog_vis = model.visual_proj(visual_embeddings) 
    hero_full_catalog = model.item_emb.weight + hero_catalog_vis

    for batch in test_loader:
        item_seq = batch["item_seq"].to(device)
        positions = batch["positions"].to(device)
        targets = batch["target"].to(device)
        vis_seq = batch.get("visual_embeds")
        if vis_seq is not None:
            vis_seq = vis_seq.to(device)

        _, h_states = model(item_seq, positions, vis_seq)
        logits = torch.matmul(h_states, hero_full_catalog.transpose(0, 1))
        
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

    # ── Compile full results ─────────────────────────────────
    results = {
        "overall": overall,
        "per_bucket": bucket_metrics,
        "recommendation_bias": bias_analysis,
        "checkpoint": checkpoint_path,
        "checkpoint_epoch": ckpt["epoch"] + 1,
    }

    # ── Save ─────────────────────────────────────────────────
    output_dir = config["paths"]["outputs"]
    os.makedirs(output_dir, exist_ok=True)
    eval_path = os.path.join(output_dir, "hero_eval_full.json")
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Full evaluation saved → {eval_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Hero baseline")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint file (default: checkpoints/hero_best.pt)",
    )
    args = parser.parse_args()
    cfg = load_config()
    evaluate_hero(cfg, checkpoint_path=args.checkpoint)
