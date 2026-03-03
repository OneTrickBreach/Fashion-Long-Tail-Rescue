"""
ablation.py — Visual Ablation Study (Elizabeth's Phase 3 portion)
=================================================================
Team member: Ishan Biswas (covering for Elizabeth)
Key functions: run_ablation

PURPOSE:
    Trains the Hero with use_visual=False (ID-only BST) from scratch,
    evaluates on the test set, runs cold-start simulation, and produces
    a comparison table: Villain vs Hero (ID-only) vs Hero (visual).

    Saves results to outputs/hero_ablation_no_visual.json.
    Checkpoint saved to checkpoints/hero_no_visual_best.pt.
"""

import os
import copy
import json
import time
import random
import logging
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from src.utils.helpers import load_config, set_seed, get_device, setup_logging
from src.data.dataset import build_dataloaders, build_id_maps
from src.hero.model import HeroModel
from src.hero.contrastive import MultiObjectiveLoss, hard_negative_mining
from src.utils.metrics import (
    compute_all_metrics,
    compute_multi_objective_metrics,
    popularity_logit_scores,
)

logger = logging.getLogger("seeing-the-unseen")

CKPT_TAG = "no_visual"  # checkpoint naming tag


# ──────────────────────────────────────────────────────────────
# Training (ID-only Hero, from scratch)
# ──────────────────────────────────────────────────────────────

def _train_id_only_hero(
    config: dict,
    train_loader,
    val_loader,
    meta: dict,
    pop_logits_tensor: torch.Tensor,
    device: torch.device,
) -> tuple[HeroModel, float, list[dict]]:
    """
    Train the Hero model with use_visual=False from scratch.
    Returns (model, best_val_ndcg, history).
    """
    hero_cfg = config["hero"]
    eval_k = config["evaluation"]["k"]
    ckpt_dir = config["paths"]["checkpoints"]
    os.makedirs(ckpt_dir, exist_ok=True)
    num_items = meta["num_items"]
    catalog_size = num_items - 1

    # ── Build config with use_visual=False ────────────────────
    ablation_config = copy.deepcopy(config)
    ablation_config["hero"]["use_visual"] = False
    # discovery_weight stays 0.0 for a fair comparison with Phase 2 Hero
    ablation_config["hero"]["discovery_weight"] = 0.0

    # ── Model ─────────────────────────────────────────────────
    model = HeroModel(config=ablation_config, num_items=num_items).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  ID-only HeroModel: {total_params:,} parameters (no VisualProjection)")

    # ── Optimizer & Scheduler ─────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hero_cfg.get("lr", 0.0005),
        weight_decay=hero_cfg.get("weight_decay", 0.01),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3,
    )

    # ── Loss (CE + contrastive, no discovery) ─────────────────
    criterion = MultiObjectiveLoss(ablation_config).to(device)
    num_negatives = hero_cfg.get("contrastive", {}).get("hard_negatives", 10)

    # ── Training config ───────────────────────────────────────
    epochs = hero_cfg.get("epochs", 50)
    checkpoint_every = hero_cfg.get("checkpoint_every", 5)
    patience = hero_cfg.get("patience", 10)

    best_path = os.path.join(ckpt_dir, f"hero_{CKPT_TAG}_best.pt")
    latest_path = os.path.join(ckpt_dir, f"hero_{CKPT_TAG}_latest.pt")

    # ── Resume from existing ID-only checkpoint if present ────
    start_epoch = 0
    best_ndcg = 0.0
    history: list[dict] = []
    epochs_without_improvement = 0

    if os.path.exists(latest_path):
        logger.info(f"Found existing ID-only checkpoint: {latest_path}")
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_ndcg = ckpt["best_ndcg"]
        history = ckpt.get("history", [])
        if history:
            best_so_far = 0.0
            epochs_without_improvement = 0
            ndcg_key = f"val_ndcg@{eval_k}"
            for h in history:
                if h.get(ndcg_key, 0) > best_so_far:
                    best_so_far = h[ndcg_key]
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
        logger.info(f"  Resumed from epoch {ckpt['epoch']+1}, best nDCG={best_ndcg:.4f}")
    else:
        logger.info("No ID-only checkpoint found — training from scratch.")

    if start_epoch >= epochs:
        logger.info(f"Training already complete ({start_epoch}/{epochs} epochs).")
        return model, best_ndcg, history

    # ── Training loop ─────────────────────────────────────────
    logger.info(
        f"Training ID-only Hero: epochs {start_epoch+1}..{epochs}, "
        f"patience={patience}"
    )
    logger.info(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val nDCG':>8} | "
                f"{'Val MRR':>8} | {'Val Cov':>8} | {'Time':>6}")
    logger.info("─" * 60)

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        hard_negatives = hard_negative_mining(
            model.item_emb.weight, attributes=None, num_negatives=num_negatives
        ).to(device)

        # ── Train one epoch ───────────────────────────────────
        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            item_seq = batch["item_seq"].to(device)
            positions = batch["positions"].to(device)
            targets = batch["target"].to(device)
            # No visual embeddings for ID-only model

            optimizer.zero_grad()
            logits, contrastive_embeds = model(item_seq, positions, None)

            positive_embeds = model.item_emb(targets)
            batch_neg_indices = hard_negatives[targets]
            negative_embeds = model.item_emb(batch_neg_indices)

            loss, loss_ce, loss_cl, loss_disc = criterion(
                logits, targets,
                anchor=contrastive_embeds,
                positive=positive_embeds,
                negatives=negative_embeds,
                pop_logits=pop_logits_tensor,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        avg_train_loss = running_loss / max(n_batches, 1)

        # ── Validate ──────────────────────────────────────────
        model.eval()
        val_preds, val_tgts = [], []
        with torch.no_grad():
            for batch in val_loader:
                item_seq = batch["item_seq"].to(device)
                positions = batch["positions"].to(device)
                targets = batch["target"].to(device)
                logits, _ = model(item_seq, positions, None)
                logits[:, 0] = float("-inf")
                _, top_k = logits.topk(eval_k, dim=-1)
                val_preds.extend(top_k.cpu().tolist())
                val_tgts.extend(targets.cpu().tolist())

        val_metrics = compute_all_metrics(val_preds, val_tgts, num_items - 1, k=eval_k)
        val_ndcg = val_metrics[f"ndcg@{eval_k}"]
        val_mrr = val_metrics["mrr"]
        val_cov = val_metrics["catalog_coverage"]

        scheduler.step(val_ndcg)
        elapsed = time.time() - t0

        logger.info(
            f"{epoch+1:>5} | {avg_train_loss:>10.4f} | {val_ndcg:>8.4f} | "
            f"{val_mrr:>8.4f} | {val_cov:>8.4f} | {elapsed:>5.1f}s"
        )

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            f"val_ndcg@{eval_k}": val_ndcg,
            "val_mrr": val_mrr,
            "val_catalog_coverage": val_cov,
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_s": round(elapsed, 1),
        }
        history.append(epoch_record)

        # ── Best model? ───────────────────────────────────────
        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            epochs_without_improvement = 0
            _save_ckpt(best_path, model, optimizer, scheduler, epoch, best_ndcg, history)
        else:
            epochs_without_improvement += 1

        # ── Periodic checkpoint ───────────────────────────────
        if (epoch + 1) % checkpoint_every == 0 or epoch == epochs - 1:
            _save_ckpt(latest_path, model, optimizer, scheduler, epoch, best_ndcg, history)

        # ── Early stopping ────────────────────────────────────
        if epochs_without_improvement >= patience:
            logger.info(
                f"  Early stopping at epoch {epoch+1} "
                f"(no improvement for {patience} epochs)"
            )
            _save_ckpt(latest_path, model, optimizer, scheduler, epoch, best_ndcg, history)
            break

    # ── Restore best ──────────────────────────────────────────
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"Training complete. Best val nDCG@{eval_k}: {best_ndcg:.4f}")

    return model, best_ndcg, history


def _save_ckpt(path, model, optimizer, scheduler, epoch, best_ndcg, history):
    state = {
        "epoch": epoch,
        "best_ndcg": best_ndcg,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "history": history,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    logger.info(f"  Checkpoint saved → {path}")


# ──────────────────────────────────────────────────────────────
# Evaluation (ID-only — no visual catalog injection)
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def _evaluate_id_only(
    model: HeroModel,
    test_loader,
    device: torch.device,
    meta: dict,
    config: dict,
) -> dict:
    """
    Evaluate the ID-only Hero on the test set.
    Unlike evaluate.py, does NOT inject visual embeddings into the catalog.
    """
    model.eval()
    eval_k = config["evaluation"]["k"]
    num_items = meta["num_items"]
    catalog_size = num_items - 1
    idx_to_id = meta["idx_to_id"]
    item_sales_counts = meta["item_sales_counts"]
    tail_threshold = config["sampling"].get("long_tail_threshold", 10)

    all_predictions = []
    all_targets = []
    total_loss = 0.0
    n_batches = 0
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for batch in test_loader:
        item_seq = batch["item_seq"].to(device)
        positions = batch["positions"].to(device)
        targets = batch["target"].to(device)

        logits, _ = model(item_seq, positions, None)

        loss = criterion(logits, targets)
        total_loss += loss.item()
        n_batches += 1

        logits[:, 0] = float("-inf")
        _, top_indices = logits.topk(eval_k, dim=-1)

        all_predictions.extend(top_indices.cpu().tolist())
        all_targets.extend(targets.cpu().tolist())

    # Map indices → raw article IDs for tail metrics
    all_preds_raw = [
        [idx_to_id.get(idx, -1) for idx in pred_list]
        for pred_list in all_predictions
    ]
    all_targets_raw = [idx_to_id.get(t, -1) for t in all_targets]

    metrics = compute_multi_objective_metrics(
        all_preds_raw, all_targets_raw, catalog_size,
        item_sales_counts, k=eval_k, tail_threshold=tail_threshold,
    )
    metrics["avg_loss"] = total_loss / max(n_batches, 1)

    # Per-bucket breakdown (derived from meta, no redundant CSV read)
    id_to_idx = meta["id_to_idx"]
    threshold = config["sampling"].get("long_tail_threshold", 10)
    torso_upper = threshold * 5  # head boundary (consistent with evaluate.py)
    pop_bucket = {}
    for aid, count in item_sales_counts.items():
        if aid in id_to_idx:
            idx = id_to_idx[aid]
            if count < threshold:
                pop_bucket[idx] = "tail"
            elif count < torso_upper:
                pop_bucket[idx] = "torso"
            else:
                pop_bucket[idx] = "head"

    bucket_results = {"head": [], "torso": [], "tail": []}
    for pred, tgt in zip(all_predictions, all_targets):
        bucket = pop_bucket.get(tgt, "unknown")
        if bucket in bucket_results:
            bucket_results[bucket].append((pred, tgt))

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

    # Recommendation bias
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

    metrics["per_bucket"] = bucket_metrics
    metrics["recommendation_bias"] = bias_analysis
    return metrics


# ──────────────────────────────────────────────────────────────
# Cold-start simulation (ID-only)
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def _cold_start_id_only(
    model: HeroModel,
    config: dict,
    device: torch.device,
) -> dict:
    """
    Simplified cold-start simulation for the ID-only Hero.
    Mirrors evaluate_cold_start.py but without visual catalog injection.
    """
    paths = config["paths"]
    art_path = os.path.join(paths["sampled_data"], "articles_sampled.csv")
    txn_path = os.path.join(paths["sampled_data"], "transactions_sampled.csv")

    txn = pd.read_csv(txn_path, parse_dates=["t_dat"]).sort_values(["customer_id", "t_dat"])
    user_sequences = {uid: grp["article_id"].tolist() for uid, grp in txn.groupby("customer_id")}

    id_to_idx, idx_to_id = build_id_maps(art_path)
    num_items = max(id_to_idx.values()) + 1

    # Identify training items
    train_items = set()
    for seq in user_sequences.values():
        if len(seq) >= 3:
            for item in seq[:-2]:
                if item in id_to_idx:
                    train_items.add(id_to_idx[item])

    # Find cold-start test samples
    test_samples = []
    for uid, raw_items in user_sequences.items():
        if len(raw_items) < 3:
            continue
        tgt = raw_items[-1]
        if tgt in id_to_idx:
            test_samples.append((uid, raw_items[:-1], id_to_idx[tgt]))

    cold_start_samples = [s for s in test_samples if s[2] not in train_items]
    logger.info(f"  Cold-start samples: {len(cold_start_samples)}")

    random.seed(42)
    eval_samples = random.sample(cold_start_samples, min(100, len(cold_start_samples)))

    model.eval()
    ranks = []
    max_seq_len = config["hero"]["max_seq_len"]

    for uid, seq, target_idx in eval_samples:
        idx_seq = [id_to_idx[a] for a in seq if a in id_to_idx][-max_seq_len:]
        slen = len(idx_seq)
        padded = idx_seq + [0] * (max_seq_len - slen)

        item_seq = torch.tensor([padded], dtype=torch.long).to(device)
        positions = torch.arange(max_seq_len, dtype=torch.long).unsqueeze(0).to(device)

        logits, _ = model(item_seq, positions, None)
        logits[0, 0] = float("-inf")

        sorted_indices = torch.argsort(logits[0], descending=True)
        rank = (sorted_indices == target_idx).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)

    ranks_arr = np.array(ranks)
    return {
        "num_eval_samples": len(eval_samples),
        "avg_rank": float(np.mean(ranks_arr)),
        "hit_at_12": float(np.mean(ranks_arr <= 12)),
        "hit_at_50": float(np.mean(ranks_arr <= 50)),
    }


# ──────────────────────────────────────────────────────────────
# Main ablation driver
# ──────────────────────────────────────────────────────────────

def run_ablation(config: dict) -> dict:
    """
    Full ablation study: train ID-only Hero, evaluate, cold-start,
    and produce comparison table vs Villain and visual Hero.
    """
    setup_logging()
    set_seed(config["project"]["seed"])
    device = get_device(config["embedding"]["device"])

    output_dir = config["paths"]["outputs"]
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  Ablation Study — Hero (ID-only) vs Hero (visual)")
    logger.info("=" * 60)

    # ── Build dataloaders WITHOUT visual embeddings ────────────
    ablation_config = copy.deepcopy(config)
    ablation_config["hero"]["use_visual"] = False
    logger.info("Building dataloaders (use_visual=False) …")
    train_loader, val_loader, test_loader, meta = build_dataloaders(
        ablation_config, mode="hero",
    )
    num_items = meta["num_items"]
    item_sales_counts = meta["item_sales_counts"]
    idx_to_id = meta["idx_to_id"]

    # ── Pre-compute pop logits (same as trainer.py) ───────────
    pop_logit_dict = popularity_logit_scores(item_sales_counts)
    min_logit = min(pop_logit_dict.values()) if pop_logit_dict else 0.0
    pop_logits_tensor = torch.full((num_items,), min_logit, dtype=torch.float32)
    pop_logits_tensor[0] = 0.0
    for idx, aid in idx_to_id.items():
        if idx == 0:
            continue
        pop_logits_tensor[idx] = pop_logit_dict.get(aid, min_logit)
    pop_logits_tensor = pop_logits_tensor.to(device)

    # ── Step 1: Train ID-only Hero ────────────────────────────
    logger.info("\n── Step 1: Training ID-only Hero from scratch ──")
    model, best_val_ndcg, history = _train_id_only_hero(
        config, train_loader, val_loader, meta, pop_logits_tensor, device,
    )

    # ── Step 2: Full test evaluation ──────────────────────────
    logger.info("\n── Step 2: Evaluating ID-only Hero on test set ──")
    test_metrics = _evaluate_id_only(model, test_loader, device, meta, config)

    eval_k = config["evaluation"]["k"]
    logger.info("─── ID-only Hero — Test Metrics ───")
    for k_name, v in test_metrics.items():
        if isinstance(v, (int, float)):
            logger.info(f"  {k_name}: {v:.4f}")

    # ── Step 3: Cold-start simulation ─────────────────────────
    logger.info("\n── Step 3: Cold-start simulation (ID-only Hero) ──")
    cold_start = _cold_start_id_only(model, config, device)
    logger.info(f"  Avg Rank: {cold_start['avg_rank']:.1f}, "
                f"Hit@12: {cold_start['hit_at_12']:.1%}")

    # ── Step 4: Load existing results for comparison ──────────
    logger.info("\n── Step 4: Building comparison table ──")

    villain_results_path = os.path.join(output_dir, "villain_baseline_results.json")
    hero_results_path = os.path.join(output_dir, "hero_baseline_results.json")
    hero_cold_path = os.path.join(output_dir, "hero_cold_start_results.json")

    villain_test = {}
    if os.path.exists(villain_results_path):
        with open(villain_results_path) as f:
            villain_test = json.load(f).get("test_metrics", {})

    hero_visual_test = {}
    if os.path.exists(hero_results_path):
        with open(hero_results_path) as f:
            hero_visual_test = json.load(f).get("test_metrics", {})

    # Read cold-start file once (contains both villain and hero cold-start data)
    cold_start_data = {}
    if os.path.exists(hero_cold_path):
        with open(hero_cold_path) as f:
            cold_start_data = json.load(f)

    hero_visual_cold = cold_start_data.get("hero", {})
    villain_cold = cold_start_data.get("villain", {})

    # ── Comparison table ──────────────────────────────────────
    comparison = {
        "villain": {
            "ndcg@12": villain_test.get(f"ndcg@{eval_k}", None),
            "mrr": villain_test.get("mrr", None),
            "catalog_coverage": villain_test.get("catalog_coverage", None),
            "cold_start_avg_rank": villain_cold.get("avg_rank", None),
        },
        "hero_id_only": {
            "ndcg@12": test_metrics.get(f"ndcg@{eval_k}", None),
            "mrr": test_metrics.get("mrr", None),
            "catalog_coverage": test_metrics.get("catalog_coverage", None),
            "tail_item_rate": test_metrics.get("tail_item_rate@k", None),
            "cold_start_avg_rank": cold_start["avg_rank"],
        },
        "hero_visual": {
            "ndcg@12": hero_visual_test.get(f"ndcg@{eval_k}", None),
            "mrr": hero_visual_test.get("mrr", None),
            "catalog_coverage": hero_visual_test.get("catalog_coverage", None),
            "cold_start_avg_rank": hero_visual_cold.get("avg_rank", None),
        },
    }

    # Compute deltas
    if comparison["hero_visual"]["ndcg@12"] and comparison["hero_id_only"]["ndcg@12"]:
        comparison["visual_lift"] = {
            "delta_ndcg": round(comparison["hero_visual"]["ndcg@12"] - comparison["hero_id_only"]["ndcg@12"], 6),
            "delta_coverage": round(
                (comparison["hero_visual"]["catalog_coverage"] or 0) -
                (comparison["hero_id_only"]["catalog_coverage"] or 0), 6
            ),
            "delta_cold_start_rank": round(
                (comparison["hero_id_only"]["cold_start_avg_rank"] or 0) -
                (comparison["hero_visual"]["cold_start_avg_rank"] or 0), 1
            ),
        }

    # ── Print comparison ──────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("  ABLATION COMPARISON TABLE")
    logger.info("=" * 70)
    logger.info(f"{'Model':<20} | {'nDCG@12':>8} | {'MRR':>8} | {'Coverage':>8} | {'Cold-Start Rank':>15}")
    logger.info("─" * 70)
    for label, data in [("Villain", comparison["villain"]),
                         ("Hero (ID-only)", comparison["hero_id_only"]),
                         ("Hero (visual)", comparison["hero_visual"])]:
        ndcg = f"{data['ndcg@12']:.4f}" if data.get("ndcg@12") else "N/A"
        mrr_val = f"{data['mrr']:.4f}" if data.get("mrr") else "N/A"
        cov = f"{data['catalog_coverage']:.4f}" if data.get("catalog_coverage") else "N/A"
        cs = f"{data['cold_start_avg_rank']:.0f}" if data.get("cold_start_avg_rank") else "N/A"
        logger.info(f"{label:<20} | {ndcg:>8} | {mrr_val:>8} | {cov:>8} | {cs:>15}")

    if "visual_lift" in comparison:
        vl = comparison["visual_lift"]
        logger.info("─" * 70)
        logger.info(
            f"Visual Lift:  nDCG {vl['delta_ndcg']:+.4f},  "
            f"Coverage {vl['delta_coverage']:+.4f},  "
            f"Cold-Start Rank {vl['delta_cold_start_rank']:+.0f} (lower = better)"
        )

    # ── Save full results ─────────────────────────────────────
    results = {
        "id_only_hero": {
            "best_val_ndcg": best_val_ndcg,
            "test_metrics": test_metrics,
            "cold_start": cold_start,
            "total_epochs": len(history),
            "history": history,
        },
        "comparison": comparison,
    }

    results_path = os.path.join(output_dir, "hero_ablation_no_visual.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nAblation results saved → {results_path}")

    return results


if __name__ == "__main__":
    cfg = load_config()
    run_ablation(cfg)
