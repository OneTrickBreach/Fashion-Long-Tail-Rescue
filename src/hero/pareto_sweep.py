"""
pareto_sweep.py — Pareto Front λ-Discovery Sweep
===================================================
Team member: Ishan Biswas
Key functions: run_pareto_sweep, finetune_and_evaluate

PURPOSE:
    Iterates over config.yaml → pareto.lambda_values, fine-tunes the Hero
    from the Phase 2 checkpoint for each λ_disc value, evaluates on the test
    set with multi-objective metrics, and collects all Pareto points into
    outputs/pareto_sweep_results.json.

    Training strategy: load hero_best.pt → fine-tune 20 epochs per λ → evaluate.
    This avoids training from scratch for every sweep point.
"""

import os
import copy
import json
import time
import logging
import torch
import torch.nn as nn

from src.utils.helpers import load_config, set_seed, get_device, setup_logging
from src.data.dataset import build_dataloaders
from src.hero.model import HeroModel
from src.hero.contrastive import MultiObjectiveLoss, hard_negative_mining
from src.utils.metrics import (
    compute_all_metrics,
    compute_multi_objective_metrics,
    popularity_logit_scores,
)

logger = logging.getLogger("seeing-the-unseen")

# ──────────────────────────────────────────────────────────────
# Fine-tune + evaluate for a single λ value
# ──────────────────────────────────────────────────────────────

# Fine-tune budget read from config.yaml → pareto.finetune_epochs at runtime


@torch.no_grad()
def _evaluate_sweep_point(
    model: HeroModel,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    meta: dict,
    config: dict,
) -> dict:
    """
    Evaluate a single sweep point with full multi-objective metrics.

    Returns dict with ndcg@12, mrr, catalog_coverage, tail_item_rate@k,
    mean_tail_score@k.
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

    for batch in test_loader:
        item_seq = batch["item_seq"].to(device)
        positions = batch["positions"].to(device)
        targets = batch["target"].to(device)

        visual_embeds = None
        if "visual_embeds" in batch:
            visual_embeds = batch["visual_embeds"].to(device)

        logits, _ = model(item_seq, positions, visual_embeds)

        # Exclude PAD index 0
        logits[:, 0] = float("-inf")
        _, top_indices = logits.topk(eval_k, dim=-1)

        all_predictions.extend(top_indices.cpu().tolist())
        all_targets.extend(targets.cpu().tolist())

    # Map contiguous indices back to raw article IDs for metrics that need
    # sales counts keyed by raw article_id
    all_preds_raw = [
        [idx_to_id.get(idx, -1) for idx in pred_list]
        for pred_list in all_predictions
    ]
    all_targets_raw = [idx_to_id.get(t, -1) for t in all_targets]

    metrics = compute_multi_objective_metrics(
        all_preds_raw,
        all_targets_raw,
        catalog_size,
        item_sales_counts,
        k=eval_k,
        tail_threshold=tail_threshold,
    )
    return metrics


def finetune_and_evaluate(
    lambda_disc: float,
    config: dict,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    meta: dict,
    pop_logits_tensor: torch.Tensor,
    base_checkpoint_path: str,
    device: torch.device,
) -> dict:
    """
    Fine-tune the Hero from the Phase 2 checkpoint with a specific λ_disc,
    then evaluate on the test set.

    Returns a result dict for this Pareto point.
    """
    hero_cfg = config["hero"]
    eval_k = config["evaluation"]["k"]
    ckpt_dir = config["paths"]["checkpoints"]
    num_items = meta["num_items"]
    finetune_epochs = config["pareto"].get("finetune_epochs", 20)

    logger.info(f"\n{'='*60}")
    logger.info(f"  Pareto Sweep — λ_disc = {lambda_disc}")
    logger.info(f"{'='*60}")

    # ── Build a config copy with this λ_disc ──────────────────
    sweep_config = copy.deepcopy(config)
    sweep_config["hero"]["discovery_weight"] = lambda_disc

    # ── Load fresh model from Phase 2 checkpoint ──────────────
    model = HeroModel(config=config, num_items=num_items).to(device)
    ckpt = torch.load(base_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(
        f"  Loaded Phase 2 checkpoint (epoch {ckpt['epoch']+1}, "
        f"nDCG={ckpt['best_ndcg']:.4f})"
    )

    # ── Fresh optimizer for fine-tuning ───────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hero_cfg.get("lr", 0.0005),
        weight_decay=hero_cfg.get("weight_decay", 0.01),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3,
    )

    # ── Loss with this λ_disc ─────────────────────────────────
    criterion = MultiObjectiveLoss(sweep_config).to(device)
    cl_cfg = hero_cfg.get("contrastive", {})
    num_negatives = cl_cfg.get("hard_negatives", 10)
    mining_mode = cl_cfg.get("mining_mode", "random")
    jaccard_low = cl_cfg.get("jaccard_low", 0.3)
    jaccard_high = cl_cfg.get("jaccard_high", 0.7)
    item_attributes = meta.get("item_attributes", {})
    hn_cache_path = os.path.join(ckpt_dir, "hard_negatives_cache.pt")

    # ── Fine-tune loop ────────────────────────────────────────
    best_val_ndcg = 0.0
    best_model_state = None
    patience = hero_cfg.get("patience", 10)
    epochs_without_improvement = 0

    logger.info(
        f"  Fine-tuning for {finetune_epochs} epochs "
        f"(patience={patience}, lr={hero_cfg.get('lr', 0.0005)})"
    )
    logger.info(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val nDCG':>8} | "
                f"{'Val MRR':>8} | {'Val Cov':>8} | {'Time':>6}")
    logger.info("─" * 60)

    for epoch in range(finetune_epochs):
        t0 = time.time()

        # Pre-compute hard negatives
        hard_negatives = hard_negative_mining(
            num_items=num_items,
            num_negatives=num_negatives,
            item_attributes=item_attributes,
            mining_mode=mining_mode,
            jaccard_low=jaccard_low,
            jaccard_high=jaccard_high,
            cache_path=hn_cache_path,
        ).to(device)

        # ── Train one epoch ───────────────────────────────────
        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            item_seq = batch["item_seq"].to(device)
            positions = batch["positions"].to(device)
            targets = batch["target"].to(device)

            visual_embeds = None
            if "visual_embeds" in batch:
                visual_embeds = batch["visual_embeds"].to(device)

            optimizer.zero_grad()

            logits, contrastive_embeds = model(item_seq, positions, visual_embeds)

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
                visual_embeds = None
                if "visual_embeds" in batch:
                    visual_embeds = batch["visual_embeds"].to(device)
                logits, _ = model(item_seq, positions, visual_embeds)
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

        # ── Best model tracking ───────────────────────────────
        if val_ndcg > best_val_ndcg:
            best_val_ndcg = val_ndcg
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            logger.info(
                f"  Early stopping at fine-tune epoch {epoch+1} "
                f"(no improvement for {patience} epochs)"
            )
            break

    # ── Restore best model ────────────────────────────────────
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    logger.info(f"  Best val nDCG@{eval_k} during fine-tune: {best_val_ndcg:.4f}")

    # ── Save fine-tuned checkpoint ────────────────────────────
    lambda_tag = f"{lambda_disc:.1f}".replace(".", "_")
    ckpt_path = os.path.join(ckpt_dir, f"hero_lambda_{lambda_tag}.pt")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "lambda_disc": lambda_disc,
         "best_val_ndcg": best_val_ndcg},
        ckpt_path,
    )
    logger.info(f"  Checkpoint saved → {ckpt_path}")

    # ── Full test evaluation ──────────────────────────────────
    logger.info("  Running full test evaluation …")
    test_metrics = _evaluate_sweep_point(model, test_loader, device, meta, config)

    logger.info(f"  Test results for λ_disc={lambda_disc}:")
    for k_name, v in test_metrics.items():
        if isinstance(v, float):
            logger.info(f"    {k_name}: {v:.4f}")

    result = {
        "lambda_disc": lambda_disc,
        "best_val_ndcg": round(best_val_ndcg, 6),
        **{k: round(v, 6) if isinstance(v, float) else v
           for k, v in test_metrics.items()},
        "checkpoint": ckpt_path,
    }
    return result


# ──────────────────────────────────────────────────────────────
# Main sweep driver
# ──────────────────────────────────────────────────────────────

def run_pareto_sweep(config: dict) -> list[dict]:
    """
    Run the full Pareto λ-discovery sweep.

    Reads lambda_values from config.yaml → pareto.lambda_values,
    fine-tunes the Hero for each, evaluates, and saves results to
    outputs/pareto_sweep_results.json.
    """
    setup_logging()
    set_seed(config["project"]["seed"])
    device = get_device(config["embedding"]["device"])

    pareto_cfg = config["pareto"]
    lambda_values = pareto_cfg["lambda_values"]
    ckpt_dir = config["paths"]["checkpoints"]
    output_dir = config["paths"]["outputs"]
    os.makedirs(output_dir, exist_ok=True)

    base_ckpt = os.path.join(ckpt_dir, "hero_best.pt")
    if not os.path.exists(base_ckpt):
        raise FileNotFoundError(
            f"Phase 2 Hero checkpoint not found at {base_ckpt}. "
            "Run Phase 2 training first."
        )

    # ── Load data once (shared across all λ values) ───────────
    logger.info("Building dataloaders (shared across sweep) …")
    train_loader, val_loader, test_loader, meta = build_dataloaders(
        config, mode="hero",
    )
    num_items = meta["num_items"]
    idx_to_id = meta["idx_to_id"]
    item_sales_counts = meta["item_sales_counts"]

    # ── Pre-compute popularity logits (same as trainer.py) ────
    pop_logit_dict = popularity_logit_scores(item_sales_counts)
    min_logit = min(pop_logit_dict.values()) if pop_logit_dict else 0.0
    pop_logits_tensor = torch.full((num_items,), min_logit, dtype=torch.float32)
    pop_logits_tensor[0] = 0.0  # PAD — neutral
    for idx, aid in idx_to_id.items():
        if idx == 0:
            continue
        pop_logits_tensor[idx] = pop_logit_dict.get(aid, min_logit)
    pop_logits_tensor = pop_logits_tensor.to(device)

    logger.info(f"Sweep plan: λ_disc ∈ {lambda_values}")
    logger.info(f"Base checkpoint: {base_ckpt}")
    finetune_epochs = pareto_cfg.get("finetune_epochs", 20)
    logger.info(f"Fine-tune epochs per λ: {finetune_epochs}")

    # ── Sweep ─────────────────────────────────────────────────
    all_results = []
    for lam in lambda_values:
        result = finetune_and_evaluate(
            lambda_disc=lam,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            meta=meta,
            pop_logits_tensor=pop_logits_tensor,
            base_checkpoint_path=base_ckpt,
            device=device,
        )
        all_results.append(result)

        # Save incrementally (so partial results survive interruption)
        results_path = os.path.join(output_dir, "pareto_sweep_results.json")
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"  Incremental save → {results_path}")

    # ── Final summary ─────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("  Pareto Sweep Complete — Summary")
    logger.info(f"{'='*60}")
    logger.info(f"{'λ_disc':>8} | {'nDCG@12':>8} | {'MRR':>8} | {'Coverage':>8} | {'Tail Rate':>10}")
    logger.info("─" * 55)
    eval_k = config["evaluation"]["k"]
    for r in all_results:
        logger.info(
            f"{r['lambda_disc']:>8.1f} | "
            f"{r.get(f'ndcg@{eval_k}', 0):>8.4f} | "
            f"{r.get('mrr', 0):>8.4f} | "
            f"{r.get('catalog_coverage', 0):>8.4f} | "
            f"{r.get('tail_item_rate@k', 0):>10.4f}"
        )

    logger.info(f"\nResults saved → {results_path}")
    return all_results


if __name__ == "__main__":
    cfg = load_config()
    run_pareto_sweep(cfg)
