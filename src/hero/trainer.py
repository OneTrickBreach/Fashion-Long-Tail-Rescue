"""
trainer.py — Training Loop for the Hero Model
===============================================
Team: Ishan, Elizabeth, Nishant

PURPOSE:
    Handles the end-to-end training pipeline for the Hero model,
    including the multi-objective loss (CE + contrastive + discovery).
"""

import os
import json
import time
import logging
import torch
import torch.nn as nn
import numpy as np
from src.utils.helpers import load_config, set_seed, get_device, setup_logging
from src.data.dataset import build_dataloaders
from src.hero.model import HeroModel
from src.hero.contrastive import MultiObjectiveLoss, hard_negative_mining
from src.utils.metrics import compute_all_metrics, popularity_logit_scores

logger = logging.getLogger("seeing-the-unseen")


# ──────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────

def _checkpoint_path(ckpt_dir: str, tag: str) -> str:
    """Return full path for a checkpoint file."""
    return os.path.join(ckpt_dir, f"hero_{tag}.pt")


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_ndcg: float,
    history: list[dict],
) -> None:
    """Save a full training checkpoint to disk."""
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
    logger.info(f"  💾 Checkpoint saved → {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
) -> tuple[int, float, list[dict]]:
    """
    Restore training state from a checkpoint.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    start_epoch = ckpt["epoch"] + 1  # resume from the *next* epoch
    best_ndcg = ckpt["best_ndcg"]
    history = ckpt.get("history", [])
    logger.info(
        f"  ✅ Resumed from checkpoint: epoch {ckpt['epoch']}, "
        f"best nDCG@12={best_ndcg:.4f}"
    )
    return start_epoch, best_ndcg, history


# ──────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: HeroModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    k: int = 12,
    catalog_size: int = 1,
) -> dict:
    """
    Run the model on a val/test DataLoader and compute nDCG@K, MRR,
    and catalog coverage.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    all_predictions = []
    all_targets = []
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        item_seq = batch["item_seq"].to(device)
        positions = batch["positions"].to(device)
        targets = batch["target"].to(device)
        
        visual_embeds = None
        if "visual_embeds" in batch:
            visual_embeds = batch["visual_embeds"].to(device)

        logits, _ = model(item_seq, positions, visual_embeds)
        loss = criterion(logits, targets)
        
        total_loss += loss.item()
        n_batches += 1

        # Top-K predictions (exclude PAD index 0)
        logits[:, 0] = float("-inf")
        _, top_indices = logits.topk(k, dim=-1)

        all_predictions.extend(top_indices.cpu().tolist())
        all_targets.extend(targets.cpu().tolist())

    metrics = compute_all_metrics(
        all_predictions, all_targets, catalog_size, k=k,
    )
    metrics["avg_loss"] = total_loss / max(n_batches, 1)
    return metrics


# ──────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────

def train_hero(config: dict) -> dict:
    """
    Full training pipeline for the Hero baseline.
    """
    setup_logging()
    set_seed(config["project"]["seed"])
    device = get_device(config["embedding"]["device"])

    hero_cfg = config["hero"]
    eval_k = config["evaluation"]["k"]
    ckpt_dir = config["paths"]["checkpoints"]
    output_dir = config["paths"]["outputs"]
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # ── DataLoaders ──────────────────────────────────────────
    logger.info("Building dataloaders …")
    train_loader, val_loader, test_loader, meta = build_dataloaders(
        config, mode="hero",
    )
    num_items = meta["num_items"]
    catalog_size = num_items - 1
    idx_to_id = meta["idx_to_id"]
    item_sales_counts = meta["item_sales_counts"]
    item_attributes = meta.get("item_attributes", {})

    # ── Phase 3: Popularity Logits Pre-computation ───────────
    # Reuse sales counts already computed in build_dataloaders (rules.md §6)
    pop_logit_dict = popularity_logit_scores(item_sales_counts)
    
    # Map raw-article-id logits → contiguous-index tensor
    # Zero-transaction items are NOT in pop_logit_dict. Using 0.0 as fallback
    # would wrongly treat them as *more* popular than rare items (whose logits
    # are negative). Instead, default to the minimum logit so unseen items are
    # treated as the least popular.
    min_logit = min(pop_logit_dict.values()) if pop_logit_dict else 0.0
    pop_logits_tensor = torch.full((num_items,), min_logit, dtype=torch.float32)
    pop_logits_tensor[0] = 0.0  # PAD index — neutral, no penalty
    for idx, aid in idx_to_id.items():
        if idx == 0:  # PAD
            continue
        pop_logits_tensor[idx] = pop_logit_dict.get(aid, min_logit)
        
    pop_logits_tensor = pop_logits_tensor.to(device)
    logger.info(
        f"Pre-computed popularity logits for {num_items} items "
        f"(min={min_logit:.2f}, max={max(pop_logit_dict.values()):.2f})."
    )

    # ── Model ────────────────────────────────────────────────
    logger.info("Initialising HeroModel …")
    model = HeroModel(config=config, num_items=num_items).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Parameters: {total_params:,}")

    # ── Optimizer & Scheduler ────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hero_cfg.get("lr", 0.0005),
        weight_decay=hero_cfg.get("weight_decay", 0.01),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3,
    )

    # ── Training config ──────────────────────────────────────
    epochs = hero_cfg.get("epochs", 50)
    checkpoint_every = hero_cfg.get("checkpoint_every", 5)
    patience = hero_cfg.get("patience", 10)
    
    criterion = MultiObjectiveLoss(config).to(device)
    cl_cfg = hero_cfg.get("contrastive", {})
    num_negatives = cl_cfg.get("hard_negatives", 10)
    mining_mode = cl_cfg.get("mining_mode", "random")
    jaccard_low = cl_cfg.get("jaccard_low", 0.3)
    jaccard_high = cl_cfg.get("jaccard_high", 0.7)
    hn_cache_path = os.path.join(ckpt_dir, "hard_negatives_cache.pt")

    # ── Checkpoint resume ────────────────────────────────────
    latest_path = _checkpoint_path(ckpt_dir, "latest")
    best_path = _checkpoint_path(ckpt_dir, "best")
    start_epoch = 0
    best_ndcg = 0.0
    history: list[dict] = []
    epochs_without_improvement = 0

    if os.path.exists(latest_path):
        logger.info(f"Found existing checkpoint: {latest_path}")
        start_epoch, best_ndcg, history = load_checkpoint(
            latest_path, model, optimizer, scheduler, device,
        )
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
    else:
        logger.info("No checkpoint found — starting fresh.")

    if start_epoch >= epochs:
        logger.info(f"Training already completed ({start_epoch}/{epochs} epochs).")
        logger.info("Running final test evaluation …")
        test_metrics = evaluate(model, test_loader, device, eval_k, catalog_size)
        return test_metrics

    # ── Training loop ────────────────────────────────────────
    logger.info(
        f"Training: epochs {start_epoch+1}..{epochs}, "
        f"checkpoint_every={checkpoint_every}, patience={patience}"
    )
    logger.info(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>8} | "
                f"{'nDCG@12':>8} | {'MRR':>8} | {'Coverage':>8} | {'LR':>10} | {'Time':>6}")
    logger.info("─" * 82)

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        # Pre-compute hard negatives for this epoch
        hard_negatives = hard_negative_mining(
            num_items=num_items,
            num_negatives=num_negatives,
            item_attributes=item_attributes,
            mining_mode=mining_mode,
            jaccard_low=jaccard_low,
            jaccard_high=jaccard_high,
            cache_path=hn_cache_path,
        ).to(device)

        # ── Train one epoch ──────────────────────────────────
        model.train()
        running_loss = 0.0
        running_ce = 0.0
        running_cl = 0.0
        running_disc = 0.0
        n_batches = 0

        for batch in train_loader:
            item_seq = batch["item_seq"].to(device)
            positions = batch["positions"].to(device)
            targets = batch["target"].to(device)
            
            visual_embeds = None
            if "visual_embeds" in batch:
                visual_embeds = batch["visual_embeds"].to(device)

            optimizer.zero_grad()
            
            # Forward pass
            logits, contrastive_embeds = model(item_seq, positions, visual_embeds)
            
            # Gather contrastive components
            # Positive embeddings for the targets
            positive_embeds = model.item_emb(targets)
            
            # Negative embeddings for the targets
            # targets: (B,) -> hard_negatives[targets]: (B, num_negatives)
            batch_neg_indices = hard_negatives[targets]
            negative_embeds = model.item_emb(batch_neg_indices)
            
            # Compute multi-objective loss
            loss, loss_ce, loss_cl, loss_disc = criterion(
                logits, 
                targets, 
                anchor=contrastive_embeds, 
                positive=positive_embeds, 
                negatives=negative_embeds,
                pop_logits=pop_logits_tensor
            )
            
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item()
            running_ce += loss_ce.item()
            running_cl += loss_cl.item()
            running_disc += loss_disc.item()
            n_batches += 1

        avg_train_loss = running_loss / max(n_batches, 1)
        avg_ce_loss = running_ce / max(n_batches, 1)
        avg_cl_loss = running_cl / max(n_batches, 1)
        avg_disc_loss = running_disc / max(n_batches, 1)

        # ── Validate ─────────────────────────────────────────
        val_metrics = evaluate(model, val_loader, device, eval_k, catalog_size)
        val_ndcg = val_metrics[f"ndcg@{eval_k}"]
        val_mrr = val_metrics["mrr"]
        val_cov = val_metrics["catalog_coverage"]
        val_loss = val_metrics["avg_loss"]

        # Step the LR scheduler
        scheduler.step(val_ndcg)
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - t0

        # ── Log ──────────────────────────────────────────────
        logger.info(
            f"{epoch+1:>5} | {avg_train_loss:>10.4f} | {val_loss:>8.4f} | "
            f"{val_ndcg:>8.4f} | {val_mrr:>8.4f} | {val_cov:>8.4f} | "
            f"{current_lr:>10.6f} | {elapsed:>5.1f}s"
        )
        logger.debug(
            f"          └─ Losses — CE: {avg_ce_loss:.4f}, CL: {avg_cl_loss:.4f}, Disc: {avg_disc_loss:.4f}"
        )

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_ce": avg_ce_loss,
            "train_cl": avg_cl_loss,
            "train_disc": avg_disc_loss,
            "val_loss": val_loss,
            f"val_ndcg@{eval_k}": val_ndcg,
            "val_mrr": val_mrr,
            "val_catalog_coverage": val_cov,
            "lr": current_lr,
            "elapsed_s": round(elapsed, 1),
        }
        history.append(epoch_record)

        # ── Best model? ──────────────────────────────────────
        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            epochs_without_improvement = 0
            save_checkpoint(
                best_path, model, optimizer, scheduler,
                epoch, best_ndcg, history,
            )
        else:
            epochs_without_improvement += 1

        # ── Periodic checkpoint ──────────────────────────────
        if (epoch + 1) % checkpoint_every == 0 or epoch == epochs - 1:
            save_checkpoint(
                latest_path, model, optimizer, scheduler,
                epoch, best_ndcg, history,
            )

        # ── Early stopping ───────────────────────────────────
        if epochs_without_improvement >= patience:
            logger.info(
                f"  ⏹ Early stopping at epoch {epoch+1} "
                f"(no improvement for {patience} epochs)"
            )
            # Save final state
            save_checkpoint(
                latest_path, model, optimizer, scheduler,
                epoch, best_ndcg, history,
            )
            break

    # ── Load best model for final test eval ──────────────────
    logger.info("─" * 82)
    logger.info(f"Training complete. Best val nDCG@{eval_k}: {best_ndcg:.4f}")

    if os.path.exists(best_path):
        logger.info(f"Loading best model from {best_path} for test evaluation …")
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    # ── Test evaluation ──────────────────────────────────────
    test_metrics = evaluate(model, test_loader, device, eval_k, catalog_size)
    logger.info(f"Test results:")
    for k_name, v in test_metrics.items():
        logger.info(f"  {k_name}: {v:.4f}")

    # ── Save results ─────────────────────────────────────────
    results = {
        "best_val_ndcg": best_ndcg,
        "test_metrics": test_metrics,
        "total_epochs": len(history),
        "history": history,
    }
    results_path = os.path.join(output_dir, "hero_baseline_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved → {results_path}")

    return test_metrics


if __name__ == "__main__":
    cfg = load_config()
    train_hero(cfg)
