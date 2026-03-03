"""
contrastive.py — Attribute-Aware Contrastive Learning
======================================================
Team: Ishan, Elizabeth, Nishant

PURPOSE:
    Implements the contrastive learning objective that encourages the model
    to learn fine-grained visual similarities between items sharing attributes
    (e.g., same colour family, similar silhouette) while pushing apart items
    with different attributes.

    This is the key mechanism for rescuing long-tail items: even if a niche
    item has few transactions, it can borrow representation strength from
    visually similar popular items.

KEY COMPONENTS:
    - ContrastiveLoss:       InfoNCE / NT-Xent style loss with temperature
    - MultiObjectiveLoss:    Phase 3 three-term loss: L_CE + λ_CL * L_CL + λ_disc * L_discovery
                             Discovery term penalises placing softmax mass on popular items.
    - hard_negative_mining:  Select informative negatives based on attribute
                             overlap (items that share SOME but not ALL
                             attributes are the hardest negatives)
"""

import os
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("seeing-the-unseen")


class ContrastiveLoss(nn.Module):
    """Attribute-Aware Contrastive Loss (InfoNCE variant)."""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, negatives):
        """
        Args:
            anchor (Tensor):    (batch_size, hidden_dim)
            positive (Tensor):  (batch_size, hidden_dim)
            negatives (Tensor): (batch_size, num_negatives, hidden_dim)

        Returns:
            Tensor: scalar loss
        """
        # Normalize embeddings for cosine similarity
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negatives = F.normalize(negatives, p=2, dim=2)
        
        # Positive similarities (B, 1)
        sim_pos = torch.sum(anchor * positive, dim=1).unsqueeze(1)
        
        # Negative similarities (B, num_negatives)
        # anchor: (B, 1, hidden_dim) * negatives: (B, num_negatives, hidden_dim) -> sum -> (B, num_negatives)
        sim_neg = torch.sum(anchor.unsqueeze(1) * negatives, dim=2)
        
        # Logits: (B, 1 + num_negatives)
        logits = torch.cat([sim_pos, sim_neg], dim=1) / self.temperature
        
        # Labels: the positive is always at index 0
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=anchor.device)
        
        return F.cross_entropy(logits, labels)


class MultiObjectiveLoss(nn.Module):
    """
    Multi-Objective Loss for HeroModel (Phase 3):
    L_total = L_CE(predictions, targets) 
            + lambda_CL * L_contrastive(anchor, pos, negs)
            + lambda_disc * L_discovery(logits, pop_logits)
    """
    def __init__(self, config: dict):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=0) # PAD index is 0
        
        cl_config = config.get("hero", {}).get("contrastive", {})
        temperature = cl_config.get("temperature", 0.07)
        self.cl_weight = cl_config.get("weight", 0.3)
        self.disc_weight = config.get("hero", {}).get("discovery_weight", 0.0)
        
        self.contrastive_loss = ContrastiveLoss(temperature=temperature)

    def forward(self, logits, targets, anchor, positive, negatives, pop_logits=None):
        """
        Computes the multi-objective loss.
        """
        loss_ce = self.ce_loss(logits, targets)
        loss_cl = self.contrastive_loss(anchor, positive, negatives)
        
        loss_total = loss_ce + self.cl_weight * loss_cl
        
        # Phase 3: Discovery Loss
        # Always return a tensor for consistent API (zero when disabled)
        loss_disc = torch.tensor(0.0, device=logits.device)
        if self.disc_weight > 0.0 and pop_logits is not None:
            # logits: (batch_size, num_items)
            # pop_logits: (num_items,) — pre-computed, PAD index should be 0.0
            probs = F.softmax(logits, dim=-1)
            # L_discovery = mean over batch of dot(probs, pop_logits)
            # Penalises placing high softmax mass on high-popularity items
            loss_disc = (probs * pop_logits.unsqueeze(0)).sum(dim=1).mean()
            loss_total = loss_total + self.disc_weight * loss_disc
            
        return loss_total, loss_ce, loss_cl, loss_disc


def hard_negative_mining(
    num_items: int,
    num_negatives: int = 10,
    item_attributes: dict[int, set] | None = None,
    mining_mode: str = "attribute",
    jaccard_low: float = 0.3,
    jaccard_high: float = 0.7,
    cache_path: str | None = None,
) -> torch.Tensor:
    """
    Select hard negatives per item using attribute-aware Jaccard similarity.

    Phase 3 upgrade: instead of random negatives, mine items that share
    *some* but not *all* attributes with the anchor (Jaccard between
    jaccard_low and jaccard_high).  This produces "hard" negatives that
    are semantically close enough to challenge the contrastive head.

    For items with fewer than `num_negatives` valid hard negatives in the
    Jaccard range, the shortfall is filled with random sampling (fallback).

    The full (num_items, num_negatives) index matrix is pre-computed once
    per epoch and cached to a .pt file to avoid redundant O(n²) work on
    subsequent calls within the same run.

    Args:
        num_items:        Total item count (including PAD at index 0).
        num_negatives:    Number of negatives to mine per anchor.
        item_attributes:  {contiguous_idx: set(attribute_values)}.
                          If None or empty, falls back to random mining.
        mining_mode:      "attribute" for Jaccard mining, "random" for
                          uniform random (Phase 2 behaviour).
        jaccard_low:      Minimum Jaccard similarity for a hard negative.
        jaccard_high:     Maximum Jaccard similarity for a hard negative.
        cache_path:       Optional path to save/load the pre-computed
                          hard-negative index matrix as a .pt file.

    Returns:
        Tensor: (num_items, num_negatives) — indices of hard negatives.
                Index 0 (PAD) row is filled with random indices.
    """
    # ── Try loading from cache ────────────────────────────────
    if cache_path and os.path.exists(cache_path):
        cached_data = torch.load(cache_path, map_location="cpu", weights_only=False)
        # Guard against stale cache: check shape AND Jaccard params
        if isinstance(cached_data, dict):
            mat = cached_data.get("matrix")
            if (mat is not None
                    and mat.shape == (num_items, num_negatives)
                    and cached_data.get("jaccard_low") == jaccard_low
                    and cached_data.get("jaccard_high") == jaccard_high
                    and cached_data.get("mining_mode") == mining_mode):
                logger.info(f"  Loaded cached hard-negative matrix from {cache_path}")
                return mat
            else:
                logger.info("  Cache params mismatch — recomputing.")
        else:
            # Legacy format (raw tensor) — recompute
            logger.info("  Legacy cache format — recomputing.")

    # ── Random fallback (Phase 2 behaviour) ───────────────────
    if mining_mode == "random" or not item_attributes:
        logger.info("  Using random hard-negative mining (no attributes).")
        hard_negatives = torch.randint(1, num_items, size=(num_items, num_negatives))
        return hard_negatives

    # ── Attribute-aware Jaccard mining ────────────────────────
    logger.info(
        f"  Pre-computing attribute-aware hard negatives "
        f"(Jaccard [{jaccard_low:.2f}, {jaccard_high:.2f}], "
        f"{num_items} items) …"
    )

    # Build a binary attribute matrix for vectorised Jaccard.
    # Collect all unique attribute values across all items.
    all_attrs = set()
    for attrs in item_attributes.values():
        all_attrs.update(attrs)
    attr_list = sorted(all_attrs)
    attr_to_col = {a: i for i, a in enumerate(attr_list)}
    num_attrs = len(attr_list)

    # Binary matrix: (num_items, num_attrs)  — row 0 (PAD) is all-zero
    attr_matrix = np.zeros((num_items, num_attrs), dtype=np.float32)
    for idx, attrs in item_attributes.items():
        for a in attrs:
            attr_matrix[idx, attr_to_col[a]] = 1.0

    # Vectorised pairwise Jaccard in chunks to limit peak memory.
    # Jaccard(A, B) = |A ∩ B| / |A ∪ B|
    # Using dot products:  intersection = A @ B^T,  union = |A| + |B| - intersection
    hard_negatives = torch.zeros((num_items, num_negatives), dtype=torch.long)

    attr_tensor = torch.from_numpy(attr_matrix)  # (N, D)
    row_sums = attr_tensor.sum(dim=1)             # (N,)

    # Process in chunks to avoid O(N²) memory spike
    chunk_size = 512

    for start in range(1, num_items, chunk_size):
        end = min(start + chunk_size, num_items)
        chunk = attr_tensor[start:end]           # (C, D)
        chunk_sums = row_sums[start:end]          # (C,)

        # intersection: (C, N)
        intersection = chunk @ attr_tensor.T
        # union: (C, N)
        union = chunk_sums.unsqueeze(1) + row_sums.unsqueeze(0) - intersection

        # Jaccard similarity: (C, N)
        # Avoid division by zero (both items have no attributes)
        jaccard = torch.where(
            union > 0,
            intersection / union,
            torch.zeros_like(union),
        )

        # Vectorised masking: exclude PAD (col 0) and self for all rows in chunk
        jaccard[:, 0] = -1.0
        diag_indices = torch.arange(end - start)
        jaccard[diag_indices, diag_indices + start] = -1.0

        # Find items in the Jaccard range [jaccard_low, jaccard_high]
        for local_i in range(end - start):
            global_i = start + local_i
            row = jaccard[local_i]
            mask = (row >= jaccard_low) & (row <= jaccard_high)
            candidates = torch.where(mask)[0]

            if len(candidates) >= num_negatives:
                # Randomly sample num_negatives from candidates
                perm = torch.randperm(len(candidates))[:num_negatives]
                hard_negatives[global_i] = candidates[perm]
            elif len(candidates) > 0:
                # Use all candidates + fill remainder with random
                hard_negatives[global_i, :len(candidates)] = candidates
                fill = torch.randint(1, num_items, (num_negatives - len(candidates),))
                hard_negatives[global_i, len(candidates):] = fill
            else:
                # No valid Jaccard candidates — full random fallback
                hard_negatives[global_i] = torch.randint(1, num_items, (num_negatives,))

    # Row 0 (PAD) — fill with random (never used as an anchor, but safe)
    hard_negatives[0] = torch.randint(1, num_items, (num_negatives,))

    logger.info(f"  Hard-negative matrix computed: {hard_negatives.shape}")

    # ── Cache to disk ─────────────────────────────────────────
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        cache_payload = {
            "matrix": hard_negatives,
            "jaccard_low": jaccard_low,
            "jaccard_high": jaccard_high,
            "mining_mode": mining_mode,
        }
        torch.save(cache_payload, cache_path)
        logger.info(f"  Cached hard-negative matrix → {cache_path}")

    return hard_negatives

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # Smoke test — MultiObjectiveLoss
    B, H, Neg = 4, 128, 10
    config_mock = {
        "hero": {
            "contrastive": {"temperature": 0.07, "weight": 0.3},
            "discovery_weight": 0.5
        }
    }
    loss_fn = MultiObjectiveLoss(config_mock)
    
    logits = torch.randn(B, 100)
    targets = torch.randint(1, 100, (B,))
    anchor = torch.randn(B, H)
    pos = torch.randn(B, H)
    negs = torch.randn(B, Neg, H)
    pop_logits = torch.randn(100)
    
    loss_total, loss_ce, loss_cl, loss_disc = loss_fn(logits, targets, anchor, pos, negs, pop_logits)
    print("=== MultiObjectiveLoss Smoke Test ===")
    print(f"Total Loss: {loss_total.item():.4f}")
    print(f"CE Loss: {loss_ce.item():.4f}")
    print(f"CL Loss: {loss_cl.item():.4f}")
    print(f"Disc Loss: {loss_disc.item():.4f}")
    assert loss_total.item() > 0
    print("✓ multi-objective forward pass successful")

    # Smoke test — Attribute-aware hard-negative mining
    print("\n=== Hard-Negative Mining Smoke Test ===")
    NUM_ITEMS = 50
    NUM_NEG = 5
    mock_attrs = {
        i: {f"product_group_name={'A' if i % 3 == 0 else 'B'}",
            f"colour_group_name={'Red' if i % 2 == 0 else 'Blue'}",
            f"garment_group_name={'Dress' if i % 5 == 0 else 'Top'}"}
        for i in range(1, NUM_ITEMS)
    }

    # Test random mode
    hn_random = hard_negative_mining(
        num_items=NUM_ITEMS, num_negatives=NUM_NEG,
        item_attributes=mock_attrs, mining_mode="random",
    )
    assert hn_random.shape == (NUM_ITEMS, NUM_NEG)
    assert (hn_random >= 1).all() and (hn_random < NUM_ITEMS).all()
    print(f"  Random mode: shape={hn_random.shape} ✓")

    # Test attribute mode
    hn_attr = hard_negative_mining(
        num_items=NUM_ITEMS, num_negatives=NUM_NEG,
        item_attributes=mock_attrs, mining_mode="attribute",
        jaccard_low=0.3, jaccard_high=0.7,
    )
    assert hn_attr.shape == (NUM_ITEMS, NUM_NEG)
    assert (hn_attr >= 1).all() and (hn_attr < NUM_ITEMS).all()
    print(f"  Attribute mode: shape={hn_attr.shape} ✓")

    # Verify no self-negatives (item i should not appear in its own negatives)
    self_negs = sum(1 for i in range(1, NUM_ITEMS) if i in hn_attr[i].tolist())
    print(f"  Self-negatives found: {self_negs} (should be 0)")

    # Test empty attributes (should fall back to random)
    hn_empty = hard_negative_mining(
        num_items=NUM_ITEMS, num_negatives=NUM_NEG,
        item_attributes={}, mining_mode="attribute",
    )
    assert hn_empty.shape == (NUM_ITEMS, NUM_NEG)
    print(f"  Empty attrs fallback: shape={hn_empty.shape} ✓")

    print("✓ hard-negative mining smoke test passed")
