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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def hard_negative_mining(embeddings, attributes, num_negatives=10):
    """
    Select hard negatives based on attribute overlap.

    Args:
        embeddings (Tensor):  (num_items, hidden_dim)
        attributes (dict):    item_id → set of attribute values (or list)
        num_negatives (int):  number of negatives to mine per anchor

    Returns:
        Tensor: indices of hard negatives. Shape (num_items, num_negatives)
    """
    num_items = embeddings.size(0)

    # Vectorised random negatives — avoids O(num_items) Python loop.
    # Samples uniformly from [1, num_items) to exclude PAD index 0.
    # Full attribute-based Jaccard mining is deferred until Phase 3.
    hard_negatives = torch.randint(1, num_items, size=(num_items, num_negatives))

    return hard_negatives

if __name__ == "__main__":
    # Smoke test
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
