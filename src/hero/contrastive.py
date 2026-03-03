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

KEY COMPONENTS (to implement):
    - ContrastiveLoss:      InfoNCE / NT-Xent style loss with temperature
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


class CombinedLoss(nn.Module):
    """
    Combined Loss for HeroModel:
    L_total = L_CE(predictions, targets) + lambda * L_contrastive(anchor, pos, negs)
    """
    def __init__(self, config: dict):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=0) # PAD index is 0
        
        cl_config = config.get("hero", {}).get("contrastive", {})
        temperature = cl_config.get("temperature", 0.07)
        self.cl_weight = cl_config.get("weight", 0.3)
        
        self.contrastive_loss = ContrastiveLoss(temperature=temperature)

    def forward(self, logits, targets, anchor, positive, negatives):
        """
        Computes the combined loss.
        """
        loss_ce = self.ce_loss(logits, targets)
        loss_cl = self.contrastive_loss(anchor, positive, negatives)
        return loss_ce + self.cl_weight * loss_cl


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
    loss_fn = ContrastiveLoss(temperature=0.07)
    anchor = torch.randn(B, H)
    pos = torch.randn(B, H)
    negs = torch.randn(B, Neg, H)
    
    loss = loss_fn(anchor, pos, negs)
    print("=== ContrastiveLoss Smoke Test ===")
    print(f"Loss value: {loss.item():.4f}")
    assert loss.item() > 0
    print("✓ contrastive forward pass successful")
