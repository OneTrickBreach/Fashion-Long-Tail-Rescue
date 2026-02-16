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

import torch
import torch.nn as nn


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
        raise NotImplementedError("TODO: Implement contrastive loss")


def hard_negative_mining(embeddings, attributes, num_negatives=10):
    """
    Select hard negatives based on attribute overlap.

    Args:
        embeddings (Tensor):  (num_items, hidden_dim)
        attributes (dict):    item_id → set of attribute values
        num_negatives (int):  number of negatives to mine per anchor

    Returns:
        Tensor: indices of hard negatives
    """
    raise NotImplementedError("TODO: Implement hard negative mining")
