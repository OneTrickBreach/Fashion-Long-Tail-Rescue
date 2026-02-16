"""
model.py — The 'Villain': Position-Aware Sequential Baseline
==============================================================
Team: Ishan, Elizabeth, Nishant

PURPOSE:
    Implements a text-only sequential recommendation baseline that is
    intentionally BLIND to product images. This serves as our control model
    to quantify the value added by visual features in the Hero.

ARCHITECTURE OPTIONS:
    - SASRec: Self-Attentive Sequential Recommendation
      (Kang & McAuley, ICDM 2018)
    - ELO-ranking: A simpler position-aware scoring baseline

    The model encodes a user's purchase history as a sequence of item IDs
    (+ optional text features like product group, colour) and predicts
    the next item.

INPUTS:
    - User interaction sequences (item IDs, positional encodings)
    - Article metadata features (text-only: product_type, colour, section)

OUTPUTS:
    - Next-item probability distribution over the item catalog

MATRIX SHAPES (document in docs/matrix_shapes.md):
    - Input sequence:       (batch_size, max_seq_len)
    - Embedding lookup:     (batch_size, max_seq_len, hidden_dim)
    - Attention output:     (batch_size, max_seq_len, hidden_dim)
    - Final prediction:     (batch_size, num_items)
"""

import torch
import torch.nn as nn


class VillainModel(nn.Module):
    """
    Position-aware sequential recommender — text features only.
    """

    def __init__(self, config):
        super().__init__()
        # TODO: Implement model architecture
        raise NotImplementedError("TODO: Build Villain model")

    def forward(self, item_seq, positions):
        """
        Args:
            item_seq (Tensor):   (batch_size, max_seq_len) — item ID sequences
            positions (Tensor):  (batch_size, max_seq_len) — positional indices

        Returns:
            Tensor: (batch_size, num_items) — predicted scores
        """
        raise NotImplementedError("TODO: Implement forward pass")
