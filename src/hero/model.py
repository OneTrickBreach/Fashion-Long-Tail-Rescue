"""
model.py — The 'Hero': Multimodal Behavior Sequence Transformer
=================================================================
Team: Ishan, Elizabeth, Nishant

PURPOSE:
    Implements our main contribution — a Behavior Sequence Transformer (BST)
    that fuses textual article features with pre-computed ResNet50 visual
    embeddings, coupled with an Attribute-Aware Contrastive Learning head.

    This model is designed to rescue the long tail by learning richer item
    representations that capture visual similarity patterns invisible to
    text-only baselines.

ARCHITECTURE:
    ┌──────────────────────────────────────────────┐
    │  User Behavior Sequence                      │
    │  [item_1, item_2, ..., item_T]               │
    │       │          │              │             │
    │  ┌────▼───┐ ┌────▼───┐    ┌────▼───┐        │
    │  │Text Emb│ │Text Emb│    │Text Emb│        │
    │  │+Vis Emb│ │+Vis Emb│    │+Vis Emb│        │
    │  └────┬───┘ └────┬───┘    └────┬───┘        │
    │       │          │              │             │
    │  ┌────▼──────────▼──────────────▼───┐        │
    │  │  Transformer Encoder (BST)       │        │
    │  │  (Multi-head Self-Attention)      │        │
    │  └──────────────┬───────────────────┘        │
    │                 │                             │
    │       ┌─────────▼─────────┐                  │
    │       │  Prediction Head  │                  │
    │       │  + Contrastive CL │                  │
    │       └───────────────────┘                  │
    └──────────────────────────────────────────────┘

MATRIX SHAPES (document in docs/matrix_shapes.md):
    - Text embedding:       (batch_size, max_seq_len, text_dim)
    - Visual embedding:     (batch_size, max_seq_len, 2048)
    - Fused embedding:      (batch_size, max_seq_len, hidden_dim)
    - Transformer output:   (batch_size, max_seq_len, hidden_dim)
    - Contrastive anchors:  (batch_size, hidden_dim)
    - Final prediction:     (batch_size, num_items)
"""

import torch
import torch.nn as nn


class HeroModel(nn.Module):
    """
    Multimodal BST with Attribute-Aware Contrastive Learning head.
    """

    def __init__(self, config):
        super().__init__()
        # TODO: Implement model architecture
        raise NotImplementedError("TODO: Build Hero model")

    def forward(self, item_seq, positions, visual_embeds):
        """
        Args:
            item_seq (Tensor):       (batch_size, max_seq_len) — item ID sequences
            positions (Tensor):      (batch_size, max_seq_len) — positional indices
            visual_embeds (Tensor):  (batch_size, max_seq_len, 2048) — ResNet50 features

        Returns:
            tuple: (predictions, contrastive_embeds)
                - predictions:        (batch_size, num_items)
                - contrastive_embeds: (batch_size, hidden_dim)
        """
        raise NotImplementedError("TODO: Implement forward pass")
