"""
model.py — The 'Villain': Position-Aware Sequential Baseline (SASRec)
======================================================================
Team member: Ishan Biswas
Key classes: VillainModel

PURPOSE:
    Implements a text-only sequential recommendation baseline that is
    intentionally BLIND to product images. This serves as our control model
    to quantify the value added by visual features in the Hero.

ARCHITECTURE:
    SASRec (Self-Attentive Sequential Recommendation, Kang & McAuley 2018)
    with an added POSITION-BIAS vector that deliberately boosts popular
    items and buries the long tail — the "injustice" the Hero must fix.

    Input IDs ─► Item Embed ──┐
                              ├─ + ─► Transformer Encoder (×N) ─► last hidden
    Position  ─► Pos  Embed ──┘                                       │
                                                                      ▼
                                                          dot(all item embeds)
                                                                      │
                                                                 × pop_bias
                                                                      │
                                                                   logits

MATRIX SHAPES:
    - Input item_seq:       (B, S)            — B=batch, S=max_seq_len
    - Embedding lookup:     (B, S, D)         — D=hidden_dim
    - Attention output:     (B, S, D)
    - Final hidden:         (B, D)            — last valid position
    - Prediction logits:    (B, num_items)

INPUTS:
    - item_seq  (LongTensor): (B, S) — padded item-index sequences
    - positions (LongTensor): (B, S) — positional indices [0..S-1]

OUTPUTS:
    - logits    (FloatTensor): (B, num_items) — unnormalized scores
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class VillainModel(nn.Module):
    """
    Position-aware SASRec sequential recommender — text features only.

    The model encodes a user's purchase history as a sequence of item IDs
    and predicts the next item. A learnable popularity-bias vector is
    multiplied into the final logits to simulate real-world e-commerce
    position/popularity bias (the Villain's deliberate unfairness).
    """

    def __init__(self, num_items: int, config: dict):
        """
        Args:
            num_items: Total vocabulary size (including PAD at index 0).
            config:    Villain config dict (from get_villain_config).
        """
        super().__init__()
        self.num_items = num_items
        self.hidden_dim = config["hidden_dim"]
        self.max_seq_len = config["max_seq_len"]

        # ── Embeddings ───────────────────────────────────────
        self.item_embedding = nn.Embedding(
            num_items, self.hidden_dim, padding_idx=0,
        )
        self.position_embedding = nn.Embedding(
            self.max_seq_len, self.hidden_dim,
        )
        self.embed_dropout = nn.Dropout(config["dropout"])
        self.embed_norm = nn.LayerNorm(self.hidden_dim)

        # ── Transformer encoder ──────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=config["num_heads"],
            dim_feedforward=self.hidden_dim * 4,
            dropout=config["dropout"],
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config["num_layers"],
        )

        # ── Prediction head ──────────────────────────────────
        self.output_norm = nn.LayerNorm(self.hidden_dim)

        # ── Position/popularity bias (the Villain's "unfairness") ──
        # One learnable scalar per item, initialized to 1.0 (neutral).
        # During training the model will learn to boost popular items.
        self.pop_bias = nn.Parameter(torch.ones(num_items))

        self._init_weights()

    # ── Weight initialization ────────────────────────────────

    def _init_weights(self):
        """Xavier-uniform for embeddings, default init for Transformer."""
        nn.init.xavier_uniform_(self.item_embedding.weight[1:])  # skip PAD
        nn.init.xavier_uniform_(self.position_embedding.weight)

    # ── Causal mask ──────────────────────────────────────────

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Upper-triangular causal mask so position t only attends to ≤ t.

        Returns:
            (S, S) float mask with -inf for masked positions.
        """
        return nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=device,
        )

    # ── Padding mask ─────────────────────────────────────────

    @staticmethod
    def _padding_mask(item_seq: torch.Tensor) -> torch.Tensor:
        """
        Boolean mask: True where PAD (index 0).

        Args:
            item_seq: (B, S) int tensor.

        Returns:
            (B, S) bool tensor.
        """
        return item_seq == 0

    # ── Forward ──────────────────────────────────────────────

    def forward(
        self,
        item_seq: torch.Tensor,
        positions: torch.Tensor,
        seq_len: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            item_seq  (LongTensor): (B, S) — padded item indices.
            positions (LongTensor): (B, S) — position indices.
            seq_len   (LongTensor): (B,)   — actual sequence lengths (optional).

        Returns:
            logits (FloatTensor): (B, num_items) — prediction scores.
        """
        B, S = item_seq.shape

        # ── 1. Embedding: item + position ────────────────────
        item_emb = self.item_embedding(item_seq)       # (B, S, D)
        pos_emb = self.position_embedding(positions)    # (B, S, D)
        x = self.embed_norm(item_emb + pos_emb)        # (B, S, D)
        x = self.embed_dropout(x)

        # ── 2. Transformer with causal + padding masks ───────
        causal_mask = self._causal_mask(S, item_seq.device)
        pad_mask = self._padding_mask(item_seq)

        x = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=pad_mask,
        )  # (B, S, D)

        # ── 3. Extract last valid hidden state ───────────────
        if seq_len is not None:
            # Gather the hidden state at position (seq_len - 1)
            idx = (seq_len - 1).clamp(min=0).unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
            idx = idx.expand(-1, -1, self.hidden_dim)                     # (B, 1, D)
            last_hidden = x.gather(1, idx).squeeze(1)                     # (B, D)
        else:
            last_hidden = x[:, -1, :]  # (B, D)

        last_hidden = self.output_norm(last_hidden)

        # ── 4. Dot product with all item embeddings ──────────
        # item_embedding.weight: (num_items, D)
        logits = last_hidden @ self.item_embedding.weight.T  # (B, num_items)

        # ── 5. Multiply by popularity bias (the Villain's trick) ─
        logits = logits * self.pop_bias.unsqueeze(0)  # (B, num_items)

        return logits

    def predict_top_k(
        self,
        item_seq: torch.Tensor,
        positions: torch.Tensor,
        seq_len: torch.Tensor | None = None,
        k: int = 12,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience method for inference: returns top-K item indices and scores.

        Returns:
            (top_scores, top_indices) — both (B, K).
        """
        logits = self.forward(item_seq, positions, seq_len)
        # Zero out the PAD position so it's never recommended
        logits[:, 0] = float("-inf")
        top_scores, top_indices = logits.topk(k, dim=-1)
        return top_scores, top_indices
