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


class VisualProjection(nn.Module):
    """
    Projects the high-dimensional visual/multimodal vectors (e.g., 2048-dim)
    down to the Transformer's hidden_dim.
    """
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, input_dim)
        x = self.proj(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class HeroModel(nn.Module):
    """
    Multimodal Behavior Sequence Transformer (BST)
    """

    def __init__(self, config: dict, num_items: int):
        super().__init__()
        self.hidden_dim = config["hero"].get("hidden_dim", 128)
        self.max_seq_len = config["hero"].get("max_seq_len", 16)
        self.use_visual = config["hero"].get("use_visual", True)
        self.visual_dim = config["embedding"].get("dim", 2048)
        
        # 1. ID & Positional Embeddings
        # PAD is at index 0, so padding_idx=0
        self.item_emb = nn.Embedding(num_items, self.hidden_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(self.max_seq_len, self.hidden_dim)
        self.emb_dropout = nn.Dropout(config["hero"].get("dropout", 0.1))

        # 2. Visual Projection (Task 4)
        if self.use_visual:
            self.visual_proj = VisualProjection(self.visual_dim, self.hidden_dim, config["hero"].get("dropout", 0.1))
            self.fusion_norm = nn.LayerNorm(self.hidden_dim)
            
        # 3. Transformer Encoder (Task 5b)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=config["hero"].get("num_heads", 4),
            dim_feedforward=self.hidden_dim * 4,
            dropout=config["hero"].get("dropout", 0.1),
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config["hero"].get("num_layers", 3),
        )

    def forward(self, item_seq: torch.Tensor, positions: torch.Tensor, visual_embeds: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            item_seq (Tensor):       (batch_size, max_seq_len) — item ID sequences
            positions (Tensor):      (batch_size, max_seq_len) — positional indices
            visual_embeds (Tensor):  (batch_size, max_seq_len, 2048) — ResNet50 features

        Returns:
            tuple: (logits, contrastive_embeds)
                - logits:             (batch_size, num_items) — prediction for the next item
                - contrastive_embeds: (batch_size, hidden_dim) — sequence representation for CL
        """
        batch_size, seq_len = item_seq.size()
        
        # 1. Base Embeddings
        x_item = self.item_emb(item_seq)  # (B, S, H)
        x_pos = self.pos_emb(positions)   # (B, S, H)
        
        seq_repr = x_item + x_pos
        
        # 2. Multimodal Fusion Layer (Task 4)
        if self.use_visual and visual_embeds is not None:
            # Project visual vector: (B, S, 2048) -> (B, S, H)
            v_proj = self.visual_proj(visual_embeds)
            
            # Fuse via element-wise addition + LayerNorm (SASRec style)
            seq_repr = seq_repr + v_proj
            seq_repr = self.fusion_norm(seq_repr)
            
        seq_repr = self.emb_dropout(seq_repr)

        # Causal Attention Mask (to prevent looking at future items in the sequence)
        # Bool upper-triangular mask: True = blocked position (consistent with Villain)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=item_seq.device),
            diagonal=1,
        )

        # Padding mask: True where item_seq == 0 (PAD_IDX)
        # size (B, S)
        key_padding_mask = (item_seq == 0)

        # 3. Transformer Forward
        # transformer expects (B, S, H) if batch_first=True
        # and causal mask as src_mask, padding mask as src_key_padding_mask
        encoded_seq = self.transformer(
            seq_repr,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
            is_causal=True,
        )

        # 4. Extract seq representation (we take the last item's representation)
        # However, for variable lengths we should take the last NON-PAD token.
        # But wait, our sequences are right-padded to max_seq_len, and the causal mask
        # is used. Typically in SASRec, we just use the final representation of the 
        # sequence at the target position. Since we padded at the END, the last 
        # non-pad token holds the sequence repr.
        
        # Find the length of each sequence
        seq_lens = (item_seq != 0).sum(dim=1) - 1 # 0-indexed last valid token
        # Handle case where sequence might be entirely padding (shouldn't happen, but safe)
        seq_lens = torch.clamp(seq_lens, min=0)
        
        # Gather the final valid state for each batch item
        # seq_lens: (B,) -> (B, 1, H) -> (B, H)
        batch_indices = torch.arange(batch_size, device=item_seq.device)
        final_states = encoded_seq[batch_indices, seq_lens, :] # (B, H)

        # 5. Prediction Head (Task 5c) — catalog dot-product
        all_item_embs = self.item_emb.weight # (num_items, H)
        logits = torch.matmul(final_states, all_item_embs.transpose(0, 1)) # (B, num_items)
        
        return logits, final_states


if __name__ == "__main__":
    # Smoke test
    config = {
        "hero": {"hidden_dim": 128, "max_seq_len": 16, "use_visual": True, "num_heads": 4, "num_layers": 3, "dropout": 0.1},
        "embedding": {"dim": 2048}
    }
    model = HeroModel(config, num_items=1000)
    
    # Dummy data
    B, S, D = 4, 16, 2048
    item_seq = torch.randint(1, 1000, (B, S))
    item_seq[0, 10:] = 0 # add some padding
    positions = torch.arange(S).unsqueeze(0).expand(B, S)
    visual_embeds = torch.randn(B, S, D)
    
    logits, contrastive_embeds = model(item_seq, positions, visual_embeds)
    
    print("=== HeroModel Smoke Test ===")
    print(f"Input item_seq: {item_seq.shape}")
    print(f"Input visual_embeds: {visual_embeds.shape}")
    print(f"Output logits: {logits.shape} (Expected: B, num_items)")
    print(f"Output contrastive_embeds: {contrastive_embeds.shape} (Expected: B, H)")
    print("✓ model forward pass successful")

