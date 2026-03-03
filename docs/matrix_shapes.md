# Matrix Shape Transformations
### CS7180 Rubric Requirement — Seeing the Unseen
**Team: Ishan, Elizabeth, Nishant**

---

> This document tracks every tensor shape transformation through the pipeline,
> from raw data to final predictions. Required by the CS7180 rubric.
>
> **Values below are actual** from the beefed-up training run (Feb 18, 2026).

---

## Notation

| Symbol | Value | Source |
|--------|-------|--------|
| `B` | 256 | `config.yaml → villain.batch_size` |
| `S` | 50 | `config.yaml → villain.max_seq_len` |
| `D` | 128 | `config.yaml → villain.hidden_dim` |
| `H` | 4 | `config.yaml → villain.num_heads` |
| `L` | 3 | `config.yaml → villain.num_layers` |
| `V` | 26,933 | 26,932 articles + 1 PAD token (index 0) |

---

## 1. Data Pipeline

| Stage | Tensor / Variable | Shape | Notes |
|-------|-------------------|-------|-------|
| Raw transactions | `transactions_df` | `(~31M, 5)` | date, customer, article, price, channel |
| Temporal pruning | `pruned_df` | `(~8.4M, 5)` | Last 12 weeks of transactions |
| Sampled transactions | `sampled_df` | `(430,879, 5)` | 15% stratified user sample |
| User history (raw) | `user_seq[uid]` | `(variable,)` | 5–100+ article IDs per user |
| Padded input | `item_seq` | `(B, S)` = `(256, 50)` | Right-padded with 0 (PAD) |
| Position indices | `positions` | `(B, S)` = `(256, 50)` | `[0, 1, 2, ..., 49]` per sample |
| Sequence lengths | `seq_len` | `(B,)` = `(256,)` | Actual unpadded length |
| Target (next item) | `target` | `(B,)` = `(256,)` | Ground-truth item index |

---

## 2. Visual Embedding Extraction *(Hero)*

| Stage | Tensor | Shape | Notes |
|-------|--------|-------|-------|
| Raw images | `img_batch` | `(64, 3, 224, 224)` | RGB, batch_size=64 |
| ResNet50 conv output | `features` | `(64, 2048, 7, 7)` | Before pooling |
| Global avg pooling | `embeddings` | `(64, 2048)` | Per-image vector |
| Stored embeddings | `article_embeddings` | `(V, 2048)` | Pre-computed, .npy on disk |

---

## 3. Villain Model (SASRec — Text-Only Baseline)

### Forward Pass

```
item_seq (B, S) ──► nn.Embedding(V, D) ──► item_emb (B, S, D)
                                                    │
positions (B, S) ──► nn.Embedding(S, D) ──► pos_emb (B, S, D)
                                                    │
                                              ┌─────┤
                                              ▼     ▼
                                         add + LayerNorm ──► x (B, S, D)
                                              │
                                              ▼
                                         Dropout(0.2)
                                              │
                                              ▼
                              ┌── TransformerEncoder (×3 layers) ──┐
                              │    MultiHeadAttention(D, H=4)      │
                              │    causal_mask: (S, S) float       │
                              │    pad_mask:    (B, S) bool        │
                              └────────────────────────────────────┘
                                              │
                                              ▼
                                     x_out (B, S, D)
                                              │
                                    gather(seq_len - 1)
                                              │
                                              ▼
                                    last_hidden (B, D)
                                              │
                                         LayerNorm
                                              │
                                              ▼
                              last_hidden @ item_embed.weight.T
                                              │
                                              ▼
                                     logits (B, V)
                                              │
                                     × pop_bias (V,)
                                              │
                                              ▼
                                  final_logits (B, V) = (256, 26933)
```

### Shape Summary Table

| Stage | Tensor | Shape | Actual |
|-------|--------|-------|--------|
| Input item IDs | `item_seq` | `(B, S)` | `(256, 50)` |
| Item embedding lookup | `item_emb` | `(B, S, D)` | `(256, 50, 128)` |
| Position embedding | `pos_emb` | `(B, S, D)` | `(256, 50, 128)` |
| Combined + LayerNorm | `x` | `(B, S, D)` | `(256, 50, 128)` |
| Causal attention mask | `causal_mask` | `(S, S)` | `(50, 50)` |
| Padding mask | `pad_mask` | `(B, S)` | `(256, 50)` |
| Transformer output | `x_out` | `(B, S, D)` | `(256, 50, 128)` |
| Last valid hidden | `last_hidden` | `(B, D)` | `(256, 128)` |
| Raw logits | `logits` | `(B, V)` | `(256, 26933)` |
| Pop-bias vector | `pop_bias` | `(V,)` | `(26933,)` |
| Final logits | `logits` | `(B, V)` | `(256, 26933)` |
| CE loss target | `target` | `(B,)` | `(256,)` |

### Parameter Count

| Component | Parameters |
|-----------|-----------|
| `item_embedding` | V × D = 26,933 × 128 = **3,447,424** |
| `position_embedding` | S × D = 50 × 128 = **6,400** |
| `embed_norm` (LayerNorm) | 2 × D = **256** |
| `TransformerEncoder` (×3) | 3 × (4D² + 4D + 4D² + 4D + 4D) ≈ **396,672** |
| `output_norm` (LayerNorm) | 2 × D = **256** |
| `pop_bias` | V = **26,933** |
| **Total** | | **4,076,085** |

---

## 4. Hero Model (Multimodal BST + Contrastive)

> **Note:** Hero uses `B=128` (config.yaml → hero.batch_size), not 256 like the Villain.

| Symbol | Value | Source |
|--------|-------|--------|
| `B` | 128 | `config.yaml → hero.batch_size` |
| `S` | 50 | `config.yaml → hero.max_seq_len` |
| `D` | 128 | `config.yaml → hero.hidden_dim` |
| `H` | 4 | `config.yaml → hero.num_heads` |
| `L` | 3 | `config.yaml → hero.num_layers` |
| `V` | 26,933 | Same vocabulary as Villain (26,932 articles + PAD) |
| `E` | 2048 | `config.yaml → embedding.dim` (ResNet50 output) |

### Forward Pass

```
item_seq (B, S) ──► nn.Embedding(V, D) ──► item_emb (B, S, D)
                                                  │
positions (B, S) ──► nn.Embedding(S, D) ──► pos_emb (B, S, D)
                                                  │
                                            ┌─────┤
                                            ▼     ▼
                                        add ──► seq_repr (B, S, D)
                                            │
     ┌──────────────────────────────────────┤ use_visual=True?
     │                                      │
     ▼                                      │ (skip if no visual)
visual_embeds (B, S, E)                     │
     │                                      │
     ▼ VisualProjection                     │
  Linear(E→D) ──► (B, S, D)                │
  LayerNorm(D)                              │
  Dropout(0.1)                              │
     │                                      │
     ▼── v_proj (B, S, D)                   │
     │                                      │
     ▼                                      ▼
   seq_repr = seq_repr + v_proj ──► fusion_norm(LayerNorm)
                                            │
                                            ▼
                                      fused (B, S, D)
                                            │
                                       Dropout(0.1)
                                            │
                                            ▼
                            ┌── TransformerEncoder (×3 layers) ──┐
                            │    MultiHeadAttention(D, H=4)      │
                            │    FFN: D → 4D → D (GELU)         │
                            │    causal_mask: (S, S) bool        │
                            │    pad_mask:    (B, S) bool        │
                            └────────────────────────────────────┘
                                            │
                                            ▼
                                   encoded_seq (B, S, D)
                                            │
                                  gather(seq_lens = last non-PAD)
                                            │
                                            ▼
                                  final_states (B, D) ──────────────┐
                                            │                       │
                               final_states @ item_emb.weight.T     │
                                            │                       │
                                            ▼                       │
                                   logits (B, V)                    │
                                                                    │
                              ┌─────────────────────────────────────┘
                              │   Contrastive Learning Head
                              │
                              ▼
                        anchor = final_states                (B, D)
                        positive = item_emb(target)          (B, D)
                        negatives = item_emb(hard_neg_idx)   (B, N_neg, D)
                              │
                              ▼
                      InfoNCE Loss (temperature=0.07)
                              │
                              ▼
                L_total = L_CE(logits, target) + 0.3 × L_InfoNCE
```

### Shape Summary Table

| Stage | Tensor | Shape | Actual |
|-------|--------|-------|--------|
| Input item IDs | `item_seq` | `(B, S)` | `(128, 50)` |
| Item embedding lookup | `item_emb` | `(B, S, D)` | `(128, 50, 128)` |
| Position embedding | `pos_emb` | `(B, S, D)` | `(128, 50, 128)` |
| Pre-fusion sum | `seq_repr` | `(B, S, D)` | `(128, 50, 128)` |
| Visual input | `visual_embeds` | `(B, S, E)` | `(128, 50, 2048)` |
| Visual projection | `v_proj` | `(B, S, D)` | `(128, 50, 128)` |
| Fused + LayerNorm | `fused` | `(B, S, D)` | `(128, 50, 128)` |
| Causal attention mask | `causal_mask` | `(S, S)` | `(50, 50)` |
| Padding mask | `pad_mask` | `(B, S)` | `(128, 50)` |
| Transformer output | `encoded_seq` | `(B, S, D)` | `(128, 50, 128)` |
| Last valid hidden | `final_states` | `(B, D)` | `(128, 128)` |
| Prediction logits | `logits` | `(B, V)` | `(128, 26933)` |
| Contrastive anchor | `anchor` | `(B, D)` | `(128, 128)` |
| Contrastive positive | `positive` | `(B, D)` | `(128, 128)` |
| Contrastive negatives | `negatives` | `(B, N, D)` | `(128, 10, 128)` |
| CE loss target | `target` | `(B,)` | `(128,)` |

### Parameter Count

| Component | Parameters |
|-----------|-----------|
| `item_embedding` | V × D = 26,933 × 128 = **3,447,424** |
| `position_embedding` | S × D = 50 × 128 = **6,400** |
| `emb_dropout` | 0 (no learnable params) |
| `VisualProjection.proj` | E × D + D = 2048 × 128 + 128 = **262,272** |
| `VisualProjection.norm` | 2 × D = **256** |
| `fusion_norm` (LayerNorm) | 2 × D = **256** |
| `TransformerEncoder` (×3) | 3 × (4D² + 4D + 4D² + 4D + 4D) ≈ **396,672** |
| **Total** | ≈ **4,113,280** |

---

> **Last updated:** Mar 3, 2026 — Hero forward pass diagram added; Hero batch size corrected to 128.
