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

| Stage | Tensor | Shape | Notes |
|-------|--------|-------|-------|
| Text embedding | `text_emb` | `(B, S, D_hero)` | `(128, 50, 128)` — ID + Positional embeddings |
| Visual embedding | `vis_emb` | `(B, S, 2048)` | `(128, 50, 2048)` — From pre-computed ResNet50 |
| Projection layer | `vis_proj` | `(B, S, D_hero)` | `(128, 50, 128)` — Linear(2048, 128) |
| Fused input | `fused` | `(B, S, D_hero)` | `(128, 50, 128)` — text_emb + vis_proj + LayerNorm |
| BST encoder output | `bst_out` | `(B, S, D_hero)` | `(128, 50, 128)` — 3-layer, 4-head |
| Pooled representation | `user_repr` | `(B, D_hero)` | `(128, 128)` — Last valid hidden state |
| Prediction logits | `logits` | `(B, V)` | `(128, 26933)` — Dot product with item embeddings + visual embeddings |
| Contrastive anchor | `anchor` | `(B, D_hero)` | `(128, 128)` — User representation |
| Contrastive positive | `positive` | `(B, D_hero)` | `(128, 128)` — Target item embed + target visual embed |
| Contrastive negatives | `negatives` | `(N, D_hero)` | `(num_negatives, 128)` — Hard negatives drawn per epoch |

**Config:** `batch_size=256` · `max_seq_len=50` · `hidden_dim=128` · `num_heads=4` · `num_layers=3`

---

## Layer 1 — Input Fusion

| Tensor | Shape | Notes |
|--------|-------|-------|
| `text_emb` | `(128, 50, 128)` | ID embedding + positional embedding |
| `vis_raw` | `(128, 50, 2048)` | Raw ResNet50 output (Elizabeth's pipeline) |
| `vis_proj` | `(128, 50, 128)` | Linear(2048 → 128) projection of visual features |
| `fused` | `(128, 50, 128)` | `text_emb + vis_proj + LayerNorm` — output of Layer 1 |

---

## Layer 2 — Position + Time Encoding

| Tensor | Shape | Notes |
|--------|-------|-------|
| `fused` | `(128, 50, 128)` | Input from Layer 1 |
| `pos_embed` | `(128, 50, 128)` | Learnable position table `nn.Embedding(50, 128)` |
| `time_embed` | `(128, 50, 128)` | Time-gap encoding — days since purchase |
| `X` | `(128, 50, 128)` | `fused + pos_embed + time_embed` — input to transformer |

---

## Transformer Block ×3 (Layers 3–7)

> Layers 3–7 repeat `num_layers=3` times. Output of each block becomes input `X` for the next.

### Layer 3 — Linear Projections → Q, K, V

| Tensor | Shape | Notes |
|--------|-------|-------|
| `X` | `(128, 50, 128)` | Input sequence |
| `W_Q` | `(128, 128)` | Learned query weight matrix |
| `W_K` | `(128, 128)` | Learned key weight matrix |
| `W_V` | `(128, 128)` | Learned value weight matrix |
| `Q` | `(128, 4, 50, 32)` | `X @ W_Q` split into 4 heads · `head_dim = 128 ÷ 4 = 32` |
| `K` | `(128, 4, 50, 32)` | `X @ W_K` split into 4 heads |
| `V` | `(128, 4, 50, 32)` | `X @ W_V` split into 4 heads |

---

### Layer 4 — Attention Scores

| Tensor | Shape | Notes |
|--------|-------|-------|
| `Q` | `(128, 4, 50, 32)` | From Layer 3 |
| `Kᵀ` | `(128, 4, 32, 50)` | K transposed on last two dims |
| `scores` | `(128, 4, 50, 50)` | `Q @ Kᵀ / √32` · scale factor `√32 ≈ 5.66` |

> Causal mask applied — item `i` cannot attend to items after position `i`.

---

### Layer 5 — Softmax → Attention Weights

| Tensor | Shape | Notes |
|--------|-------|-------|
| `scores` | `(128, 4, 50, 50)` | Input from Layer 4 |
| `attn_weights` | `(128, 4, 50, 50)` | `softmax(scores, dim=-1)` · each row sums to 1.0 |

> `dropout=0.1` applied to attention weights during training.

---

### Layer 6 — Weighted Values

| Tensor | Shape | Notes |
|--------|-------|-------|
| `attn_weights` | `(128, 4, 50, 50)` | From Layer 5 |
| `V` | `(128, 4, 50, 32)` | From Layer 3 |
| `context` | `(128, 4, 50, 32)` | `attn_weights @ V` per head |
| `context` | `(128, 50, 128)` | Concat 4 heads · `4 × 32 = 128` |

---

### Layer 7 — Output Projection + Residual

| Tensor | Shape | Notes |
|--------|-------|-------|
| `context` | `(128, 50, 128)` | From Layer 6 |
| `W_O` | `(128, 128)` | Learned output projection matrix |
| `X` | `(128, 50, 128)` | Residual — same X that entered Layer 3 |
| `hidden_state` | `(128, 50, 128)` | `LayerNorm(context @ W_O + X)` — output of block |

> `hidden_state` becomes the new `X` for the next transformer block.

---

## Layer 8 — Extract Last Position

| Tensor | Shape | Notes |
|--------|-------|-------|
| `transformer_out` | `(128, 50, 128)` | Output of Block 3 (final transformer block) |
| `user_repr` | `(128, 128)` | `transformer_out[:, -1, :]` — last valid hidden state |

---

## Layer 9 — Dot With Item Bank

| Tensor | Shape | Notes |
|--------|-------|-------|
| `user_repr` | `(128, 128)` | From Layer 8 |
| `item_embeddings` | `(26933, 128)` | Full item bank — one 128-dim row per article |
| `logits` | `(128, 26933)` | `user_repr @ item_embeddings.T` — one score per item per user |

---

> **Last updated:** Nishant: March 3, 2026 — after Phase 2 hero run.

