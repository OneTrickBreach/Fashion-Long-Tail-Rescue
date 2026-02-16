# Matrix Shape Transformations
### CS7180 Rubric Requirement — Seeing the Unseen
**Team: Ishan, Elizabeth, Nishant**

---

> This document tracks every tensor shape transformation through the pipeline,
> from raw data to final predictions. Required by the CS7180 rubric.

---

## 1. Data Pipeline

| Stage                  | Tensor / Variable       | Shape                                   |
|------------------------|-------------------------|-----------------------------------------|
| Raw transactions       | `transactions_df`       | `(~31M, 5)` — columns: date, customer, article, price, channel |
| Sampled transactions   | `sampled_df`            | `(~1.5M, 5)` — after 5% stratified sampling |
| User history sequences | `item_seq`              | `(batch_size, max_seq_len)` — padded item IDs |
| Positional indices     | `positions`             | `(batch_size, max_seq_len)` — 0..T-1 |

## 2. Visual Embedding Extraction

| Stage                  | Tensor                  | Shape                                   |
|------------------------|-------------------------|-----------------------------------------|
| Raw images             | `img_batch`             | `(batch_size, 3, 224, 224)` — RGB |
| ResNet50 conv output   | `features`              | `(batch_size, 2048, 7, 7)` |
| Global avg pooling     | `embeddings`            | `(batch_size, 2048)` |
| Stored embeddings      | `article_embeddings`    | `(num_articles, 2048)` — .npy on disk |

## 3. Villain Model (Text-Only Baseline)

| Stage                  | Tensor                  | Shape                                   |
|------------------------|-------------------------|-----------------------------------------|
| Item ID embedding      | `item_emb`              | `(batch_size, max_seq_len, hidden_dim)` |
| Positional encoding    | `pos_emb`               | `(batch_size, max_seq_len, hidden_dim)` |
| Combined input         | `x`                     | `(batch_size, max_seq_len, hidden_dim)` |
| Self-attention output  | `attn_out`              | `(batch_size, max_seq_len, hidden_dim)` |
| Final MLP              | `logits`                | `(batch_size, num_items)` |

## 4. Hero Model (Multimodal BST + Contrastive)

| Stage                  | Tensor                  | Shape                                   |
|------------------------|-------------------------|-----------------------------------------|
| Text embedding         | `text_emb`              | `(batch_size, max_seq_len, text_dim)` |
| Visual embedding       | `vis_emb`               | `(batch_size, max_seq_len, 2048)` |
| Projection layer       | `vis_proj`              | `(batch_size, max_seq_len, hidden_dim)` |
| Fused input            | `fused`                 | `(batch_size, max_seq_len, hidden_dim)` |
| BST encoder output     | `bst_out`               | `(batch_size, max_seq_len, hidden_dim)` |
| Pooled representation  | `user_repr`             | `(batch_size, hidden_dim)` |
| Prediction head        | `logits`                | `(batch_size, num_items)` |
| Contrastive anchor     | `anchor`                | `(batch_size, hidden_dim)` |
| Contrastive positive   | `positive`              | `(batch_size, hidden_dim)` |
| Contrastive negatives  | `negatives`             | `(batch_size, num_neg, hidden_dim)` |

---

> **⚠️ Note:** All shapes above use placeholder dimension names. Update with
> actual values once hyperparameters are finalized in `config.yaml`.
