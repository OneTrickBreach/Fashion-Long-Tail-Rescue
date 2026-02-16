# Week 1 Plan — Ishan Biswas
## Phase 1: Data Engineering & Custom Baseline (Feb 15 – Feb 21)

> **Role:** Transformer Architect & Loss Tuning  
> **Branch:** `week1-ishan`  
> **Tools:** PyTorch, CUDA, NumPy, Pandas

---

## Overview

Week 1 establishes the foundation for the entire project.  My responsibilities fall into
two buckets:

1. **Local Environment** — confirm PyTorch + CUDA are functional and the full pipeline
   can train on the sampled dataset within local RAM/VRAM limits.
2. **The Villain Baseline** — build and evaluate a **Position-Aware Sequential Recommender
   (SASRec variant)** that is deliberately blind to visual features.  This becomes the
   "control" that the Hero must beat in Phase 2.

All hyperparameters and paths come from `config.yaml` (single source of truth per
`rules.md §2`).  All derived data is written to `data/sampled/` — raw files in
`h-and-m-personalized-fashion-recommendations/` are **never** modified (`rules.md §4`).

---

## Task Breakdown

### 1. Environment Verification  *(~0.5 day)*

| # | Sub-task | Details |
|---|----------|---------|
| 1a | Activate `.venv` and verify packages | Run `python -c "import torch; print(torch.cuda.is_available())"` — must print `True`. |
| 1b | Confirm `config.yaml` loads | Run `from src.utils.helpers import load_config; cfg = load_config()` and inspect. |
| 1c | Verify GPU memory | Run `torch.cuda.get_device_properties(0)` — note total VRAM for batch-size tuning. |
| 1d | Add any missing packages | e.g., `pyyaml`, `pandas`, etc.  Update `requirements.txt` after each install. |

**Deliverable:** A passing "smoke test" that imports all project modules and touches the GPU.

---

### 2. Data Sampling Pipeline  *(~1 day)*

> **Dependency:** Elizabeth's preprocessing  (may run in parallel since sampling can use raw CSVs directly).

| # | Sub-task | Details |
|---|----------|---------|
| 2a | Implement `src/data/sampler.py::create_sample()` | Read `transactions_train.csv` in chunks (`pd.read_csv(..., chunksize=500_000)`).  Apply 5 % stratified sample preserving head/torso/tail bins.  Filter users with < 5 interactions (`config.yaml → sampling.min_interactions`). |
| 2b | Write sampled CSVs | Save `transactions_sampled.csv`, `articles_sampled.csv`, `customers_sampled.csv` to `data/sampled/`. |
| 2c | Long-tail labeling | Add a `is_long_tail` boolean column to the articles sample (articles with < 10 total purchases).  This supports Nishant's Visibility Skew chart. |
| 2d | Distribution analysis | Print & log item-frequency distributions before/after sampling to verify the long-tail shape is preserved. |

**Key technical considerations:**
- `transactions_train.csv` is **3.5 GB** — must use chunked I/O, never `pd.read_csv()` on the whole file.
- Convert IDs to `int32` for RAM savings (per Elizabeth's plan, but we need this regardless).
- Temporal pruning: keep only the last 4–6 weeks of transactions (configurable in `config.yaml`).

**Deliverable:** `data/sampled/` populated with manageable CSVs (~5 % of original).

---

### 3. Dataset & DataLoader  *(~1 day)*

| # | Sub-task | Details |
|---|----------|---------|
| 3a | Implement `TransactionDataset` class in `src/data/dataset.py` | Each sample = one user's chronological purchase sequence (item IDs).  Pad/truncate to `max_seq_len = 50`.  Positional indices as a second tensor. |
| 3b | Train / Val / Test split | Temporal split: last purchase → test, second-to-last → val, rest → train.  This follows the leave-one-out evaluation protocol standard in sequential rec. |
| 3c | Implement `build_dataloaders()` | Returns `(train_loader, val_loader, test_loader)` using `torch.utils.data.DataLoader` with configurable `batch_size`. |
| 3d | Negative sampling | For each positive next-item, sample *N* negatives from the catalog.  Needed for pairwise loss (BPR) or cross-entropy. |

**Matrix shapes (input to Villain):**
```
item_seq:   (batch_size, max_seq_len)       — int64 item IDs
positions:  (batch_size, max_seq_len)       — int64 position indices [0 .. seq_len-1]
labels:     (batch_size,)                   — int64 next-item ground truth
```

**Deliverable:** Working DataLoaders that feed batches to the model.

---

### 4. Villain Model Architecture  *(~1.5 days)*

> This is the core deliverable for Week 1.

| # | Sub-task | Details |
|---|----------|---------|
| 4a | Item + positional embeddings | `nn.Embedding(num_items, hidden_dim)` + `nn.Embedding(max_seq_len, hidden_dim)`.  Sum them (SASRec style). |
| 4b | Transformer encoder stack | `nn.TransformerEncoder` with `num_layers=2`, `num_heads=2`, `hidden_dim=64` (from `config.yaml → villain`).  Causal mask so position *t* only attends to ≤ *t*. |
| 4c | Prediction head | Inner product of final hidden state with all item embeddings → `(batch_size, num_items)` logits.  Apply softmax for next-item probabilities. |
| 4d | Position-bias injection | The "ELO / Bubblesort" flavour: multiply logits by a learned position-bias vector so that items appearing in high-frequency positions get boosted.  This deliberately biases toward popular items, burying the long tail. |
| 4e | Loss function | Cross-entropy over the full catalog, or BPR (Bayesian Personalized Ranking) pairwise loss.  Start with CE, switch to BPR if CE is too slow over a large item vocabulary. |

**Architecture diagram:**
```
Input IDs ─► Item Embed ──┐
                          ├─ + ─► Transformer (×2 layers) ─► Final Hidden ─► dot(Item Embeds) ─► logits
Position  ─► Pos  Embed ──┘                                                      × pos_bias
```

**Deliverable:** `VillainModel` class that accepts `(item_seq, positions)` and returns `(batch_size, num_items)` scores.

---

### 5. Training Loop  *(~1 day)*

| # | Sub-task | Details |
|---|----------|---------|
| 5a | Implement `src/villain/trainer.py::train_villain()` | Standard PyTorch training loop: forward → loss → backward → optimizer step. |
| 5b | Optimizer & scheduler | Adam with `lr = 0.001`.  Optional: ReduceLROnPlateau or CosineAnnealing. |
| 5c | Validation & early stopping | Evaluate nDCG@12 on val set every epoch; stop if no improvement for 5 epochs. |
| 5d | Checkpoint saving | Save best model to `checkpoints/villain_best.pt`.  Git-ignore checkpoints (`rules.md §5`). |
| 5e | Logging | Use `src/utils/helpers.py::setup_logging()`.  Print epoch, loss, val metrics per epoch. |

**Deliverable:** End-to-end trainable Villain that converges on the sampled data.

---

### 6. Evaluation Hookup  *(~0.5 day)*

| # | Sub-task | Details |
|---|----------|---------|
| 6a | Wire up `src/utils/metrics.py` | Confirm nDCG@12 and MRR implementations are correct — Nishant owns these, but I need to call them in the Villain eval. |
| 6b | Full eval pass | After training, run the Villain on the test set and report nDCG@12, MRR, and Catalog Coverage. |
| 6c | Baseline results log | Write results to `outputs/villain_baseline_results.json` for comparison with the Hero later. |

**Deliverable:** Printed & saved evaluation scores for the Villain baseline.

---

### 7. Decision Log & Documentation  *(ongoing)*

| # | Sub-task | Details |
|---|----------|---------|
| 7a | Decision log | Document *why* we chose a position-biased baseline (per `plan.md`): it simulates real e-commerce popularity bias so the Hero has a clear "injustice" to fix. |
| 7b | Matrix shapes doc | Update `docs/matrix_shapes.md` with all tensor dimensions for the Villain. |
| 7c | Code docstrings | Every new `.py` file follows the format in `rules.md §3`: module docstring with purpose, team member, key classes/functions. |

---

## Dependencies on Teammates

| Teammate | What I need from them | Blocking? |
|----------|----------------------|-----------|
| **Elizabeth** | `int32` preprocessed transactions (temporal pruning, memory pivot) | **No** — I can sample from raw CSVs and re-sample later if her format changes. |
| **Nishant** | nDCG@12, MRR metric implementations in `src/utils/metrics.py` | **No** — I can write a quick placeholder and swap in his version. |

---

## Risk & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Full-catalog softmax is too slow (105k items) | Training won't converge in time | Use sampled softmax or BPR pairwise loss instead of full CE. |
| 3.5 GB CSV blows up RAM during sampling | Crashes / swap thrashing | Chunked reading with `chunksize=500_000`. |
| GPU VRAM insufficient for batch_size=256 | CUDA OOM | Reduce batch_size; use gradient accumulation. |
| Leave-one-out split gives too few train sequences for rare users | Poor long-tail recall | Enforce `min_interactions ≥ 5` during sampling. |

---

## Suggested Execution Order

```
Day 1 (Feb 16):  Tasks 1 (env) + 2 (sampling pipeline)
Day 2 (Feb 17):  Task 3 (dataset / dataloaders)
Day 3 (Feb 18):  Task 4 (Villain model architecture)
Day 4 (Feb 19):  Task 5 (training loop) — first training run overnight
Day 5 (Feb 20):  Task 6 (evaluate baseline) + Task 7 (docs & decision log)
Day 6 (Feb 21):  Buffer / bug fixes / coordination with Elizabeth & Nishant
```

---

## Success Criteria (End of Week 1)

- [x] `.venv` activates, PyTorch + CUDA confirmed working
- [x] `data/sampled/` contains ≤ 5 % subset with preserved long-tail distribution
- [ ] `VillainModel` trains to convergence on the sampled data
- [ ] Baseline nDCG@12, MRR, and Catalog Coverage are recorded
- [ ] Decision log explains the position-biased baseline choice
- [ ] All code follows `rules.md` (docstrings, config-driven, no hard-coded paths)

---

## Progress Journal

### Day 1 — Feb 16, 2026

#### ✅ Task 1: Environment Verification — DONE

| Component | Value |
|-----------|-------|
| PyTorch | 2.10+cu128 |
| CUDA | 12.8 |
| GPU | NVIDIA GeForce RTX 5070 Ti |
| VRAM | 11.9 GB |
| NumPy | 2.3.5 |
| Pandas | 3.0.0 |

All utility functions (`load_config`, `set_seed`, `get_device`) confirmed working.

#### ✅ Task 2: Data Sampling Pipeline — DONE

**What was built:** `src/data/sampler.py` — full stratified long-tail sampler with:
- 2-pass chunked reading of the 3.5 GB transactions CSV (chunksize=500k, no OOM)
- Temporal pruning to last 6 weeks (configurable via `config.yaml → sampling.temporal_weeks`)
- Stratified user sampling by popularity bin (head / torso / tail)
- Long-tail article labeling (`is_long_tail` column, threshold from config)
- `int32` compact dtypes for article IDs

**Config changes:** Added `temporal_weeks: 6` and `long_tail_threshold: 10` to `config.yaml → sampling`.

**Sampled dataset (`data/sampled/`):**

| File | Rows | Notes |
|------|------|-------|
| `transactions_sampled.csv` | 58,971 | 5,930 users · 12,559 articles |
| `articles_sampled.csv` | 12,559 | 88.3% flagged as long-tail |
| `customers_sampled.csv` | 5,930 | |

**Date range:** 2020-08-11 → 2020-09-22 (6-week window)

**Distribution stats (confirms long-tail preserved):**

| Metric | Value |
|--------|-------|
| Median purchases/item | 2 |
| Mean purchases/item | 4.7 |
| Max purchases/item | 121 |
| Head (80% of txns) | 4,599 articles (36.6%) |
| Torso (next 15%) | 5,011 articles (39.9%) |
| Tail (last 5%) | 2,949 articles (23.5%) |

**Next up (Day 2 — Feb 17):** Task 3 — Dataset class & DataLoaders.
