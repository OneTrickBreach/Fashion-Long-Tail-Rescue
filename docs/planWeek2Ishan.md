# Week 2 Plan — Ishan Biswas
## Phase 2: The "Hero" Model – Style & Sequential Intent (Feb 22 – Feb 28)

> **Role:** Transformer Architect & Loss Tuning + Multimodal Feature Bank (covering for Elizabeth)  
> **Branch:** `week2-ishan`  
> **Tools:** PyTorch, CUDA, NumPy, Pandas, torchvision (ResNet50)

---

## Overview

Week 2 builds the core "Hero" model that rescues the long tail.  Because Elizabeth is
absent, I am also covering her Phase 2 responsibilities (ResNet50 feature extraction
and multimodal metadata fusion).  My responsibilities fall into three buckets:

1. **Multimodal Feature Bank (Elizabeth's portion)** — pass all ~105,000 H&M product
   images through a frozen **ResNet50** to extract `(num_articles, 2048)` visual
   embeddings, then merge those vectors with article metadata (color, fabric, etc.)
   from `articles.csv`.
2. **Behavior Sequence Transformer (BST)** — implement the multimodal sequential
   recommender that fuses item-ID embeddings with the visual embeddings from step 1.
   Input shape is `(batch_size, max_seq_len, hidden_dim)` where `hidden_dim` is the
   fused visual/ID embedding dimension.
3. **Cold Start Simulation** — Steve's requirement: simulate a "New Product"
   (e.g., a "Kirkland" version) that has zero interaction history and prove that the
   Hero ranks it meaningfully using only its image vector, whereas the Villain cannot
   rank it at all.

All hyperparameters and paths come from `config.yaml` (single source of truth per
`rules.md §2`).  The hero section already exists with preliminary settings
(`hidden_dim=128`, `num_heads=4`, `num_layers=3`, `lr=0.0005`, `batch_size=128`).
The embedding section has `backbone: resnet50`, `dim: 2048`, `batch_size: 64`.

---

## Task Breakdown

### 1. ResNet50 Feature Extraction  *(~1 day)*  — *Elizabeth's portion*

> **Originally assigned to Elizabeth.**  I am covering this because she is absent.

| # | Sub-task | Details |
|---|----------|---------|
| 1a | Locate H&M product images | The raw images live under `h-and-m-personalized-fashion-recommendations/images/`.  Verify the directory structure and total image count (~105k JPEGs organised in subfolders). |
| 1b | Build extraction script | Create `src/data/extract_visual_embeddings.py`.  Load a **frozen** `torchvision.models.resnet50(pretrained=True)`, strip the final classification head, and use the 2048-dim average-pool output as the visual embedding. |
| 1c | Batch extraction on GPU | Process all images in batches of 64 (`config.yaml → embedding.batch_size`).  Use `torchvision.transforms` (resize 224×224, normalize with ImageNet mean/std).  Save output to `data/embeddings/resnet50_embeddings.pt` — a dict mapping `article_id → (2048,)` tensor. |
| 1d | Handle missing images | Some article IDs in `articles.csv` may lack images.  For these, store a zero vector `(2048,)` and log the missing IDs. |
| 1e | Verify embeddings | Load the saved `.pt` file, spot-check 10 random articles, confirm shape `(2048,)` and non-zero values. |

**Key technical considerations:**
- ~105k images × 2048 floats × 4 bytes ≈ 820 MB on disk.  Fits comfortably in RAM.
- Use `torch.no_grad()` and `model.eval()` — no gradients needed during extraction.
- RTX 5070 Ti (11.9 GB VRAM) can handle batch_size=64 of 224×224 images easily.
- Save as a single `.pt` file for fast loading during training.

**Deliverable:** `data/embeddings/resnet50_embeddings.pt` — a complete visual feature bank.

---

### 2. Multimodal Metadata Fusion  *(~0.5 day)*  — *Elizabeth's portion*

> **Originally assigned to Elizabeth.**  Merge ResNet50 vectors with article metadata.

| # | Sub-task | Details |
|---|----------|---------|
| 2a | Parse article metadata | Read `articles.csv` columns: `colour_group_name`, `perceived_colour_value_name`, `product_type_name`, `product_group_name`, `graphical_appearance_name`, `garment_group_name`, etc. |
| 2b | Encode categorical metadata | Use label encoding or small learned embeddings for each metadata field.  Concatenate into a metadata vector per article. |
| 2c | Fuse visual + metadata | Concatenate the ResNet50 `(2048,)` vector with the encoded metadata vector, then project through a linear layer to produce a final `(dim_fused,)` multimodal feature per article.  Save to `data/embeddings/multimodal_embeddings.pt`. |
| 2d | Alignment with ID maps | Ensure the multimodal embedding matrix is indexed consistently with `dataset.py::build_id_maps()`.  PAD token (index 0) maps to a zero vector. |

**Deliverable:** `data/embeddings/multimodal_embeddings.pt` — fused visual + metadata feature bank.

---

### 3. Visual Embedding Integration into DataLoaders  *(~0.5 day)*

| # | Sub-task | Details |
|---|----------|---------|
| 3a | Build `load_visual_embeddings()` utility | Function in `src/data/` that loads the embedding bank (`.pt` file) and maps article IDs to their visual vectors.  Returns a `(num_articles, 2048)` tensor aligned with the article ID map from `dataset.py::build_id_maps()`. |
| 3b | Extend `TransactionDataset` | Modify `src/data/dataset.py` to optionally include visual embeddings per item in each batch.  Each sample now returns `(item_seq, positions, visual_embeds, label)`. |
| 3c | Verify batch shapes | Smoke test: feed one batch through the extended DataLoader and confirm shapes: `item_seq: (B, S)`, `positions: (B, S)`, `visual_embeds: (B, S, 2048)`, `label: (B,)`. |

**Key technical considerations:**
- Keep the full embedding matrix on CPU and index into it per-batch to save GPU memory.
- PAD token (index 0) should map to a zero vector in the visual embedding bank.
- Pre-filter to sampled articles only if memory is tight.

**Deliverable:** DataLoaders that produce multimodal batches (IDs + visual vectors).

---

### 4. Multimodal Fusion Layer  *(~0.5 day)*

| # | Sub-task | Details |
|---|----------|---------|
| 2a | Design fusion strategy | Implement a projection + concatenation / gating approach.  Project the 2048-dim visual vector down to `hidden_dim` via a linear layer, then fuse with the item-ID embedding. |
| 2b | Implement `VisualProjection` module | `nn.Linear(2048, hidden_dim)` + LayerNorm + Dropout.  Placed inside `HeroModel`. |
| 2c | Fusion operation | Sum or concatenate the projected visual embedding with the item-ID + position embedding.  Start with element-wise addition (SASRec style). |
| 2d | Ablation toggle | Add a `config.yaml → hero.use_visual` boolean.  When `False`, the model should behave identically to the Villain (ID-only), enabling ablation comparison later. |

**Matrix shapes after fusion:**
```
item_embed:     (batch_size, max_seq_len, hidden_dim)   — from nn.Embedding
pos_embed:      (batch_size, max_seq_len, hidden_dim)   — from nn.Embedding
visual_proj:    (batch_size, max_seq_len, hidden_dim)   — projected ResNet50
fused:          (batch_size, max_seq_len, hidden_dim)   — item + pos + visual
```

**Deliverable:** A tested fusion module that produces the combined embedding for the transformer.

---

### 5. BST Model Architecture  *(~1.5 days)*

> This is the core deliverable for Week 2.

| # | Sub-task | Details |
|---|----------|---------|
| 3a | Item + positional + visual embeddings | Extend the existing `HeroModel.__init__()` stub.  Three embedding sources summed: `nn.Embedding(num_items, hidden_dim)` + `nn.Embedding(max_seq_len, hidden_dim)` + `VisualProjection(2048, hidden_dim)`. |
| 3b | Transformer encoder stack | `nn.TransformerEncoder` with `num_layers=3`, `num_heads=4`, `hidden_dim=128` (from `config.yaml → hero`).  Causal mask so position *t* only attends to ≤ *t*.  Padding mask from sequence lengths. |
| 3c | Prediction head | Dot product of final hidden state with all item embeddings → `(batch_size, num_items)` logits.  No position bias (unlike the Villain — the Hero relies on visual features instead). |
| 3d | Contrastive learning head | Implement the Attribute-Aware Contrastive Loss using the existing `src/hero/contrastive.py` module.  For each anchor item, hard negatives are items from the same category but different visual style. |
| 3e | Combined loss function | `L_total = L_CE + λ * L_contrastive` where `λ = config.hero.contrastive.weight` (default 0.3). |

**Architecture diagram (Hero vs. Villain):**
```
Villain (Phase 1):
  Input IDs ─► Item Embed ──┐
                             ├─ + ─► Transformer ─► dot(Item Embeds) × pop_bias ─► logits
  Position  ─► Pos  Embed ──┘

Hero (Phase 2):
  Input IDs ─► Item Embed ──┐
                             │
  Position  ─► Pos  Embed ──┤─ + ─► Transformer ─► dot(Item Embeds) ─► logits
                             │                  └─► Contrastive Head ─► L_CL
  ResNet50  ─► Visual Proj ──┘
```

**Deliverable:** `HeroModel` class that accepts `(item_seq, positions, visual_embeds)` and returns `(predictions, contrastive_embeds)`.

---

### 6. Training Loop  *(~1 day)*

| # | Sub-task | Details |
|---|----------|---------|
| 4a | Implement `src/hero/trainer.py::train_hero()` | Extend the existing stub.  Standard PyTorch loop: forward → combined loss → backward → optimizer step. |
| 4b | Optimizer & scheduler | AdamW with `lr = 0.0005`, `weight_decay = 0.01`.  CosineAnnealingLR or ReduceLROnPlateau. |
| 4c | Validation & early stopping | Evaluate nDCG@12 on val set every epoch; stop if no improvement for 10 epochs. |
| 4d | Checkpoint saving | Save best model to `checkpoints/hero_best.pt`, latest to `checkpoints/hero_latest.pt`.  Same format as Villain checkpoints. |
| 4e | Logging | Use `src/utils/helpers.py::setup_logging()`.  Print epoch, CE loss, contrastive loss, total loss, val metrics per epoch. |
| 4f | Contrastive hard-negative mining | Each epoch, sample hard negatives from the same product group but with different visual clusters. |

**Training command (run from project root with venv activated):**
```powershell
.venv\Scripts\Activate.ps1; python -m src.hero.trainer
```

**Deliverable:** End-to-end trainable Hero that converges on the sampled data.

---

### 7. Cold Start Simulation  *(~1 day)*

> Steve's specific requirement — must be demo-ready for the presentation.

| # | Sub-task | Details |
|---|----------|---------|
| 5a | Define "cold start" items | Select 50–100 real articles from the catalog that were excluded from training (zero interactions in sampled data).  These simulate "new arrivals." |
| 5b | Villain cold-start evaluation | Feed these items through the Villain.  Expected: the Villain cannot distinguish them (all get near-zero scores since they have no position/interaction history). |
| 5c | Hero cold-start evaluation | Feed the same items through the Hero, providing only their ResNet50 visual embeddings.  Expected: the Hero ranks visually similar items near existing popular items of the same style. |
| 5d | Quantitative comparison | For each cold-start item, record its rank in both models.  Compute: average rank, % appearing in top-12, % appearing in top-50. |
| 5e | Qualitative showcase | Find 3–5 compelling examples where the Hero rescued a visually appealing niche item from obscurity.  Save example images and rank comparisons for the presentation. |
| 5f | Cold-start results log | Write results to `outputs/hero_cold_start_results.json`. |

**Deliverable:** Quantitative proof that visual embeddings enable cold-start recommendations.

---

### 8. Evaluation & Comparison  *(~0.5 day)*

| # | Sub-task | Details |
|---|----------|---------|
| 6a | Full eval pass | Run the Hero on the test set.  Report nDCG@12, MRR, Catalog Coverage (same metrics as Villain). |
| 6b | Per-bucket breakdown | Evaluate separately on head / torso / tail items.  Compare with Villain Phase 1 results. |
| 6c | Recommendation bias analysis | Where do the Hero's top-12 recs go?  Compare: Villain was 76.3% head / 21.6% torso / 2.2% tail. |
| 6d | Baseline results log | Write results to `outputs/hero_baseline_results.json` for comparison with Phase 3 multi-objective version. |

**Expected improvement targets:**
- Tail-item recommendation rate: 2.2% (Villain) → **8–15%** (Hero)
- Catalog Coverage: significant increase
- nDCG@12: maintain or modestly improve over Villain

**Deliverable:** Printed & saved evaluation scores for the Hero model alongside Villain comparison.

---

### 9. Documentation  *(ongoing)*

| # | Sub-task | Details |
|---|----------|---------|
| 7a | Update `docs/matrix_shapes.md` | Add Hero model tensor dimensions: fused embeddings, contrastive anchors, visual projections. |
| 7b | Update `docs/decision_log.md` | Document BST architecture choices: fusion strategy, contrastive loss design, cold-start simulation methodology. |
| 7c | Code docstrings | Every new/modified `.py` file follows the format in `rules.md §3`: module docstring with purpose, team member, key classes/functions. |
| 7d | Update `config.yaml` | Add any new hero-specific config keys (e.g., `use_visual`, cold-start params). |

---

## Dependencies on Teammates

| Teammate | What I need from them | Blocking? |
|----------|----------------------|-----------|
| **Elizabeth** | ~~ResNet50 visual embeddings~~ — **I am covering this (absent).** | N/A |
| **Elizabeth** | ~~Multimodal metadata fusion~~ — **I am covering this (absent).** | N/A |
| **Nishant** | Matrix shape diagrams for the Hero model | **No** — I provide the shapes, he creates the visuals. |
| **Nishant** | The "Aha! Moment" qualitative rescue example | **No** — I provide the model and cold-start evaluation; he finds the best showcase examples. |

---

## Risk & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| ResNet50 extraction takes too long (~105k images) | Delays all downstream tasks | Use GPU batched inference (batch_size=64); can run overnight if needed. |
| Some article images are missing or corrupted | Incomplete embedding bank | Zero-fill missing articles; log count and verify coverage. |
| 2048-dim visual vectors blow up GPU memory | CUDA OOM during training | Project to `hidden_dim=128` before feeding to transformer; keep embedding bank on CPU. |
| Contrastive loss destabilizes training | Loss diverges or NaN gradients | Use gradient clipping, temperature scaling (`τ=0.07`), and loss weighting (`λ=0.3`). |
| Cold-start items have no visual diversity | Unimpressive demo results | Pre-screen cold-start candidates for visual variety across categories. |
| BST overfits on small sampled dataset | Poor generalization | Dropout=0.1, early stopping, weight decay. |
| Covering two people's work is too much for one week | Tasks slip | Prioritise extraction + BST core first; metadata fusion can be simplified if needed. |

---

## Suggested Execution Order

```
Day 1 (Feb 22):  Task 1 (ResNet50 extraction — run overnight on GPU)
Day 2 (Feb 23):  Task 2 (metadata fusion) + Task 3 (embedding integration into DataLoaders)
Day 3 (Feb 24):  Task 4 (fusion layer) + Task 5a–5c (BST model architecture — core)
Day 4 (Feb 25):  Task 5d–5e (contrastive head + combined loss)
Day 5 (Feb 26):  Task 6 (training loop) — first training run overnight
Day 6 (Feb 27):  Task 7 (cold start simulation) + Task 8 (evaluation & comparison)
Day 7 (Feb 28):  Task 9 (docs) + buffer / bug fixes / coordination with Nishant
```

---

## Progress Journal

### March 1, 2026

#### ✅ Task 1: ResNet50 Feature Extraction — DONE

**What was built:** `src/data/extract_visual_embeddings.py` — GPU-batched ResNet50 extraction pipeline with:
- Frozen `torchvision.models.resnet50(IMAGENET1K_V2)`, final FC head stripped → 2048-dim avg-pool output
- `ArticleImageDataset` for lazy image loading with `torchvision.io.read_image`
- ImageNet normalisation (resize 256, center-crop 224, mean/std normalise)
- Automatic zero-fill for missing/corrupt images, logged as `missing_ids`
- Post-extraction verification: spot-checks 10 random embeddings for shape and non-zero values

**Extraction results:**

| Metric | Value |
|--------|-------|
| Total articles | 105,542 |
| Missing images | 442 (0.4%) |
| Embedding shape | `[105542, 2048]` |
| Throughput | 113 img/s |
| Total time | 932.7 s (~15.5 min) |
| Output file | `data/embeddings/resnet50_embeddings.pt` |

**Run command:** `.venv\Scripts\Activate.ps1; python -m src.data.extract_visual_embeddings`

---

## Success Criteria (End of Week 2)

- [x] ResNet50 embeddings extracted for all ~105k articles → `data/embeddings/resnet50_embeddings.pt`
- [ ] Multimodal (visual + metadata) embeddings fused → `data/embeddings/multimodal_embeddings.pt`
- [ ] `HeroModel` trains to convergence on the sampled data with visual embeddings
- [ ] Hero nDCG@12 ≥ Villain nDCG@12 (0.145)
- [ ] Tail-item recommendation rate improves from Villain's 2.2% to ≥ 8%
- [ ] Cold-start simulation demonstrates the Hero can rank unseen items using visual features
- [ ] Contrastive learning head is functional and contributes to long-tail discovery
- [ ] All evaluation results saved to `outputs/hero_baseline_results.json`
- [ ] Decision log updated with BST design choices and ResNet50 extraction methodology
- [ ] All code follows `rules.md` (docstrings, config-driven, no hard-coded paths)
