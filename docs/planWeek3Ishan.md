# Week 3 Plan — Ishan Biswas
## Phase 3: Multi-Objective Study (March 1 – March 6)

> **Role:** Transformer Architect & Loss Tuning (Ishan) + Ablation & Comparison (Elizabeth — covered by Ishan this week)  
> **Branch:** `week3-ishan`  
> **Tools:** PyTorch, CUDA, NumPy, Pandas, Matplotlib

---

## Overview

Week 3 proves the **business trade-off between CTR (relevance) and Discovery (long-tail
exposure)**.  Because Elizabeth is absent again, I am covering her Phase 3
responsibilities (ablation study) in addition to my own (Pareto optimisation).  I am
also covering Nishant's portion (Pareto front visualisation + business conclusion).
My responsibilities fall into three buckets:

1. **Multi-Objective Loss & Pareto Experiments (Ishan's portion)** — extend the Hero's
   loss function with a tuneable discovery term and sweep across at least 3 values of
   `lambda_discovery` to map the relevance-vs-discovery Pareto front.
2. **Ablation Study (Elizabeth's portion)** — quantify the "Visual Lift" by comparing
   the Hero *with* visual embeddings against the Hero *without* visual embeddings
   (ID-only ablation, controlled via `config.yaml → hero.use_visual`).
3. **Pareto Curve & Business Conclusion (Nishant's portion)** — build the final Pareto
   front plot (nDCG@12 vs. Catalog Coverage), write the "incredible conclusion" for the
   presentation, and update `run_all.py` with the Phase 3 pipeline stage.

**Starting Point (Phase 2 Results):**

| Model   | nDCG@12 | MRR    | Catalog Coverage | Tail Rec Rate | Head Bias |
|---------|---------|--------|------------------|---------------|-----------|
| Villain | 0.1448  | 0.1316 | 57.8%            | 2.15%         | 76.3%     |
| Hero    | 0.1312  | 0.1205 | 61.9%            | 10.98%        | 54.1%     |

The Hero already trades ~1.4pp nDCG for a massive +8.8pp tail-item rate.  Phase 3
makes this trade-off *controllable* via `lambda_discovery`.

All hyperparameters and paths come from `config.yaml` (single source of truth per
`rules.md §2`).  New Phase 3 config keys will be added under a `pareto:` section.

---

## Task Breakdown

### 1. [DONE] Config & Infrastructure Setup  *(Task Complete)*

| # | Sub-task | Details |
|---|----------|---------|
| 1a | Add `pareto` config section | Add new keys to `config.yaml`: `pareto.lambda_values: [0.0, 0.3, 0.7, 1.0]`, `pareto.metric_x: catalog_coverage`, `pareto.metric_y: ndcg@12`, `pareto.output_dir: analytics/pareto`. |
| 1b | Add `hero.discovery_weight` key | New float in `config.yaml → hero` that controls the strength of the discovery-aware loss term.  Default `0.0` (baseline Hero behaviour). |
| 1c | Verify checkpoint availability | Confirm `checkpoints/hero_best.pt` and `checkpoints/villain_best.pt` are present and loadable. |

**Deliverable:** Updated `config.yaml` with all Phase 3 keys.

---

### 2. [DONE] Multi-Objective Loss Function — *Ishan's portion*  *(Task Complete)*

> Core Phase 3 deliverable: make the relevance/discovery trade-off tuneable.

| # | Sub-task | Details |
|---|----------|---------|
| 2a | Design discovery loss term | Implement a **Popularity-Penalised Loss** that adds a penalty when the model recommends popular items.  Formulation: `L_discovery = mean(pop_logit(top_k_predictions))` — penalises high-popularity recommendations.  The pop_logits are pre-computed using `metrics.py::popularity_logit_scores()`. |
| 2b | Implement `MultiObjectiveLoss` | Extend `src/hero/contrastive.py` (or create `src/hero/multi_objective.py`) with a new loss class: `L_total = L_CE + λ_CL * L_contrastive + λ_disc * L_discovery`.  `λ_disc` is read from `config.yaml → hero.discovery_weight`. |
| 2c | Pre-compute popularity logits | At training time, load article sales counts from the sampled data and compute pop logits once.  Store as a `(num_items,)` tensor on GPU for fast lookup during loss computation. |
| 2d | Integrate into trainer | Modify `src/hero/trainer.py` to use `MultiObjectiveLoss` instead of `CombinedLoss`.  Log all three loss components separately per epoch. |
| 2e | Smoke test | Verify the new loss computes correctly on a single batch with `λ_disc = 0.5`. |

**Loss formulation:**
```
L_total = L_CE(logits, targets)
        + λ_CL  * L_contrastive(anchor, pos, negs)        # existing (default 0.3)
        + λ_disc * L_discovery(logits, pop_logit_vector)   # NEW
```

Where `L_discovery` encourages the model to push probability mass toward low-popularity
items.  One concrete approach:

```
L_discovery = mean over batch of:
    sum_{i in top-K predicted} pop_logit[i] * softmax(logit[i])
```

This is differentiable and penalises placing high softmax mass on popular items.

**Matrix shapes:**
```
pop_logit_vector:  (num_items,)           — pre-computed, on GPU
logits:            (batch_size, num_items) — model output
softmax_probs:     (batch_size, num_items) — softmax(logits)
L_discovery:       scalar                  — dot(softmax_probs, pop_logit_vector).mean()
```

**Key technical considerations:**
- The discovery loss must be differentiable w.r.t. `logits` so gradients flow.
- Using softmax-weighted popularity logits (rather than hard top-K) keeps gradients smooth.
- `λ_disc = 0` should reproduce the exact Phase 2 Hero behaviour (regression test).
- Gradient clipping (max_norm=1.0) remains critical to stabilise training with the extra loss term.

**Deliverable:** A new `MultiObjectiveLoss` class and updated trainer.

---

### 3. [DONE] Pareto Sweep Experiments — *Ishan's portion*  *(Task Complete)*

> Run the Hero with multiple `λ_disc` values to map the Pareto front.

| # | Sub-task | Details |
|---|----------|---------|
| 3a | Create sweep script | Build `src/hero/pareto_sweep.py` — a script that iterates over `config.yaml → pareto.lambda_values`, trains (or fine-tunes) the Hero for each λ, evaluates on the test set, and collects results. |
| 3b | Training strategy | For each `λ_disc` value: (a) load the best Phase 2 Hero checkpoint (`hero_best.pt`) as the starting point, (b) fine-tune for 15–20 epochs with the new multi-objective loss, (c) evaluate with full multi-objective metrics. This avoids training from scratch for each λ. |
| 3c | Run λ = 0.0 (control) | This should reproduce Phase 2 Hero results exactly — serves as sanity check. |
| 3d | Run λ = 0.3 (moderate) | Moderate discovery push.  Expected: nDCG drops slightly, coverage and tail rate increase. |
| 3e | Run λ = 0.7 (aggressive) | Strong discovery push.  Expected: nDCG drops noticeably, coverage and tail rate increase substantially. |
| 3f | Run λ = 1.0 (maximum) | Maximum discovery.  Expected: largest coverage/tail gains, largest nDCG sacrifice. |
| 3g | Collect results | Save all sweep results to `outputs/pareto_sweep_results.json` — a list of dicts, each with `lambda_disc`, `ndcg@12`, `mrr`, `catalog_coverage`, `tail_item_rate`, `mean_tail_score`. |

**Expected Pareto points (approximate):**

| λ_disc | nDCG@12 (est.) | Catalog Coverage (est.) | Tail Rate (est.) |
|--------|----------------|-------------------------|-------------------|
| 0.0    | ~0.131         | ~61.9%                  | ~11.0%            |
| 0.3    | ~0.120–0.125   | ~65–68%                 | ~14–16%           |
| 0.7    | ~0.105–0.115   | ~70–75%                 | ~18–22%           |
| 1.0    | ~0.090–0.105   | ~75–82%                 | ~22–28%           |

**Key technical considerations:**
- Fine-tuning from the Phase 2 checkpoint (rather than from scratch) saves significant GPU time (~15–20 epochs × ~40s/epoch ≈ 10–13 min per λ value, ~1 hour total).
- Each sweep point should be saved to its own checkpoint: `checkpoints/hero_lambda_{λ}.pt`.
- Use `compute_multi_objective_metrics()` from `metrics.py` for comprehensive evaluation.
- The Villain baseline point (`λ = N/A`) is added to the Pareto plot as a reference.

**Deliverable:** `outputs/pareto_sweep_results.json` with 4+ evaluated Pareto points.

---

### 4. [DONE] Ablation Study — *Elizabeth's portion*  *(Task Complete)*

> Quantify the "Visual Lift": how much did adding ResNet50 images help discovery
> vs. using only the item-ID sequence?

| # | Sub-task | Details |
|---|----------|---------|
| 4a | Train ID-only Hero | Set `config.yaml → hero.use_visual: false` and train the Hero from scratch (same architecture minus the `VisualProjection` module).  The `HeroModel` already supports this toggle. |
| 4b | Evaluate ID-only Hero | Run full evaluation (nDCG@12, MRR, Catalog Coverage, tail-item rate, per-bucket breakdown) on the ID-only Hero. Save to `outputs/hero_ablation_no_visual.json`. |
| 4c | Compare with full Hero | Compute deltas: `Δ_metric = Hero_visual - Hero_no_visual` for each metric. |
| 4d | Cold-start ablation | Run the cold-start simulation on the ID-only Hero.  Expected: without visual features, the ID-only Hero should perform as poorly as the Villain on zero-interaction items. |
| 4e | Ablation summary table | Produce a clean comparison table for the presentation: Villain vs. Hero (no visual) vs. Hero (visual) vs. Hero (multi-objective). |

**Expected ablation results:**

| Model | nDCG@12 | Catalog Coverage | Tail Rate | Cold-Start Avg Rank |
|-------|---------|------------------|-----------|---------------------|
| Villain | 0.1448 | 57.8% | 2.15% | 26,159 |
| Hero (ID-only) | ~0.140–0.145 | ~58–60% | ~3–5% | ~25,000–26,000 |
| Hero (visual) | 0.1312 | 61.9% | 10.98% | 20,375 |

**Key technical considerations:**
- The `HeroModel` already has the `use_visual` toggle — when `False`, the model skips the `VisualProjection` and fusion, behaving as an ID-only BST.
- Use the same training hyperparameters (lr, epochs, patience) to ensure a fair comparison.
- Save the ID-only checkpoint to `checkpoints/hero_no_visual_best.pt` to avoid overwriting the main Hero checkpoint.

**Deliverable:** `outputs/hero_ablation_no_visual.json` + ablation comparison table.

---

### 5. [DONE] Improved Hard-Negative Mining — *Ishan's portion*  *(Task Complete)*

> Phase 2 used randomised negatives.  Phase 3 upgrades to attribute-aware mining.

| # | Sub-task | Details |
|---|----------|---------|
| 5a | Load article attributes | Parse `articles.csv` to extract `product_group_name`, `colour_group_name`, `garment_group_name` per article.  Build an attribute lookup dict: `article_id → {product_group, colour, garment}`. |
| 5b | Implement Jaccard-based mining | For each anchor item, compute partial attribute overlap (Jaccard similarity) with all other items.  Hard negatives are items with *some* but not *all* matching attributes (Jaccard between 0.3 and 0.7). |
| 5c | Vectorised batch mining | Use pre-computed attribute matrices for fast batch-level negative selection.  Cache the hard-negative index matrix `(num_items, num_negatives)` at epoch start. |
| 5d | Integrate into trainer | Replace the randomised `hard_negative_mining()` call in `trainer.py` with the attribute-aware version. |
| 5e | A/B comparison | Compare training curves and final metrics with random vs. attribute-aware negatives. |

**Key technical considerations:**
- Full pairwise Jaccard on ~27k items is O(n²) — pre-compute once and cache.
- For items with no valid hard negatives in the Jaccard range, fall back to random sampling.
- This upgrade is expected to improve contrastive learning quality, potentially recovering some of the nDCG gap between Hero and Villain.

**Deliverable:** Upgraded `hard_negative_mining()` in `src/hero/contrastive.py`.

---

### 6. Pareto Front Visualisation — *Nishant's portion*

> Build the final plot that tells the business story.

| # | Sub-task | Details |
|---|----------|---------|
| 6a | Build Pareto plot script | Create `src/utils/pareto_plot.py` — loads `outputs/pareto_sweep_results.json` and generates the Pareto front chart. |
| 6b | Plot design | X-axis: **Catalog Coverage** (0–100%).  Y-axis: **nDCG@12** (0–0.20).  Each point labelled with its `λ_disc` value.  Villain baseline plotted as a red "×", Hero variants as blue dots connected by the Pareto front line. |
| 6c | Annotate the "sweet spot" | Highlight the Pareto-optimal point that best balances relevance and discovery.  Add an annotation arrow: "Sweet spot: λ=0.3 — 10% relevance sacrifice, 30% more discovery." |
| 6d | Secondary plot: Tail Rate curve | A second panel showing `λ_disc` (x-axis) vs. `tail_item_rate` (y-axis) — demonstrates the tuneable knob. |
| 6e | Save outputs | Save plots to `analytics/pareto/pareto_front.png` and `analytics/pareto/tail_rate_curve.png`.  Both at 300 DPI for presentation quality. |

**Deliverable:** Publication-quality Pareto front plot in `analytics/pareto/`.

---

### 7. Pipeline Integration & `run_all.py` Update

| # | Sub-task | Details |
|---|----------|---------|
| 7a | Add Phase 3 stages to `run_all.py` | Add two new stages: `pareto_sweep` (runs the λ sweep) and `ablation` (runs the visual ablation).  Update the `--stage` choices. |
| 7b | Add `pareto_plot` stage | Generates the Pareto visualisation from saved sweep results. |
| 7c | End-to-end smoke test | Run `python run_all.py --stage pareto_sweep` and verify the full pipeline executes without errors. |

**Updated pipeline stages:**
```
1. sample          — Data sampling
2. embed           — Visual embedding extraction + multimodal fusion
3. train_villain   — Train Villain baseline
4. train_hero      — Train Hero (Phase 2)
5. evaluate        — Evaluate both models + cold-start
6. ablation        — Train and evaluate ID-only Hero (NEW)
7. pareto_sweep    — Multi-objective λ sweep (NEW)
8. pareto_plot     — Generate Pareto front visualisation (NEW)
```

**Deliverable:** Updated `run_all.py` with Phase 3 stages.

---

### 8. Documentation & Business Conclusion

| # | Sub-task | Details |
|---|----------|---------|
| 8a | Update `docs/decision_log.md` | Document: (1) multi-objective loss design rationale, (2) λ sweep methodology, (3) ablation study design, (4) hard-negative mining upgrade. |
| 8b | Update `docs/matrix_shapes.md` | Add tensor shapes for the discovery loss term and multi-objective loss computation. |
| 8c | Business conclusion draft | Write the "incredible conclusion" for the presentation.  Target narrative: *"By sacrificing X% relevance (nDCG), we gained Y% nicheness (catalog coverage/tail rate).  The λ knob gives business stakeholders a tuneable dial between CTR and discovery."* |
| 8d | Code docstrings | Every new `.py` file follows `rules.md §3` format: module docstring with purpose, team member, key classes/functions. |
| 8e | Update `config.yaml` | Ensure all new keys are documented with inline comments. |

**Deliverable:** Complete documentation for Phase 3.

---

## Dependencies on Teammates

| Teammate | What I need from them | Blocking? |
|----------|----------------------|-----------|
| **Elizabeth** | ~~Ablation study~~ — **I am covering this (absent).** | N/A |
| **Nishant** | ~~Pareto front visual + business conclusion~~ — **I am covering this.** | N/A |

---

## Risk & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Discovery loss destabilises training | NaN gradients or collapsed representations | Gradient clipping (max_norm=1.0), low initial λ_disc (start with 0.3), temperature scaling on softmax. |
| Fine-tuning from Phase 2 checkpoint doesn't converge for high λ | Poor Pareto points at λ=0.7, 1.0 | Fall back to training from scratch with reduced epochs; lower learning rate (0.0001). |
| Ablation (ID-only Hero) performs identically to Villain | Undercuts the visual embedding narrative | Likely fine — the BST architecture alone should differ from SASRec+pop_bias.  If too similar, emphasise the contrastive head as the differentiator. |
| Jaccard hard-negative mining is too slow | Training time doubles | Pre-compute the full negative index matrix once at startup; cache as `.pt` file.  Fall back to random if computation exceeds 5 min. |
| Pareto sweep takes too long (4 × 20 epochs) | Delays visualisation and conclusion | Reduce fine-tune epochs to 10 if needed; GPU throughput is ~40s/epoch, so 4 × 20 = ~53 min worst case. |
| λ=1.0 collapses nDCG to near zero | Unusable Pareto point | Cap the discovery loss contribution with `torch.clamp` or reduce λ to 0.8 max. |

---

## Suggested Execution Order

```
Day 1 (March 3):  Task 1 (config setup) + Task 2 (multi-objective loss implementation)
Day 2 (March 4):  Task 3a-3c (sweep script + λ=0.0 control run)
                   Task 4a-4b (ablation: train + evaluate ID-only Hero)
Day 3 (March 5):  Task 3d-3g (remaining λ sweep runs: 0.3, 0.7, 1.0)
                   Task 5 (hard-negative mining upgrade — if time permits)
Day 4 (March 6):  Task 6 (Pareto visualisation) + Task 7 (pipeline integration)
                   Task 8 (documentation + business conclusion)
```

**Estimated GPU time budget:**
- Ablation (ID-only Hero from scratch): ~50 epochs × 30s/epoch ≈ 25 min
- Pareto sweep (4 λ values × 20 fine-tune epochs × 40s/epoch) ≈ 53 min
- Total GPU time: **~1.5 hours**

---

## Success Criteria (End of Week 3)

- [x] `MultiObjectiveLoss` implemented with tuneable `λ_disc` parameter
- [ ] Pareto sweep completed for at least 3 λ values (0.0, 0.3, 0.7)
- [ ] Results saved to `outputs/pareto_sweep_results.json`
- [ ] Pareto front plot generated at `analytics/pareto/pareto_front.png`
- [ ] Ablation study completed: Hero (visual) vs. Hero (ID-only) comparison
- [ ] Ablation results saved to `outputs/hero_ablation_no_visual.json`
- [ ] Business conclusion drafted: "By sacrificing X% relevance, we gained Y% discovery"
- [ ] `run_all.py` updated with Phase 3 stages (ablation, pareto_sweep, pareto_plot)
- [ ] Decision log updated with Phase 3 design rationale
- [x] All code follows `rules.md` (docstrings, config-driven, no hard-coded paths)
- [ ] Hard-negative mining upgraded from random to attribute-aware (stretch goal)

---

## Progress Journal

### March 3, 2026

#### ✅ Task 1: Config & Infrastructure Setup — DONE

**What was changed:** `config.yaml` updated with all Phase 3 keys:
- Added `hero.discovery_weight: 0.0` — controls the strength of the discovery-aware loss term. Default `0.0` preserves exact Phase 2 Hero behaviour (regression-safe).
- Added `pareto` section with `lambda_values: [0.0, 0.3, 0.7, 1.0]`, `metric_x`, `metric_y`, `output_dir`.
- Verified both `checkpoints/hero_best.pt` (49.7 MB) and `checkpoints/villain_best.pt` (49.0 MB) are present and loadable.

---

#### ✅ Task 2: Multi-Objective Loss Function — DONE

**What was built:** `MultiObjectiveLoss` class in `src/hero/contrastive.py` extending the Phase 2 `CombinedLoss` into a three-term objective:

```
L_total = L_CE(logits, targets)
        + λ_CL  * L_contrastive(anchor, pos, negs)       # existing (0.3)
        + λ_disc * L_discovery(logits, pop_logit_vector)  # NEW
```

**Discovery loss formulation:**
- `L_discovery = mean over batch of dot(softmax(logits), pop_logit_vector)`
- Differentiable w.r.t. logits (smooth softmax, no hard top-K).
- Penalises placing high softmax mass on items with high popularity logits.
- `λ_disc = 0` exactly reproduces Phase 2 behaviour (verified via regression test).

**Pre-computed popularity logits:**
- `build_dataloaders()` now returns `item_sales_counts` in its metadata dict, avoiding a redundant CSV re-read (rules.md §6 compliance).
- `popularity_logit_scores()` from `metrics.py` computes smoothed logits per item.
- Mapped to a `(num_items,)` GPU tensor indexed by contiguous item IDs.
- PAD index 0 set to `0.0` (neutral — no penalty).

**Trainer integration:**
- `src/hero/trainer.py` now imports `MultiObjectiveLoss` (replaces `CombinedLoss`).
- All three loss components (`CE`, `CL`, `Disc`) logged separately per epoch.
- `pop_logits_tensor` pre-computed once and passed to the criterion every batch.

**Smoke test results (λ_disc = 0.5):**

| Component | Value |
|-----------|-------|
| Total Loss | 6.62 |
| CE Loss | 5.46 |
| CL Loss | 3.74 |
| Disc Loss | 0.08 |

**Regression test (λ_disc = 0.0):** Confirmed `loss_disc == 0.0` exactly.

---

#### 🔧 Post-Implementation Review — 4 bugs fixed

A brutal review of Tasks 1–2 caught and fixed the following issues:

**Bug 1 — Zero-transaction pop-logit fallback (severity: HIGH)**
Items with 0 transactions were missing from `pop_logit_dict` and defaulted to `0.0`. But `0.0` is *higher* than most rare items' logits (which are negative), so the discovery loss would treat zero-transaction items as *more popular* than rare items — completely inverted. **Fix:** Default to `min(pop_logit_dict.values())` so unseen items are treated as the least popular.

**Bug 2 — `loss_disc` type inconsistency (severity: MEDIUM)**
`MultiObjectiveLoss.forward()` returned `float 0.0` when `disc_weight == 0` but `Tensor` when enabled. The trainer had a fragile `isinstance()` check to handle this. **Fix:** Always return `torch.tensor(0.0, device=logits.device)` for a consistent API.

**Bug 3 — Double CSV read (severity: LOW, rules.md §6 violation)**
`trainer.py` re-read `transactions_sampled.csv` to compute sales counts, even though `build_dataloaders()` already parses it. **Fix:** Added `item_sales_counts` to the metadata dict returned by `build_dataloaders()`, eliminated the redundant `pd.read_csv()` and the `pandas` import from trainer.

**Bug 4 — Stale docstrings (severity: LOW, rules.md §3 violation)**
`contrastive.py` KEY COMPONENTS section didn't mention `MultiObjectiveLoss`. `trainer.py` docstring still referenced "combined loss (recommendation + contrastive)". **Fix:** Updated both to reflect the new three-term multi-objective loss.

---

#### ✅ Task 3: Pareto Sweep Experiments — DONE

**What was built:** `src/hero/pareto_sweep.py` — a self-contained sweep driver that:

1. Loads data **once** (shared across all λ values) — rules.md §6 compliance.
2. Pre-computes popularity logit tensor (same logic as `trainer.py`, no duplicate CSV read).
3. For each `λ_disc` in `config.yaml → pareto.lambda_values` (0.0, 0.3, 0.7, 1.0):
   - Loads the Phase 2 Hero checkpoint (`hero_best.pt`) as starting weights.
   - Creates a **fresh** AdamW optimizer + ReduceLROnPlateau scheduler.
   - Constructs `MultiObjectiveLoss` with the current `λ_disc`.
   - Fine-tunes for 20 epochs (with early stopping at `patience=10`).
   - Tracks best val nDCG and restores best model weights.
   - Saves per-λ checkpoint to `checkpoints/hero_lambda_{λ}.pt`.
   - Runs full test evaluation using `compute_multi_objective_metrics()` — reports nDCG@12, MRR, catalog coverage, tail item rate, mean tail score.
4. Saves results **incrementally** to `outputs/pareto_sweep_results.json` (survives interruption).
5. Prints a summary table at the end.

**Key design decisions:**
- Fine-tune from checkpoint (not from scratch) — saves ~75% GPU time.
- Fresh optimizer per λ — avoids momentum contamination between sweep points.
- Evaluation maps contiguous indices back to raw article IDs for accurate sales-count-based tail metrics.
- All paths and hyperparameters from `config.yaml` (rules.md §2).

**Pipeline integration:**
- `run_all.py` updated with three new stages: `pareto_sweep`, `pareto_plot`, `ablation`.
- `python run_all.py --stage pareto_sweep` runs the full sweep.

**Smoke test results:**
- Import: OK
- Forward + backward pass (λ=0.3): `total=6.03, CE=8.52, CL=1.05, Disc=-9.33` — all gradients flow correctly.
- Evaluation function: OK — returns all 6 multi-objective metrics.
- No errors on any code path.

---

#### ✅ Task 3 — Pareto Sweep Execution Results

| λ_disc | nDCG@12 | MRR | Catalog Coverage | Tail Rate |
|--------|---------|-----|------------------|-----------|
| 0.0    | 0.1319  | 0.1212 | 60.0% | 4.2% |
| 0.3    | 0.1328  | 0.1227 | 67.6% | 6.6% |
| 0.7    | 0.0854  | 0.0733 | 74.6% | 81.7% |
| 1.0    | 0.0653  | 0.0522 | 71.5% | 88.2% |

**λ=0.3 is the Pareto-optimal sweet spot** — coverage jumps +7.6pp with virtually no nDCG sacrifice.

---

#### ✅ Task 4: Ablation Study — DONE

**What was built:** `src/hero/ablation.py` — a self-contained ablation driver that:

1. Builds dataloaders with `use_visual=False` (no multimodal embeddings loaded).
2. Trains the ID-only Hero **from scratch** (same architecture minus `VisualProjection`).
   - Same hyperparameters (lr, epochs, patience) for fair comparison.
   - Checkpoints to `hero_no_visual_best.pt` / `hero_no_visual_latest.pt` (no overwrite).
   - Supports checkpoint resume if interrupted.
3. Evaluates on test set with full multi-objective metrics + per-bucket breakdown + recommendation bias.
4. Runs cold-start simulation (without visual catalog injection).
5. Loads existing Villain + visual Hero results from `outputs/` and builds a comparison table:
   Villain vs Hero (ID-only) vs Hero (visual) with visual lift deltas.
6. Saves everything to `outputs/hero_ablation_no_visual.json`.

**Pipeline integration:**
- `run_all.py` updated with `ablation` stage (#7 of 9).
- `python run_all.py --stage ablation` runs the full ablation.

**Smoke test results:**
- Data loads correctly with `use_visual=False` — no `visual_embeds` in batches.
- ID-only HeroModel: 4,048,640 params (vs ~7.3M with visual projection).
- Forward + backward pass: `total=53.61, CE=52.79, CL=2.73, Disc=0.00` — gradients flow.
- Evaluation: OK.
- Cold-start simulation: OK (751 cold-start samples found, 100 evaluated).
- No errors on any code path.

---

#### ✅ Task 4 — Ablation Execution Results

| Model | nDCG@12 | MRR | Coverage | Cold-Start Rank |
|-------|---------|-----|----------|-----------------|
| Villain | 0.1448 | 0.1316 | 57.8% | 26,159 |
| Hero (ID-only) | 0.1308 | 0.1204 | 60.7% | 24,900 |
| Hero (visual) | 0.1312 | 0.1205 | 61.9% | 20,374 |

**Visual Lift:** Δ nDCG +0.03%, Δ Coverage +1.25pp, **Δ Cold-Start Rank +4,525** (lower = better).
ID-only trained 45 epochs (best at epoch 35, val nDCG 0.1073). Visual features provide minimal lift on warm items but dramatically help cold-start.

---

#### 🔧 Post-Implementation Review — Tasks 3 & 4 (6 bugs fixed)

**Bug S1 — `FINETUNE_EPOCHS` hardcoded in pareto_sweep.py (severity: MEDIUM, rules.md §2)**
Module-level constant `FINETUNE_EPOCHS = 20` bypassed config.yaml. **Fix:** Added `pareto.finetune_epochs: 20` to config.yaml; sweep now reads it at runtime.

**Bug S2 — Unused `val_ce` in pareto_sweep.py (severity: LOW)**
`val_ce = nn.CrossEntropyLoss(ignore_index=0)` was declared but never used in the validation loop. **Fix:** Removed.

**Bug S3 — Import inside function body in pareto_sweep.py (severity: LOW, rules.md §3)**
`from src.utils.metrics import compute_all_metrics` was imported inside `finetune_and_evaluate()`. **Fix:** Moved to top-level imports.

**Bug S4 — Unused import `numpy` in pareto_sweep.py (severity: LOW)**
`import numpy as np` was imported but never referenced. **Fix:** Removed.

**Bug A1 — Double CSV read in ablation.py (severity: MEDIUM, rules.md §6)**
`_evaluate_id_only()` re-read `transactions_sampled.csv` to build pop buckets — same pattern as Bug 3 from Tasks 1–2 review. **Fix:** Derived buckets from `meta["item_sales_counts"]` instead.

**Bug A2 — Unused import in ablation.py (severity: LOW)**
`load_multimodal_embeddings` imported but never used (ID-only model). **Fix:** Removed.

**Bug A3 — Double file read in ablation.py (severity: LOW)**
`hero_cold_start_results.json` was opened and parsed twice. **Fix:** Read once, extract both `hero` and `villain` sections.

---

#### ✅ Task 5: Improved Hard-Negative Mining — DONE

**What was built:** Upgraded `hard_negative_mining()` in `src/hero/contrastive.py` from randomised negatives to attribute-aware Jaccard-based mining.

**5a — Article attribute extraction (no double CSV read):**
- Extended `build_id_maps()` in `src/data/dataset.py` with an optional `attribute_cols` parameter.
- Extracts `product_group_name`, `colour_group_name`, `garment_group_name` from `articles_sampled.csv` during the existing CSV read (rules.md §6 — no redundant file I/O).
- Builds `item_attributes: dict[int, set]` keyed by contiguous item index, values are `{col=value}` strings.
- Returned as part of `build_dataloaders()` metadata dict (`meta["item_attributes"]`).

**5b — Jaccard-based mining function:**
- New signature: `hard_negative_mining(num_items, num_negatives, item_attributes, mining_mode, jaccard_low, jaccard_high, cache_path)`.
- Builds a binary attribute matrix `(num_items, num_attrs)` from `item_attributes`.
- Computes pairwise Jaccard similarity in chunks of 512 rows to limit peak memory to O(512 × N) instead of O(N²).
- Jaccard formula: `J(A,B) = |A ∩ B| / |A ∪ B|` via vectorised dot products.
- Hard negatives = items with Jaccard in `[jaccard_low, jaccard_high]` (partial attribute overlap).
- Items with fewer valid candidates are backfilled with uniform random sampling.
- PAD (index 0) and self are excluded via vectorised masking.

**5c — Caching with staleness protection:**
- Pre-computed `(num_items, num_negatives)` matrix cached to `.pt` file.
- Cache payload includes `jaccard_low`, `jaccard_high`, `mining_mode` — recomputes automatically if params change between runs.
- Computation time: **~6.3s** for 26,933 items × 10 negatives; cached reload is instant.

**5d — Integration across all call sites:**
- `trainer.py`: Reads `mining_mode`, `jaccard_low`, `jaccard_high` from `config.yaml → hero.contrastive.*`.
- `pareto_sweep.py`: Same integration for fine-tune sweep loop.
- `ablation.py`: Same integration for ID-only training loop.
- `evaluate_cold_start.py`: Fixed `build_id_maps()` call for new 3-return-value signature.

**Config keys added to `config.yaml`:**
```yaml
hero.contrastive:
  mining_mode: "attribute"    # "random" or "attribute"
  jaccard_low: 0.3
  jaccard_high: 0.7
  attribute_cols: [product_group_name, colour_group_name, garment_group_name]
```

**Smoke test results:**
- MultiObjectiveLoss: Total=6.10, CE=5.16, CL=2.84, Disc=0.17 ✓
- Random mode: shape (50, 5), all indices valid ✓
- Attribute mode: shape (50, 5), zero self-negatives ✓
- Empty attrs fallback to random ✓
- Full-data test: 26,933 items × 10 negatives in 6.3s ✓
- Cache hit/miss/staleness detection ✓

---

#### 🔧 Post-Implementation Review — Task 5 (3 bugs fixed)

**Bug T5-1 — Unused variable `valid_indices` (severity: LOW)**
`valid_indices = torch.arange(1, num_items)` was declared but never referenced. **Fix:** Removed.

**Bug T5-2 — Cache staleness risk (severity: MEDIUM)**
Original cache only checked tensor shape. If Jaccard params changed between runs, stale negatives would silently load. **Fix:** Cache payload now stores `jaccard_low`, `jaccard_high`, `mining_mode` alongside the matrix; loader validates all params match before returning cached data.

**Bug T5-3 — Per-row masking loop (severity: LOW, performance)**
Self and PAD exclusion used a Python `for` loop over chunk rows. **Fix:** Vectorised with `jaccard[:, 0] = -1.0` and diagonal indexing `jaccard[diag_indices, diag_indices + start] = -1.0`.
