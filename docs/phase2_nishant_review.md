# Phase 2 — Review & Refinements to Nishant's Deliverables
### Reviewer: Ishan Biswas | March 3, 2026

---

## Overview

This document summarizes the review and refinements applied to Nishant's Phase 2 work
(Matrix Mechanics & Qualitative Rescue) before merging into `main`.

---

## 1. `docs/matrix_shapes.md` — Matrix Shapes (Hero Section)

**Nishant's contribution:** Added the Hero model section with a layer-by-layer breakdown
of tensor shapes through the forward pass (Layers 1–9), covering input fusion, attention
mechanics, and output projection.

**Refinements applied:**

- **Added Hero-specific notation table.** Nishant's version reused the Villain notation
  without clarifying that the Hero uses `B=128` (not 256) and introduces `E=2048` for
  the ResNet50 visual dimension. A dedicated notation block was added for clarity.

- **Added ASCII forward pass diagram.** To match the Villain section's style and satisfy
  the rubric's "label shapes on every layer diagram" requirement, a full box-drawing
  diagram was added showing the entire Hero pipeline — including the contrastive learning
  head branch (anchor, positive, negatives → InfoNCE → combined loss).

- **Removed fabricated `time_embed` tensor.** Layer 2 referenced a `time_embed` (days
  since purchase) tensor that does not exist in the actual `HeroModel` implementation.
  This was removed to keep the document accurate to the codebase.

- **Added Hero parameter count table.** Nishant's version omitted this. A full breakdown
  was added: VisualProjection (262,272), fusion_norm (256), TransformerEncoder (396,672),
  etc. — totalling ~4,113,280 parameters.

- **Fixed batch size inconsistency.** The summary table and config line now consistently
  use `B=128` as defined in `config.yaml → hero.batch_size`.

---

## 2. `src/hero/evaluate_cold_start.py` — The "Aha!" Moment

**Nishant's contribution:** Identified the need for qualitative rescue examples showing
dramatic rank improvements (Villain buried → Hero surfaced). Implemented a search for
the single most compelling test case with strict criteria (`h_rank ≤ 12` and `v_rank ≥ 500`).

**Refinements applied:**

- **Added top-5 fallback logic.** The plan specifies capturing the "top-5 biggest rank
  improvements" as a fallback when the strict criteria (`h_rank ≤ 12 & v_rank > 500`)
  find no matches on purely cold-start items. An `all_rank_deltas` tracking list was
  added that records every sample's `v_rank − h_rank`, and when strict matches are
  empty, the top-5 by rank delta are surfaced instead.

- **Kept changes within the existing file.** Rather than creating a separate script,
  the fallback was integrated directly into `evaluate_cold_start.py` to keep the
  pipeline consistent with `run_all.py` and avoid duplicating model-loading logic.

- **Ensured config compliance.** All file paths reference `config.yaml` via the
  `config["paths"]` dict, consistent with `rules.md § 2`.

---

## 3. `src/data/dataset.py` — Customer ID in Batch

**Nishant's contribution:** Added `customer_id` (uid) to the dataset return dictionary
so that downstream scripts (e.g., cold-start analysis) can trace predictions back to
specific users.

**Refinements applied:**

- **Extended uid storage to training mode.** Nishant's original change only added `uid`
  to test/val sample tuples, which would cause an unpack error during training (train
  samples were still 2-tuples). The fix stores `uid` consistently across all three
  modes (train, val, test) so `__getitem__` can safely unpack `(inp, tgt, uid)` in
  every case.

---

## Summary of Files Changed

| File | Change |
|------|--------|
| `docs/matrix_shapes.md` | Hero diagram, notation table, param count, accuracy fixes |
| `src/hero/evaluate_cold_start.py` | Top-5 fallback for qualitative rescue examples |
| `src/data/dataset.py` | Customer ID in batch dict (all modes) |

---

> These refinements preserve Nishant's design intent while aligning with the codebase,
> `config.yaml` conventions, and the project plan requirements.
