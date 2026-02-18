# Decision Log — Villain Baseline
### CS7180 Seeing the Unseen | Author: Ishan Biswas

> This document records the *why* behind key design choices in the Villain
> baseline, as required by the CS7180 rubric and `rules.md §7`.

---

## D1. Why a Position-Biased Baseline?

**Decision:** The Villain deliberately includes a learnable `pop_bias` vector
that multiplies into the final logits, boosting popular items.

**Rationale (from `plan.md`):** Real e-commerce recommender systems exhibit
strong popularity bias — the rich get richer while niche products are buried.
By *intentionally* building this bias into the Villain, we create a measurable
"injustice" that the Hero model must fix with visual features. This makes the
multi-objective Pareto study meaningful: we can quantify exactly how much
accuracy the Hero trades for improved tail-item coverage.

**Evidence:** After training, 76.3% of the Villain's top-12 recommendations go
to head items (top 10% of catalog), while only 2.2% go to tail items (bottom
50%). The learned `pop_bias` values range from 0.83 to 2.08, confirming the
model actively amplifies popularity.

---

## D2. SASRec Architecture over Simpler Alternatives

**Decision:** Use SASRec (Transformer-based) rather than GRU4Rec or
matrix factorisation.

**Rationale:**
- SASRec captures long-range dependencies in purchase sequences via
  self-attention, unlike GRUs which struggle with distant items.
- The causal attention mask ensures the model only attends to past items
  (no future leakage).
- The dot-product prediction head (hidden × item\_embedding^T) is
  parameter-efficient — no separate output layer needed.

**Trade-off:** SASRec is heavier than GRU4Rec, but with 4.1M parameters
and a 12GB GPU, training completes in ~5 minutes. Acceptable.

---

## D3. Leave-One-Out Temporal Split

**Decision:** Last item → test, second-to-last → val, rest → train
(with sliding-window augmentation for training).

**Rationale:**
- Standard protocol for sequential recommendation (Kang & McAuley 2018).
- Temporal ordering is preserved — no random splits that would leak future
  information into training.
- Sliding window on the training prefix gives more training samples from
  each user, important given our relatively small sampled dataset.

**Alternative considered:** Chronological split by date. Rejected because
it would create very uneven split sizes and lose users who only shopped
in one time window.

---

## D4. Stratified Sampling Strategy

**Decision:** Sample 15% of users stratified by popularity-bin of their
most-purchased item category, with temporal pruning to the last 12 weeks.

**Rationale:**
- Random user sampling would over-represent head-item-only users and
  under-represent users who explore tail items.
- Stratifying by dominant purchase bin preserves the head/torso/tail
  distribution ratio.
- 12-week temporal window keeps sequences fresh and relevant while being
  large enough for meaningful histories.

**Result:** 430,879 transactions, 36,234 users, 26,932 articles.
Long-tail articles (< 10 purchases): ~85% of catalog — realistic.

---

## D5. Cross-Entropy over BPR Loss

**Decision:** Use cross-entropy (CE) loss over the full item vocabulary
instead of BPR pairwise loss.

**Rationale:**
- With 26,932 items, full-vocab CE is computationally feasible (unlike
  100k+ catalogs where sampled softmax is needed).
- CE directly optimises for ranking across all items simultaneously,
  while BPR only compares one positive vs one negative per sample.
- CE gives us a clean loss signal for monitoring convergence.

**Trade-off:** CE loss values are large (8–10) because log(26932) ≈ 10.2.
This is expected and does not indicate poor training.

---

## D6. AdamW + ReduceLROnPlateau

**Decision:** AdamW (lr=0.001, weight\_decay=0.01) with ReduceLROnPlateau
(factor=0.5, patience=3).

**Rationale:**
- AdamW decouples weight decay from gradient updates, leading to better
  generalisation than standard Adam.
- ReduceLROnPlateau is reactive: it only reduces LR when validation nDCG
  plateaus, avoiding premature decay.

**Observed:** LR dropped 0.001 → 0.0005 → 0.00025 over 14 epochs.
Model peaked at epoch 4, early-stopped at epoch 14 (patience=10).

---

## D7. Checkpoint & Resume Strategy

**Decision:** Maintain two checkpoints:
- `villain_best.pt` — saved on new best val nDCG@12
- `villain_latest.pt` — saved every 5 epochs for crash recovery

**Rationale:** Best-model checkpointing ensures the final test evaluation
uses the model at peak generalization (not the overfit final epoch). The
periodic latest checkpoint enables overnight training recovery.
