This is a comprehensive `plan.md` tailored for your team and optimized for a local, high-power PC. It incorporates Steve Schmidt's latest feedback—focusing on the **Long Tail**, ignoring **accidental clicks**, and prioritizing **modeling/cost functions** over front-end work.

# Project Plan: Seeing the Unseen - Multi-Objective Rescue of the Fashion Long Tail

## 1. Project Overview

* **The Problem:** Global fashion platforms suffer from the "Long-Tail Gap." Standard models prioritize popular items, burying niche products (90% of the catalog) that often match a user's visual style better but have zero interaction history.


* **The Narrative:**
* **Iteration 1 (The Villain):** A custom-built **Position-Aware ELO (Bubblesort)** baseline that is blind to visual style and only rewards high-volume interactions.


* **Iteration 2 (The Hero):** A **Behavior Sequence Transformer (BST)** augmented with **ResNet50 visual embeddings** to "see" niche style.


* **Iteration 3 (The Brain):** A **Multi-Objective re-ranker** that balances general relevance with catalog discovery (The "Million Dollar Couch" trade-off).





---

## 2. Phase 1: Data Engineering & Custom Baseline (Feb 15 – Feb 21)

**Goal:** Establish a fair "Villain" to one-up later and calibrate for local hardware.

> **Status: ✅ COMPLETE** — All Phase 1 deliverables finished, merged to `main` on March 1, 2026.

### Elizabeth: Memory-Efficient Preprocessing — ✅ DONE

* ✅ **Memory Pivot:** `src/data/sampler.py::convert_full_transactions_to_int32()` — 2-pass chunked conversion with customer ID mapping (int32/float32 dtypes).
* ⚠️ **Temporal Pruning:** Not implemented in Elizabeth's converter, but **covered** by Ishan's `create_sample()` function (temporal_weeks configurable via `config.yaml`).
* ✅ **Labeling:** `src/data/sampler.py::enrich_articles_with_sales_columns()` — enriches articles with total/online/in-store sales, popularity logit smoothing, first sale dates. Long-tail `is_long_tail` boolean added by Ishan's sampler.
* ✅ **Bonus:** Full evaluation metrics suite in `src/utils/metrics.py` — nDCG@12, MRR, catalog coverage, plus multi-objective metrics (popularity logits, tail-item rate).

### [Architecture] Ishan: Local Environment & Custom Baseline — ✅ DONE

* ✅ **Environment:** PyTorch 2.10+cu128, RTX 5070 Ti (11.9 GB VRAM), all packages verified.
* ✅ **The Villain Build:** `src/villain/model.py::VillainModel` — SASRec variant with learnable `pop_bias` vector, 4.1M parameters. Trained to convergence on sampled data.
* ✅ **Decision Log:** `docs/decision_log.md` — 7 design decisions documented (position bias rationale, SASRec choice, leave-one-out split, stratified sampling, CE loss, AdamW optimizer, checkpoint strategy).
* ✅ **Baseline Results:** nDCG@12=0.145, MRR=0.132, 76.3% of recommendations → head items, 2.2% → tail items. Saved to `outputs/villain_baseline_results.json`.
* ✅ **Additional:** Data sampling pipeline (`create_sample()`), PyTorch Dataset & DataLoaders (`dataset.py`), standalone evaluator (`evaluate.py`), matrix shapes documentation (`docs/matrix_shapes.md`).

### [KPIs] Nishant: The Discovery Gap Dashboard — ✅ DONE (partial)

* ⚠️ **KPI Metrics:** nDCG@12 and MRR evaluation scripts were **not implemented by Nishant**. Covered by Elizabeth (`metrics.py`) and Ishan (integration in `trainer.py`/`evaluate.py`).
* ✅ **EDA:** `src/utils/EDA.py::generate_visibility_skew_chart()` — generates the "Visibility Skew" chart. Output: `analytics/metrics/EDA_Long_tail_phase1.png`.

---

## 3. Phase 2: The "Hero" Model – Style & Sequential Intent (Feb 22 – Feb 28)

**Goal:** Implement the "Sight-Enabled" architecture that rescues the long tail.

### [Vision] Elizabeth: Multimodal Feature Bank

* **ResNet extraction:** Pass the 105,000 H&M images through a frozen **ResNet50** to get  vectors.
* **Multimodal Fusion:** Merge these visual vectors with the article metadata (color, fabric) in `articles.csv`.

### [Architecture] Ishan: Behavior Sequence Transformer (BST)

* **Iteration 2 Build:** Build the **Sequential Transformer**. Input shape is  where  is history and  is the fused visual/ID embedding.


* **Cold Start Simulation:** Steve’s requirement: Simulate a "New Product" (e.g., a "Kirkland" version) using only its image vector to see where the Hero model ranks it vs. the Baseline.



### [Visuals] Nishant: Matrix Mechanics & Qualitative Rescue

* **Matrix Shapes:** Design the layer-by-layer diagram. Show how  matrices () interact during a forward pass.
* **The "Aha!" Moment:** Find one user and one niche item. Prove the Villain ranked it at #500+, but the Hero model brought it to the Top 12.



---

## 4. Phase 3: Multi-Objective Study (March 1 – March 6)

**Goal:** Prove the business trade-off between CTR and Discovery.

### [Loss Functions] Ishan: Pareto Optimization

* **Iteration 3 Build:** Implement the **Multi-Objective Loss**:


* **Experiments:** Run 3 versions () to find the balance.



### Elizabeth: Ablation & Comparison

* **Ablation Study:** Quantify the "Visual Lift." How much did adding images help discovery vs. just using the ID sequence?
* **Final Script:** Finalize `run_all.py` so the whole system can be reproduced in one command.



### [Conclusions] Nishant: The Pareto Curve & Final Story

* **Pareto Front Visual:** Build a plot where nDCG is the Y-axis and **Catalog Coverage** is the X-axis.


* **Business Conclusion:** "By sacrificing 10% relevance, we gained 30% nicheness." This fulfills the "Incredible Conclusion" rubric requirement.



---

## 5. Phase 4: Final Submission & Delivery (March 7 – March 9)

**Goal:** Prepare for the 15-minute presentation.

* **Checklist:**
* [ ] **"Letters never go in":** Ensure all diagrams show embeddings, not text categories.


* [ ] **"DFS > BFS":** Verify we didn't add more datasets but went deep into the H&M visual/seq trade-off.


* [ ] **Recursion:** Clearly explain how we "one-upped" the Bubblesort Villain.


* [ ] **Matrix Detail:** Label shapes (e.g., ) on every layer diagram.





## 6. Task Distribution Summary

| Team Member | Core Focus | Key Tools |
| --- | --- | --- |
| **Ishan Biswas** | Transformer Architect & Loss Tuning | PyTorch, Local GPU, NumPy |
| **Elizabeth Coquillette** | Custom Baseline & Visual Features | Pandas, ResNet50, Scikit-learn |
| **Nishant Suresh** | Evaluation Dashboard & Presentation Visuals | Matplotlib, Plotly, LaTeX |

**Next Immediate Action:** Ishan to initialize the Git repo using the "Seeing the Unseen" directory structure and Elizabeth to start the `int32` data compression script.