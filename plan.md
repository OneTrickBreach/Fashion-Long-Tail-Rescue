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

### Elizabeth: Memory-Efficient Preprocessing

* **Memory Pivot:** Convert 31M transactions into `int32` format (reduces RAM usage by 2.5x).
* **Temporal Pruning:** Focus on the **last 4-6 weeks** of H&M transactions to reflect current trends and keep training manageable.
* **Labeling:** Identify the "Long Tail" articles (those with <10 total purchases in history) for the discovery study.

### [Architecture] Ishan: Local Environment & Custom Baseline

* **Environment:** Setup local GPU environment (PyTorch/CUDA).
* **The Villain Build:** Code Iteration 1—a **Custom ELO Ranker**. This model scores items based on `Click-Rate / Position-Weight`, ensuring popular items stay at the top and niche items stay buried.
* **Decision Log:** Document why we chose a position-biased baseline to simulate real-world e-commerce bias.



### [KPIs] Nishant: The Discovery Gap Dashboard

* **KPI Metrics:** Setup evaluation scripts for **nDCG@12** and **Mean Reciprocal Rank (MRR)**.
* **EDA:** Create the "Visibility Skew" chart showing that the bottom 90% of the catalog gets 0% of the revenue in the baseline.

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