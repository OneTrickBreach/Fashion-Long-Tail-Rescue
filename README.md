# Seeing the Unseen: Multi-Objective Rescue of the Fashion Long Tail
### CS7180 — Applied Deep Learning | Spring 2026

---

## Team
| Member | Role |
|--------|------|
| **Ishan Biswas** | Transformer Architect & Loss Tuning (PyTorch, Local GPU) |
| **Elizabeth Coquillette** | Custom Baseline & Visual Features (Pandas, ResNet50) |
| **Nishant Suresh** | Evaluation Dashboard & Presentation Visuals (Matplotlib, Plotly) |

---

## Overview
This project tackles the **long-tail distribution problem** in fashion recommendation
using the H&M Personalized Fashion dataset (~31M transactions, ~105K product images).

We compare a text-only sequential baseline (the **"Villain"**) against a multimodal
Behavior Sequence Transformer with an Attribute-Aware Contrastive Learning head
(the **"Hero"**), and evaluate both on a multi-objective metric suite.

### The Narrative
1. **The Villain** — A position-aware SASRec baseline that is *intentionally blind* to product images and amplifies popularity bias via a learnable `pop_bias` vector. It buries the long tail.
2. **The Hero** — A Behavior Sequence Transformer (BST) fusing item-ID embeddings with ResNet50 visual embeddings, trained with a combined Cross-Entropy + InfoNCE contrastive loss. It rescues the long tail by learning visual similarity patterns.
3. **The Brain** *(Phase 3)* — A multi-objective re-ranker balancing relevance (nDCG) against catalog discovery (coverage).

---

## Progress

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Data Engineering & Custom Baseline (Villain) | ✅ Complete |
| **Phase 2** | The "Hero" Model — Style & Sequential Intent | ✅ Complete |
| **Phase 3** | Multi-Objective Pareto Study | ✅ Complete |
| **Phase 4** | Final Submission & Presentation | ✅ Complete |

### Key Results

| Metric | Villain | Hero (λ=0) | Hero (λ=0.3) | Hero (λ=0.7) |
|--------|---------|------------|--------------|---------------|
| **nDCG@12** | 0.145 | 0.132 | **0.133** | 0.085 |
| **Catalog Coverage** | 57.8% | 60.0% | **67.6%** | 74.6% |
| **Tail-Item Rate** | 2.2% | 4.2% | **6.6%** | 81.7% |
| **Cold-Start Rank** | 26,159 | — | — | — |

**Phase 3 headline:** λ=0.3 is the Pareto-optimal sweet spot — +7.6pp catalog coverage
and +2.3pp tail-item rate with *zero* nDCG sacrifice vs. the base Hero.

**Visual ablation:** ResNet50 embeddings provide minimal warm-item lift (+0.03% nDCG)
but dramatically improve cold-start ranking (+4,525 positions).

---

## Directory Structure

```
ADLProject1/
│
├── README.md                         # ← You are here
├── run_all.py                        # Master script: 10-stage pipeline (sample → ... → presentation)
├── requirements.txt                  # Pip-installable dependencies
├── config.yaml                       # Central hyper-parameters & paths (single source of truth)
├── plan.md                           # Project plan & task distribution
├── .gitignore
│
├── data/
│   ├── embeddings/                   # Pre-computed visual & multimodal embeddings
│   │   └── raw/
│   │       └── README.md
│   └── sampled/                      # Memory-friendly subsets for local dev
│
├── src/
│   ├── __init__.py
│   ├── data/                         # Data loading, sampling, embedding extraction
│   │   ├── __init__.py
│   │   ├── dataset.py                # PyTorch Dataset & DataLoaders (leave-one-out split)
│   │   ├── sampler.py                # Stratified long-tail sampler + int32 conversion
│   │   ├── embeddings.py             # Convenience wrappers for embedding pipelines
│   │   ├── extract_visual_embeddings.py  # ResNet50 visual feature extraction
│   │   └── fuse_multimodal_embeddings.py # Visual + metadata → fused embeddings
│   │
│   ├── villain/                      # Baseline: position-aware sequential model (text-only)
│   │   ├── __init__.py
│   │   ├── model.py                  # SASRec + learnable pop_bias vector
│   │   ├── trainer.py                # Training loop with checkpoint resume & early stopping
│   │   ├── evaluate.py               # Standalone evaluation with per-bucket breakdown
│   │   └── config.py                 # Villain-specific hyperparameter defaults
│   │
│   ├── hero/                         # Main model: multimodal BST + contrastive head
│   │   ├── __init__.py
│   │   ├── model.py                  # BST encoder with VisualProjection fusion
│   │   ├── trainer.py                # Training loop (CE + InfoNCE + Discovery loss)
│   │   ├── evaluate.py               # Standalone evaluation with tail-item analysis
│   │   ├── evaluate_cold_start.py    # Cold-start simulation (Villain vs Hero ranks)
│   │   ├── contrastive.py            # InfoNCE loss, MultiObjectiveLoss & hard-negative mining
│   │   ├── pareto_sweep.py           # Phase 3: λ-discovery fine-tune sweep
│   │   ├── ablation.py               # Phase 3: visual ablation study (ID-only Hero)
│   │   └── config.py                 # Hero-specific hyperparameter defaults
│   │
│   ├── phase3.py                     # Nishant's standalone Pareto plot script
│   ├── phase4_presentation.py        # Phase 4: presentation figure generation + narrative
│   │
│   └── utils/                        # Shared utilities
│       ├── __init__.py
│       ├── metrics.py                # nDCG@12, MRR, Catalog Coverage, tail-item rate
│       ├── helpers.py                # Config loading, seeding, device selection, logging
│       ├── pareto_plot.py            # Phase 3: Pareto front & tail-rate curve plots
│       └── EDA.py                    # Visibility skew chart (long-tail visualization)
│
├── scripts/
│   └── render_diagrams.py            # One-off: render ASCII diagrams to PNG
│
├── docs/
│   ├── matrix_shapes.md              # Full tensor shape documentation (Villain + Hero)
│   ├── architecture_diagrams.md      # Mermaid flowcharts for all 3 acts
│   ├── decision_log.md               # Design decisions D1–D13 with rationale
│   └── phase2_nishant_review.md      # Review & refinements to Nishant's Phase 2 work
│
├── analytics/
│   ├── metrics/                      # Evaluation outputs & metric logs
│   ├── pareto/                       # Pareto front plot + tail-rate curve (300 DPI)
│   └── presentation/                 # Slide-ready figures (model comparison, training curves, etc.)
│
├── notebooks/                        # Exploratory & presentation notebooks
│
├── checkpoints/                      # Saved model weights (git-ignored)
├── outputs/                          # Evaluation results JSON files
│
└── h-and-m-personalized-fashion-recommendations/   # Raw H&M dataset (git-ignored)
    ├── articles.csv
    ├── customers.csv
    ├── transactions_train.csv
    └── images/
```

---

## Quick Start
```bash
# 1. Activate virtual environment
.venv\Scripts\Activate.ps1          # Windows PowerShell

# 2. Install PyTorch with CUDA 12.8 (RTX 5070 Ti)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 3. Install remaining dependencies
pip install -r requirements.txt

# 4. Run the full pipeline (all 10 stages)
python run_all.py --config config.yaml

# Or run individual stages:
python run_all.py --stage sample          # 1. Data sampling
python run_all.py --stage embed           # 2. Visual embedding extraction + fusion
python run_all.py --stage train_villain   # 3. Train Villain baseline
python run_all.py --stage train_hero      # 4. Train Hero model
python run_all.py --stage evaluate        # 5-6. Evaluate both + cold-start analysis
python run_all.py --stage ablation        # 7. Visual ablation study (ID-only Hero)
python run_all.py --stage pareto_sweep    # 8. Multi-objective λ sweep
python run_all.py --stage pareto_plot     # 9. Generate Pareto front visualisation
python run_all.py --stage presentation    # 10. Generate presentation figures
```

---

## Architecture

### Villain (Baseline — Text-Only)
- **SASRec** with 3-layer Transformer encoder (128-dim, 4 heads)
- Learnable `pop_bias` vector amplifies popular items
- ~4.1M parameters | Batch size 256 | Max sequence length 50

### Hero (Main Model — Multimodal)
- **Behavior Sequence Transformer** with ResNet50 visual embeddings
- `VisualProjection`: Linear(2048→128) + LayerNorm + Dropout
- Element-wise fusion: item\_emb + pos\_emb + visual\_proj → TransformerEncoder
- **Phase 3 loss:** `L = L_CE + 0.3 × L_InfoNCE + λ_disc × L_discovery`
  - Discovery loss penalises softmax mass on popular items (tuneable via `λ_disc`)
  - Attribute-aware Jaccard hard-negative mining (replaces random negatives)
- ~4.1M parameters | Batch size 128 | Max sequence length 50

See `docs/matrix_shapes.md` for full tensor shape documentation through both models.

---

## Multi-Objective Evaluation
| Metric | Description |
|--------|-------------|
| **nDCG@12** | Normalized Discounted Cumulative Gain at rank 12 |
| **MRR** | Mean Reciprocal Rank |
| **Catalog Coverage** | Fraction of catalog surfaced in top-12 recommendations |
| **Tail-Item Rate** | Fraction of recommendations going to long-tail items (<10 purchases) |
| **Mean Tail Score** | Average inverse-popularity of recommended items |

See `docs/decision_log.md` for design rationale and `analytics/pareto/` for the Phase 3 trade-off study.
