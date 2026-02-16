# Seeing the Unseen: Multi-Objective Rescue of the Fashion Long Tail
### CS7180 — Applied Deep Learning | Spring 2026 *(title pending confirmation)*

---

## Team
| Member    | Role |
|-----------|------|
| **Ishan**     | TBD  |
| **Elizabeth**  | TBD  |
| **Nishant**   | TBD  |

---

## Overview
This project tackles the **long-tail distribution problem** in fashion recommendation
using the H&M Personalized Fashion dataset (~31M transactions, ~105K product images).

We compare a text-only sequential baseline (the **"Villain"**) against a multimodal
Behavior Sequence Transformer with an Attribute-Aware Contrastive Learning head
(the **"Hero"**), and evaluate both on a multi-objective metric suite.

---

## Directory Structure

```
ADLProject1/
│
├── README.md                         # ← You are here
├── run_all.py                        # Master script: data → train → evaluate → report
├── requirements.txt                  # Pip-installable dependencies
├── config.yaml                       # Central hyper-parameters & paths
├── .gitignore                        # Ignore large data files, .venv, checkpoints
│
├── data/
│   ├── raw/                          # Symlinks / pointers to the H&M CSVs
│   │   └── README.md
│   ├── sampled/                      # Memory-friendly subsets for local dev
│   │   └── README.md
│   └── embeddings/                   # Pre-computed ResNet50 visual embeddings
│       └── README.md
│
├── src/
│   ├── __init__.py
│   ├── data/                         # Data loading, sampling, embedding extraction
│   │   ├── __init__.py
│   │   ├── dataset.py                # PyTorch Dataset / DataLoader definitions
│   │   ├── sampler.py                # Stratified long-tail sampler
│   │   └── embeddings.py             # ResNet50 feature extraction pipeline
│   │
│   ├── villain/                      # Baseline: position-aware sequential model
│   │   ├── __init__.py
│   │   ├── model.py                  # SASRec / ELO-ranking backbone (text-only)
│   │   ├── trainer.py                # Training loop for the villain
│   │   └── config.py                 # Villain-specific hyperparameters
│   │
│   ├── hero/                         # Main model: multimodal BST + contrastive head
│   │   ├── __init__.py
│   │   ├── model.py                  # BST encoder + Attribute-Aware CL head
│   │   ├── trainer.py                # Training loop for the hero
│   │   ├── contrastive.py            # Contrastive loss & hard-negative mining
│   │   └── config.py                 # Hero-specific hyperparameters
│   │
│   └── utils/                        # Shared utilities
│       ├── __init__.py
│       ├── metrics.py                # nDCG@12, MRR, Catalog Coverage
│       └── helpers.py                # Logging, seeding, device selection, etc.
│
├── analytics/
│   ├── metrics/                      # Evaluation outputs & metric logs
│   │   └── README.md
│   └── pareto/                       # Pareto trade-off study artifacts
│       └── README.md
│
├── notebooks/                        # Exploratory & presentation notebooks
│   └── 01_eda.ipynb                  # (placeholder) Initial data exploration
│
├── docs/
│   └── matrix_shapes.md              # Matrix shape transformation documentation
│
├── checkpoints/                      # Saved model weights (git-ignored)
│   └── .gitkeep
│
├── outputs/                          # Predictions, submission CSVs, figures
│   └── .gitkeep
│
├── h-and-m-personalized-fashion-recommendations/   # Raw H&M dataset (already present)
│   ├── articles.csv
│   ├── customers.csv
│   ├── transactions_train.csv
│   ├── sample_submission.csv
│   └── images/
│
└── .venv/                            # Python 3.13 virtual environment (already created)
```

---

## Quick Start
```bash
# 1. Activate virtual environment
.venv\Scripts\Activate.ps1          # Windows PowerShell

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline
python run_all.py --config config.yaml
```

---

## Multi-Objective Evaluation
| Metric             | Description                                     |
|--------------------|-------------------------------------------------|
| **nDCG@12**        | Normalized Discounted Cumulative Gain at rank 12 |
| **MRR**            | Mean Reciprocal Rank                             |
| **Catalog Coverage** | Fraction of catalog surfaced in recommendations |

See `analytics/pareto/` for the Pareto trade-off study between these objectives.
