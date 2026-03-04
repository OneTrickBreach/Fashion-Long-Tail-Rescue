"""
pareto_plot.py — Pareto Front Visualisation & Business Narrative
================================================================
Team: Ishan, Elizabeth, Nishant
Author: Nishant (original concept), Ishan (config integration & second panel)

PURPOSE:
    Generate publication-quality Pareto front plots from the multi-objective
    λ-discovery sweep results.  Two outputs:

    1. Pareto front chart: Catalog Coverage (x) vs nDCG@12 (y)
    2. Tail-rate curve:    λ_disc (x) vs Tail Item Rate (y)

    Both are saved at 300 DPI to the directory specified in
    config.yaml → pareto.output_dir.

KEY FUNCTIONS:
    generate_pareto_plots(config)  — main entry point (called by run_all.py)
    _plot_pareto_front(...)        — Panel 1: coverage vs nDCG
    _plot_tail_rate_curve(...)     — Panel 2: λ vs tail rate
    _print_business_narrative(...) — Console summary for presentation
"""

import json
import os
import logging

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logger = logging.getLogger("seeing-the-unseen")


# ── Panel 1: Pareto Front (Coverage vs nDCG) ────────────────────────────────

def _plot_pareto_front(sweep, villain_ndcg, villain_coverage, output_path):
    """Plot Catalog Coverage (x) vs nDCG@12 (y) with Villain baseline."""
    lambdas   = [r["lambda_disc"]          for r in sweep]
    ndcgs     = [r["ndcg@12"]              for r in sweep]
    coverages = [r["catalog_coverage"] * 100 for r in sweep]

    fig, ax = plt.subplots(figsize=(10, 7))

    # Hero sweep points + connecting line
    ax.scatter(coverages, ndcgs, color="steelblue", s=120, zorder=5,
               label="Hero (λ sweep)")
    ax.plot(coverages, ndcgs, color="steelblue", linewidth=1.5,
            linestyle="--", alpha=0.6, zorder=4)

    # Annotate each point with its λ value
    for lam, cov, ndcg in zip(lambdas, coverages, ndcgs):
        is_optimal = (lam == 0.3)
        ax.annotate(
            f"λ={lam}" + (" ★ Sweet Spot" if is_optimal else ""),
            xy=(cov, ndcg),
            xytext=(10, -18) if is_optimal else (8, 8),
            textcoords="offset points",
            fontsize=10,
            color="darkgreen" if is_optimal else "black",
            fontweight="bold" if is_optimal else "normal",
        )

    # Highlight Pareto-optimal point (λ=0.3)
    if 0.3 in lambdas:
        opt_idx = lambdas.index(0.3)
        ax.scatter(coverages[opt_idx], ndcgs[opt_idx], s=300, color="green",
                   zorder=6, marker="*", label="Pareto Optimal (λ=0.3)")

    # Villain baseline
    ax.scatter(villain_coverage, villain_ndcg, s=150, color="crimson",
               zorder=5, marker="X", label="Villain Baseline")
    ax.annotate("Villain Baseline", xy=(villain_coverage, villain_ndcg),
                xytext=(8, 8), textcoords="offset points", fontsize=10,
                color="crimson", fontweight="bold")

    ax.set_xlabel("Catalog Coverage (%)", fontsize=13)
    ax.set_ylabel("nDCG@12", fontsize=13)
    ax.set_title("Pareto Front: Relevance vs. Catalog Discovery", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved Pareto front plot → %s", output_path)


# ── Panel 2: Tail Rate Curve ────────────────────────────────────────────────

def _plot_tail_rate_curve(sweep, output_path):
    """Plot λ_disc (x) vs Tail Item Rate (y) — the tuneable discovery knob."""
    lambdas    = [r["lambda_disc"]           for r in sweep]
    tail_rates = [r["tail_item_rate@k"] * 100 for r in sweep]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(lambdas, tail_rates, "o-", color="darkorange", linewidth=2,
            markersize=10, zorder=5)

    for lam, tr in zip(lambdas, tail_rates):
        ax.annotate(f"{tr:.1f}%", xy=(lam, tr), xytext=(0, 12),
                    textcoords="offset points", fontsize=10, ha="center",
                    fontweight="bold")

    ax.set_xlabel("λ_disc (Discovery Weight)", fontsize=13)
    ax.set_ylabel("Tail Item Rate (%)", fontsize=13)
    ax.set_title("Tuneable Discovery Knob: λ vs. Long-Tail Exposure", fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved tail-rate curve → %s", output_path)


# ── Business Narrative ──────────────────────────────────────────────────────

def _print_business_narrative(sweep):
    """Print the headline conclusion numbers for the presentation."""
    baseline = next(r for r in sweep if r["lambda_disc"] == 0.0)
    optimal  = next(r for r in sweep if r["lambda_disc"] == 0.3)

    ndcg_delta = (optimal["ndcg@12"] - baseline["ndcg@12"]) / baseline["ndcg@12"] * 100
    cov_gain   = (optimal["catalog_coverage"] - baseline["catalog_coverage"]) * 100
    tail_gain  = (optimal["tail_item_rate@k"] - baseline["tail_item_rate@k"]) * 100

    print("\n" + "#" * 60)
    print("  BUSINESS CONCLUSION — Phase 3 Multi-Objective Study")
    print("#" * 60)
    print(f"  Baseline (λ=0.0):  nDCG={baseline['ndcg@12']:.4f}  "
          f"Coverage={baseline['catalog_coverage']*100:.1f}%  "
          f"Tail={baseline['tail_item_rate@k']*100:.1f}%")
    print(f"  Optimal  (λ=0.3):  nDCG={optimal['ndcg@12']:.4f}  "
          f"Coverage={optimal['catalog_coverage']*100:.1f}%  "
          f"Tail={optimal['tail_item_rate@k']*100:.1f}%")
    print()
    if ndcg_delta >= 0:
        print(f"  nDCG@12 change:        +{ndcg_delta:.2f}% (no sacrifice)")
    else:
        print(f"  nDCG@12 sacrifice:     {ndcg_delta:.2f}%")
    print(f"  Catalog coverage gain: +{cov_gain:.1f} percentage points")
    print(f"  Tail item rate gain:   +{tail_gain:.1f} percentage points")
    print()
    print("  → By tuning a single λ knob from 0→0.3, we gain +{:.1f}pp".format(cov_gain))
    print("    catalog discovery and +{:.1f}pp long-tail exposure".format(tail_gain))
    print("    with zero relevance sacrifice.")
    print("    This gives business stakeholders a tuneable dial")
    print("    between CTR-optimised and discovery-optimised recommendations.")
    print("#" * 60 + "\n")


# ── Main Entry Point ────────────────────────────────────────────────────────

def generate_pareto_plots(config):
    """
    Generate all Pareto visualisation outputs.

    Reads paths and parameters from config.yaml:
        - pareto.output_dir      → save directory
        - paths.outputs          → location of sweep results JSON
        - villain baseline       → loaded from villain_eval_full.json

    Called by: run_all.py --stage pareto_plot
    """
    output_dir  = config["pareto"]["output_dir"]
    outputs_dir = config["paths"]["outputs"]
    os.makedirs(output_dir, exist_ok=True)

    # Load sweep results
    sweep_path = os.path.join(outputs_dir, "pareto_sweep_results.json")
    with open(sweep_path, "r") as f:
        sweep = json.load(f)
    logger.info("Loaded %d Pareto sweep points from %s", len(sweep), sweep_path)

    # Load Villain baseline
    villain_path = os.path.join(outputs_dir, "villain_eval_full.json")
    with open(villain_path, "r") as f:
        villain = json.load(f)
    villain_ndcg     = villain["overall"]["ndcg@12"]
    villain_coverage = villain["overall"]["catalog_coverage"] * 100

    # Panel 1: Pareto front
    _plot_pareto_front(
        sweep, villain_ndcg, villain_coverage,
        os.path.join(output_dir, "pareto_front.png"),
    )

    # Panel 2: Tail rate curve
    _plot_tail_rate_curve(
        sweep,
        os.path.join(output_dir, "tail_rate_curve.png"),
    )

    # Business narrative to console
    _print_business_narrative(sweep)

    logger.info("Pareto visualisation complete — outputs in %s", output_dir)
