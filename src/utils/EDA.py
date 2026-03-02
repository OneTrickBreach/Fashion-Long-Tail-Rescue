"""
EDA.py — Visibility Skew Analysis
===================================
Team member: Nishant Suresh
Key functions: generate_visibility_skew_chart

PURPOSE:
    Generates a visibility skew chart on the H&M dataset to visualize the
    Long-Tail Problem. The chart shows that a small fraction of popular items
    accounts for the vast majority of purchases, while most catalog items
    receive negligible attention.

USAGE:
    python -m src.utils.EDA
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.helpers import load_config


def generate_visibility_skew_chart(config: dict, save_path: str | None = None) -> None:
    """
    Generate the Visibility Skew chart showing the Long-Tail Gap.

    Args:
        config: Parsed config.yaml dict.
        save_path: Optional path to save the chart PNG. If None, displays
                   interactively via plt.show().
    """
    raw_dir = config["paths"]["raw_data"]
    articles_path = os.path.join(raw_dir, "articles_enriched.csv")

    # Fallback: check data/raw/ if not found in raw_dir
    if not os.path.exists(articles_path):
        articles_path = os.path.join("data", "raw", "articles_enriched.csv")

    articles = pd.read_csv(articles_path)

    # Sort by total_sales descending (most popular first)
    articles = articles.sort_values("total_sales", ascending=False).reset_index(drop=True)

    # Compute cumulative purchase share
    total_purchases = articles["total_sales"].sum()
    articles["cumulative_share"] = articles["total_sales"].cumsum() / total_purchases

    # Compute catalog percentile (x-axis)
    articles["catalog_percentile"] = (articles.index + 1) / len(articles) * 100

    # Long-tail label (articles with < 10 total purchases)
    long_tail_threshold = config.get("sampling", {}).get("long_tail_threshold", 10)
    long_tail_count = (articles["total_sales"] < long_tail_threshold).sum()
    long_tail_pct = long_tail_count / len(articles) * 100
    long_tail_purchase_share = (
        articles.loc[articles["total_sales"] < long_tail_threshold, "total_sales"].sum()
        / total_purchases * 100
    )

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        articles["catalog_percentile"],
        articles["cumulative_share"] * 100,
        color="#E63946",
        linewidth=2,
        label="Cumulative Purchase Share"
    )

    # Perfect equality line for reference
    ax.plot([0, 100], [0, 100], color="gray", linestyle="--", linewidth=1, label="Perfect Equality")

    # Shade the long-tail region
    ax.axvspan(
        100 - long_tail_pct, 100, alpha=0.15, color="#457B9D",
        label=f"Long-Tail Articles (<{long_tail_threshold} purchases)"
    )

    # Annotation
    ax.annotate(
        f"{long_tail_pct:.1f}% of catalog\ngets {long_tail_purchase_share:.1f}% of purchases",
        xy=(100 - long_tail_pct / 2, 65),
        fontsize=10,
        color="#457B9D",
        ha="center"
    )

    ax.set_xlabel("Catalog Percentile (ranked by popularity)", fontsize=12)
    ax.set_ylabel("Cumulative % of Total Purchases", fontsize=12)
    ax.set_title("Visibility Skew: The Long-Tail Gap", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved chart to: {save_path}")
    else:
        plt.show()

    # Print summary stats
    print(f"Total articles:          {len(articles):,}")
    print(f"Total purchases:         {total_purchases:,}")
    print(f"Long-tail articles:      {long_tail_count:,} ({long_tail_pct:.1f}% of catalog)")
    print(f"Long-tail purchase share:{long_tail_purchase_share:.2f}% of all purchases")


if __name__ == "__main__":
    cfg = load_config()
    generate_visibility_skew_chart(
        cfg,
        save_path=os.path.join("analytics", "metrics", "EDA_Long_tail_phase1.png"),
    )