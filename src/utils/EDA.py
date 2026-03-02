'''
EDA.py
Generates a visibility skew on the H&M dataset to show the Long-Tail Problem
'''

import pandas as pd
import matplotlib.pyplot as plt

#Load articles.csv int32 version
articles = pd.read_csv("../../data/raw/articles_enriched.csv")

#Sort by total_sales descending (most popular first) 
articles = articles.sort_values("total_sales", ascending=False).reset_index(drop=True)

#Compute cumulative purchase share
total_purchases = articles["total_sales"].sum()
articles["cumulative_share"] = articles["total_sales"].cumsum() / total_purchases

#Compute catalog percentile (x-axis)
articles["catalog_percentile"] = (articles.index + 1) / len(articles) * 100

#Long-tail label (articles with < 10 total purchases)
long_tail_count = (articles["total_sales"] < 10).sum()
long_tail_pct = long_tail_count / len(articles) * 100
long_tail_purchase_share = (articles.loc[articles["total_sales"] < 10, "total_sales"].sum() / total_purchases * 100)

#Plot
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
ax.axvspan(100 - long_tail_pct, 100, alpha=0.15, color="#457B9D", label=f"Long-Tail Articles (<10 purchases)")

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
plt.show()

#Print summary stats
print(f"Total articles:          {len(articles):,}")
print(f"Total purchases:         {total_purchases:,}")
print(f"Long-tail articles:      {long_tail_count:,} ({long_tail_pct:.1f}% of catalog)")
print(f"Long-tail purchase share:{long_tail_purchase_share:.2f}% of all purchases")