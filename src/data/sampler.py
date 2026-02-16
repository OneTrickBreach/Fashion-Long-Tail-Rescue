"""
sampler.py — Stratified Long-Tail Sampler
==========================================
Team member: Ishan Biswas
Key functions: create_sample, analyze_distribution

PURPOSE:
    Creates a memory-friendly subset of the full H&M dataset while preserving
    the long-tail distribution of item popularity. This is essential because
    training on the full 31M-row dataset is infeasible on a single local PC.

SAMPLING STRATEGY:
    1. Temporal pruning  — keep only the last N weeks of transactions.
    2. Stratified sample — bin articles by popularity (head / torso / tail),
       sample a fraction of *users* proportionally from each bin to preserve
       the long-tail shape.
    3. User filtering    — drop users with fewer than `min_interactions` txns.
    4. Long-tail label   — flag articles with < `long_tail_threshold` purchases.

    Configurable via `config.yaml → sampling.*`.
"""

import os
import logging
import pandas as pd
import numpy as np

from src.utils.helpers import load_config, setup_logging

logger = logging.getLogger("seeing-the-unseen")


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def create_sample(config: dict) -> None:
    """
    Read raw data, apply stratified sampling, and save to data/sampled/.

    Steps:
        1. Load transactions in chunks, apply temporal pruning.
        2. Compute per-article purchase counts → popularity bins.
        3. Stratified-sample *users* so head/torso/tail ratios are preserved.
        4. Filter out low-activity users (< min_interactions).
        5. Label long-tail articles.
        6. Save sampled transactions, articles, and customers CSVs.
        7. Print distribution analysis.

    Args:
        config (dict): Parsed config.yaml.
    """
    setup_logging()
    paths = config["paths"]
    sampling_cfg = config["sampling"]
    seed = config["project"]["seed"]
    raw_dir = paths["raw_data"]
    out_dir = paths["sampled_data"]
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. Temporal pruning: load transactions and keep last N weeks ─────
    logger.info("Loading transactions with temporal pruning …")
    txn_path = os.path.join(raw_dir, "transactions_train.csv")
    temporal_weeks = sampling_cfg.get("temporal_weeks", 6)

    txn = _load_transactions_chunked(txn_path, temporal_weeks)
    logger.info(f"  After temporal pruning ({temporal_weeks} wk): {len(txn):,} rows")

    # ── 2. Popularity bins ───────────────────────────────────────────────
    article_counts = txn["article_id"].value_counts()
    long_tail_thr = sampling_cfg.get("long_tail_threshold", 10)

    # Head  = top 10 % of articles by purchase count
    # Torso = next 40 %
    # Tail  = bottom 50 % (includes articles below long-tail threshold)
    bins = _assign_popularity_bins(article_counts)

    # Map each transaction to its article's bin
    txn = txn.merge(
        bins.rename("pop_bin"),
        left_on="article_id",
        right_index=True,
        how="left",
    )

    # ── 3. Stratified user sampling ──────────────────────────────────────
    fraction = sampling_cfg.get("fraction", 0.05)
    min_inter = sampling_cfg.get("min_interactions", 5)

    logger.info(f"Stratified user sampling (frac={fraction}) …")
    sampled_txn = _stratified_user_sample(txn, fraction, min_inter, seed)
    logger.info(f"  Sampled transactions: {len(sampled_txn):,} rows")

    # ── 4. Compact dtypes (RAM savings) ──────────────────────────────────
    sampled_txn["article_id"] = sampled_txn["article_id"].astype(np.int32)
    if "sales_channel_id" in sampled_txn.columns:
        sampled_txn["sales_channel_id"] = sampled_txn["sales_channel_id"].astype(np.int8)

    # ── 5. Filter articles & customers to only those in sampled txns ─────
    kept_articles = sampled_txn["article_id"].unique()
    kept_customers = sampled_txn["customer_id"].unique()

    logger.info("Loading articles and customers …")
    articles = pd.read_csv(os.path.join(raw_dir, "articles.csv"))
    customers = pd.read_csv(os.path.join(raw_dir, "customers.csv"))

    articles = articles[articles["article_id"].isin(kept_articles)].copy()
    customers = customers[customers["customer_id"].isin(kept_customers)].copy()

    # ── 6. Long-tail labeling ────────────────────────────────────────────
    purchase_counts = sampled_txn.groupby("article_id").size()
    articles["purchase_count"] = (
        articles["article_id"].map(purchase_counts).fillna(0).astype(int)
    )
    articles["is_long_tail"] = articles["purchase_count"] < long_tail_thr
    logger.info(
        f"  Long-tail articles (<{long_tail_thr} purchases): "
        f"{articles['is_long_tail'].sum():,} / {len(articles):,} "
        f"({articles['is_long_tail'].mean():.1%})"
    )

    # ── 7. Save ──────────────────────────────────────────────────────────
    # Drop the helper column before saving
    sampled_txn = sampled_txn.drop(columns=["pop_bin"], errors="ignore")

    txn_out = os.path.join(out_dir, "transactions_sampled.csv")
    art_out = os.path.join(out_dir, "articles_sampled.csv")
    cst_out = os.path.join(out_dir, "customers_sampled.csv")

    sampled_txn.to_csv(txn_out, index=False)
    articles.to_csv(art_out, index=False)
    customers.to_csv(cst_out, index=False)
    logger.info(f"  Saved → {txn_out}  ({os.path.getsize(txn_out)/1e6:.1f} MB)")
    logger.info(f"  Saved → {art_out}  ({os.path.getsize(art_out)/1e6:.1f} MB)")
    logger.info(f"  Saved → {cst_out}  ({os.path.getsize(cst_out)/1e6:.1f} MB)")

    # ── 8. Distribution analysis ─────────────────────────────────────────
    analyze_distribution(sampled_txn, articles)


def analyze_distribution(txn: pd.DataFrame, articles: pd.DataFrame) -> None:
    """
    Print item frequency distributions to verify long-tail shape is preserved.

    Args:
        txn: Sampled transactions DataFrame.
        articles: Sampled articles DataFrame (with is_long_tail column).
    """
    counts = txn.groupby("article_id").size().sort_values(ascending=False)
    total_items = len(counts)
    total_txns = len(txn)

    # Head / Torso / Tail breakdown by the 80-20 rule
    cumsum = counts.cumsum()
    head_cutoff = (cumsum <= total_txns * 0.80).sum()
    torso_cutoff = (cumsum <= total_txns * 0.95).sum()
    tail_count = total_items - torso_cutoff

    logger.info("─── Distribution Analysis ───")
    logger.info(f"  Total unique articles : {total_items:,}")
    logger.info(f"  Total transactions    : {total_txns:,}")
    logger.info(f"  Head  (80% of txns)   : {head_cutoff:,} articles ({head_cutoff/total_items:.1%})")
    logger.info(f"  Torso (next 15%)      : {torso_cutoff - head_cutoff:,} articles ({(torso_cutoff - head_cutoff)/total_items:.1%})")
    logger.info(f"  Tail  (last 5%)       : {tail_count:,} articles ({tail_count/total_items:.1%})")

    # Percentile stats
    logger.info(f"  Median purchases/item : {int(counts.median())}")
    logger.info(f"  Mean  purchases/item  : {counts.mean():.1f}")
    logger.info(f"  Max   purchases/item  : {int(counts.max())}")
    logger.info(f"  Min   purchases/item  : {int(counts.min())}")

    # Long-tail stats from articles df
    if "is_long_tail" in articles.columns:
        n_lt = articles["is_long_tail"].sum()
        logger.info(f"  Flagged long-tail     : {n_lt:,} / {len(articles):,} ({n_lt/len(articles):.1%})")
    logger.info("─────────────────────────────")


# ──────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────

def _load_transactions_chunked(
    path: str,
    temporal_weeks: int,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """
    Read the large transactions CSV in chunks, keeping only the last
    `temporal_weeks` weeks of data.  This avoids loading the full 3.5 GB
    file into memory at once.

    Returns:
        pd.DataFrame with columns: t_dat, customer_id, article_id, price,
        sales_channel_id — filtered to the recent window.
    """
    # First pass: find the maximum date by scanning chunk-by-chunk
    max_date = pd.Timestamp.min
    for chunk in pd.read_csv(path, chunksize=chunksize, parse_dates=["t_dat"]):
        chunk_max = chunk["t_dat"].max()
        if chunk_max > max_date:
            max_date = chunk_max

    cutoff = max_date - pd.Timedelta(weeks=temporal_weeks)
    logger.info(f"  Date range: cutoff={cutoff.date()} → max={max_date.date()}")

    # Second pass: keep only rows after the cutoff
    frames = []
    for chunk in pd.read_csv(path, chunksize=chunksize, parse_dates=["t_dat"]):
        filtered = chunk[chunk["t_dat"] >= cutoff]
        if len(filtered) > 0:
            frames.append(filtered)

    return pd.concat(frames, ignore_index=True)


def _assign_popularity_bins(article_counts: pd.Series) -> pd.Series:
    """
    Assign each article to a popularity bin based on its purchase count
    percentile rank.

    Returns:
        pd.Series indexed by article_id with values 'head', 'torso', 'tail'.
    """
    ranks = article_counts.rank(pct=True)
    bins = pd.Series(index=article_counts.index, dtype="object")
    bins[ranks >= 0.90] = "head"
    bins[(ranks >= 0.50) & (ranks < 0.90)] = "torso"
    bins[ranks < 0.50] = "tail"
    return bins


def _stratified_user_sample(
    txn: pd.DataFrame,
    fraction: float,
    min_interactions: int,
    seed: int,
) -> pd.DataFrame:
    """
    Sample a fraction of *users* stratified by the popularity bin of their
    most-purchased article category.  Then filter out users who end up with
    fewer than `min_interactions` transactions in the sample.

    This preserves the head/torso/tail ratio better than random row sampling.
    """
    rng = np.random.RandomState(seed)

    # Determine each user's "dominant bin" (the bin they purchase from most)
    user_bins = (
        txn.groupby(["customer_id", "pop_bin"])
        .size()
        .reset_index(name="cnt")
        .sort_values("cnt", ascending=False)
        .drop_duplicates(subset="customer_id", keep="first")
        .set_index("customer_id")["pop_bin"]
    )

    sampled_users = []
    for bin_name in ["head", "torso", "tail"]:
        users_in_bin = user_bins[user_bins == bin_name].index.values
        n_sample = max(1, int(len(users_in_bin) * fraction))
        chosen = rng.choice(users_in_bin, size=n_sample, replace=False)
        sampled_users.append(chosen)
        logger.info(f"    {bin_name}: sampled {n_sample:,} / {len(users_in_bin):,} users")

    sampled_users = np.concatenate(sampled_users)
    sampled_txn = txn[txn["customer_id"].isin(set(sampled_users))].copy()

    # Enforce minimum interactions
    user_counts = sampled_txn["customer_id"].value_counts()
    active_users = user_counts[user_counts >= min_interactions].index
    sampled_txn = sampled_txn[sampled_txn["customer_id"].isin(active_users)]
    logger.info(f"    After min_interactions≥{min_interactions}: {sampled_txn['customer_id'].nunique():,} users")

    return sampled_txn.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────
# CLI entry point (for standalone runs)
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = load_config()
    create_sample(cfg)
