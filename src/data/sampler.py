"""
sampler.py — Data Preprocessing & Stratified Long-Tail Sampler
===============================================================
Team: Elizabeth Coquillette (int32 conversion, article enrichment),
      Ishan Biswas (stratified sampling, distribution analysis)

PURPOSE:
    This module provides two complementary data preparation workflows:

    1. **Full int32 Conversion** (Elizabeth):
       - `convert_full_transactions_to_int32()` — convert all 31M transactions
         to compact int32 numeric format (2.5× RAM reduction).
       - `enrich_articles_with_sales_columns()` — augment articles.csv with
         total/online/in-store sales, popularity logit smoothing, etc.

    2. **Stratified Sampling** (Ishan):
       - `create_sample()` — temporal pruning + stratified user sampling to
         create a manageable (~5%) subset preserving the long-tail distribution.
       - `analyze_distribution()` — print item frequency statistics.

    Configurable via `config.yaml → sampling.*` and `config.yaml → paths.*`.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.helpers import load_config, setup_logging

logger = logging.getLogger("seeing-the-unseen")


# ══════════════════════════════════════════════════════════════
# Elizabeth's int32 Conversion & Article Enrichment
# ══════════════════════════════════════════════════════════════

def convert_full_transactions_to_int32(
    transactions_path: str | Path,
    output_csv_path: str | Path,
    customer_map_path: str | Path,
    chunksize: int = 2_000_000,
    keep_original_column_names: bool = False,
) -> dict:
    """
    Convert all rows in a transactions CSV to compact numeric columns and save as a new CSV.

    Output columns:
    - t_dat_days_int32
    - customer_id_int32
    - article_id_int32
    - price_float32
    - sales_channel_id_int32

    If keep_original_column_names=True, output columns are:
    - t_dat
    - customer_id
    - article_id
    - price
    - sales_channel_id
    """
    transactions_path = Path(transactions_path)
    output_csv_path = Path(output_csv_path)
    customer_map_path = Path(customer_map_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    customer_map_path.parent.mkdir(parents=True, exist_ok=True)

    if not transactions_path.exists():
        raise FileNotFoundError(f"transactions file not found: {transactions_path}")
    if chunksize <= 0:
        raise ValueError("chunksize must be > 0")

    # Pass 1: stable customer_id -> int32 mapping.
    unique_customers = set()
    for chunk in pd.read_csv(
        transactions_path,
        usecols=["customer_id"],
        dtype={"customer_id": "string"},
        chunksize=chunksize,
    ):
        unique_customers.update(chunk["customer_id"].dropna().unique().tolist())

    customer_ids = sorted(unique_customers)
    customer_map_df = pd.DataFrame({"customer_id": pd.Series(customer_ids, dtype="string")})
    customer_map_df["customer_id_int32"] = pd.Series(range(len(customer_map_df)), dtype="int32")
    customer_map_df.to_csv(customer_map_path, index=False)
    customer_map = dict(zip(customer_map_df["customer_id"], customer_map_df["customer_id_int32"]))

    # Pass 2: convert and stream-write.
    wrote_header = False
    rows_written = 0
    for chunk in pd.read_csv(
        transactions_path,
        usecols=["t_dat", "customer_id", "article_id", "price", "sales_channel_id"],
        dtype={
            "customer_id": "string",
            "article_id": "int32",
            "price": "float32",
            "sales_channel_id": "int16",
        },
        chunksize=chunksize,
    ):
        t_dat = pd.to_datetime(chunk["t_dat"], errors="coerce")
        out = pd.DataFrame(
            {
                "t_dat_days_int32": (t_dat - pd.Timestamp("1970-01-01")).dt.days.astype("Int32"),
                "customer_id_int32": chunk["customer_id"].map(customer_map).astype("Int32"),
                "article_id_int32": chunk["article_id"].astype("int32"),
                "price_float32": chunk["price"].astype("float32"),
                "sales_channel_id_int32": chunk["sales_channel_id"].astype("int32"),
            }
        )
        out = out.dropna(subset=["t_dat_days_int32", "customer_id_int32", "article_id_int32"])
        if keep_original_column_names:
            out = out.rename(
                columns={
                    "t_dat_days_int32": "t_dat",
                    "customer_id_int32": "customer_id",
                    "article_id_int32": "article_id",
                    "price_float32": "price",
                    "sales_channel_id_int32": "sales_channel_id",
                }
            )
        out.to_csv(
            output_csv_path,
            index=False,
            mode="w" if not wrote_header else "a",
            header=not wrote_header,
        )
        wrote_header = True
        rows_written += len(out)

    summary = {
        "transactions_file": str(transactions_path),
        "converted_csv": str(output_csv_path),
        "customer_map_csv": str(customer_map_path),
        "rows_written": int(rows_written),
        "unique_customers": int(len(customer_map_df)),
    }
    print(f"Saved converted CSV: {output_csv_path}")
    print(f"Saved customer map:  {customer_map_path}")
    return summary


def enrich_articles_with_sales_columns(
    articles_path: str | Path,
    transactions_path: str | Path,
    output_articles_path: str | Path,
    online_channel_id: int = 2,
    in_store_channel_id: int = 1,
    logit_alpha: float = 0.5,
    chunksize: int = 2_000_000,
) -> dict:
    """
    Create a new articles CSV with additional sales columns:
    - in_store_sales
    - online_sales
    - total_sales
    - first_sale_date
    - online_sales_pct
    - popularity_logit_smoothed
    """
    articles_path = Path(articles_path)
    transactions_path = Path(transactions_path)
    output_articles_path = Path(output_articles_path)
    output_articles_path.parent.mkdir(parents=True, exist_ok=True)

    if not articles_path.exists():
        raise FileNotFoundError(f"articles file not found: {articles_path}")
    if not transactions_path.exists():
        raise FileNotFoundError(f"transactions file not found: {transactions_path}")
    if chunksize <= 0:
        raise ValueError("chunksize must be > 0")
    if logit_alpha <= 0:
        raise ValueError("logit_alpha must be > 0")

    total_counts = pd.Series(dtype="int64")
    online_counts = pd.Series(dtype="int64")
    in_store_counts = pd.Series(dtype="int64")
    first_sale_dates: dict[int, pd.Timestamp] = {}

    for chunk in pd.read_csv(
        transactions_path,
        usecols=["article_id", "sales_channel_id", "t_dat"],
        dtype={"article_id": "int32", "sales_channel_id": "int16"},
        chunksize=chunksize,
    ):
        chunk = chunk.dropna(subset=["article_id"])

        total_chunk = chunk["article_id"].value_counts()
        total_counts = total_counts.add(total_chunk, fill_value=0)

        online_chunk = chunk.loc[chunk["sales_channel_id"] == online_channel_id, "article_id"].value_counts()
        if not online_chunk.empty:
            online_counts = online_counts.add(online_chunk, fill_value=0)

        in_store_chunk = chunk.loc[chunk["sales_channel_id"] == in_store_channel_id, "article_id"].value_counts()
        if not in_store_chunk.empty:
            in_store_counts = in_store_counts.add(in_store_chunk, fill_value=0)

        chunk["t_dat"] = pd.to_datetime(chunk["t_dat"], errors="coerce")
        date_min = chunk.dropna(subset=["t_dat"]).groupby("article_id", as_index=True)["t_dat"].min()
        for article_id, chunk_min_date in date_min.items():
            existing = first_sale_dates.get(int(article_id))
            if existing is None or chunk_min_date < existing:
                first_sale_dates[int(article_id)] = chunk_min_date

    articles = pd.read_csv(articles_path)
    total_counts = total_counts.astype("int64")
    online_counts = online_counts.astype("int64")
    in_store_counts = in_store_counts.astype("int64")

    articles["total_sales"] = articles["article_id"].map(total_counts).fillna(0).astype("int64")
    articles["online_sales"] = articles["article_id"].map(online_counts).fillna(0).astype("int64")
    articles["in_store_sales"] = articles["article_id"].map(in_store_counts).fillna(0).astype("int64")

    articles["online_sales_pct"] = np.divide(
        articles["online_sales"].to_numpy(dtype="float64"),
        articles["total_sales"].to_numpy(dtype="float64"),
        out=np.zeros(len(articles), dtype="float64"),
        where=articles["total_sales"].to_numpy() != 0,
    )
    articles["in_store_higher"] = (
        (articles["in_store_sales"] > articles["online_sales"]).astype("int8")
    )

    total_in_store = float(articles["in_store_sales"].sum())
    total_online = float(articles["online_sales"].sum())
    overall_in_store_to_online_ratio = (
        total_in_store / total_online if total_online > 0 else np.inf
    )
    item_ratio = np.divide(
        articles["in_store_sales"].to_numpy(dtype="float64"),
        articles["online_sales"].to_numpy(dtype="float64"),
        out=np.full(len(articles), np.inf, dtype="float64"),
        where=articles["online_sales"].to_numpy() != 0,
    )
    both_zero_mask = (
        (articles["in_store_sales"].to_numpy() == 0)
        & (articles["online_sales"].to_numpy() == 0)
    )
    item_ratio[both_zero_mask] = 0.0
    articles["in_store_dominant"] = (
        item_ratio > overall_in_store_to_online_ratio
    ).astype("int8")

    n_total_sales = float(articles["total_sales"].sum())
    m_items = float(len(articles))
    denom = n_total_sales + logit_alpha * m_items
    p = (articles["total_sales"].to_numpy(dtype="float64") + logit_alpha) / denom
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    articles["popularity_logit_smoothed"] = np.log(p / (1.0 - p))

    first_sale_series = pd.Series(
        {k: v.strftime("%Y-%m-%d") for k, v in first_sale_dates.items()},
        name="first_sale_date",
    )
    articles["first_sale_date"] = articles["article_id"].map(first_sale_series)

    articles.to_csv(output_articles_path, index=False)

    summary = {
        "articles_in": str(articles_path),
        "transactions_in": str(transactions_path),
        "articles_out": str(output_articles_path),
        "article_rows": int(len(articles)),
        "articles_with_sales": int((articles["total_sales"] > 0).sum()),
        "articles_without_sales": int((articles["total_sales"] == 0).sum()),
        "overall_in_store_to_online_ratio": float(overall_in_store_to_online_ratio),
        "logit_alpha": float(logit_alpha),
    }
    print(f"Saved enriched articles CSV: {output_articles_path}")
    return summary


# ══════════════════════════════════════════════════════════════
# Ishan's Stratified Long-Tail Sampler
# ══════════════════════════════════════════════════════════════

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
