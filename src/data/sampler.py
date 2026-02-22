"""
sampler.py
==========
Step 1 implementation: convert full transactions data to compact numeric format.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


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
