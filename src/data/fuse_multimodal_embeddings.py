"""
fuse_multimodal_embeddings.py — Multimodal Feature Fusion (Visual + Metadata)
=============================================================================
Team member: Ishan Biswas (covering for Elizabeth Coquillette)

PURPOSE:
    Merge pre-extracted ResNet50 visual embeddings (2048-dim) with categorical
    article metadata from articles.csv (colour, product type, garment group, etc.)
    to produce a single fused embedding per article.

    Pipeline:
        1. Load ResNet50 embeddings from ``data/embeddings/resnet50_embeddings.pt``
        2. Load article metadata from ``articles.csv`` in the raw data directory
        3. Label-encode each categorical metadata column → integer indices
        4. Concatenate the encoded metadata into a one-hot or ordinal vector
        5. Concatenate visual (2048) + metadata → project via linear layer
        6. Save fused embeddings to ``data/embeddings/multimodal_embeddings.pt``

    Output dict::

        {
            "embeddings":     Tensor (num_articles, fused_dim),
            "article_ids":    list[int],
            "visual_dim":     2048,
            "metadata_dim":   int,
            "fused_dim":      int,
            "category_maps":  dict[str, dict[str, int]],  # label encodings
        }

KEY FUNCTIONS:
    load_resnet_embeddings  — loads the .pt file from Task 1
    encode_article_metadata — label-encodes categorical columns
    fuse_and_project        — concatenates + linear projection
    fuse_all                — orchestrates the full pipeline

USAGE (from project root, venv activated):
    python -m src.data.fuse_multimodal_embeddings
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.utils.helpers import load_config, setup_logging

logger = setup_logging()

# Categorical columns from articles.csv to use as metadata features.
# These capture colour, product type, graphical style, and garment category —
# all relevant signals for visual similarity in fashion recommendation.
METADATA_COLUMNS = [
    "colour_group_name",
    "perceived_colour_value_name",
    "product_type_name",
    "product_group_name",
    "graphical_appearance_name",
    "garment_group_name",
    "index_group_name",
    "section_name",
]


# ──────────────────────────────────────────────────────────────
# Loading ResNet50 embeddings
# ──────────────────────────────────────────────────────────────

def load_resnet_embeddings(embeddings_dir: Path) -> dict:
    """
    Load the ResNet50 embedding bank produced by Task 1.

    Returns:
        dict with keys "embeddings" (Tensor), "article_ids" (list),
        "missing_ids" (list).
    """
    path = embeddings_dir / "resnet50_embeddings.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"ResNet50 embeddings not found at {path}. "
            "Run Task 1 (extract_visual_embeddings.py) first."
        )
    data = torch.load(path, weights_only=False)
    logger.info(
        "Loaded ResNet50 embeddings: shape=%s, %d article IDs.",
        list(data["embeddings"].shape), len(data["article_ids"]),
    )
    return data


# ──────────────────────────────────────────────────────────────
# Metadata encoding
# ──────────────────────────────────────────────────────────────

def encode_article_metadata(
    articles_csv: Path,
    article_ids_ordered: list[int],
) -> tuple[torch.Tensor, dict[str, dict[str, int]]]:
    """
    Load articles.csv, extract categorical metadata columns, and produce
    a one-hot encoded tensor aligned with the given article_ids_ordered.

    Args:
        articles_csv:        Path to the raw articles.csv.
        article_ids_ordered: Article IDs in the exact row order of the
                             ResNet50 embedding matrix.

    Returns:
        (metadata_tensor, category_maps)
        - metadata_tensor: float32 Tensor of shape (num_articles, total_one_hot_dim)
        - category_maps:   dict mapping column_name → {value → int_label}
    """
    logger.info("Reading metadata from %s ...", articles_csv)
    df = pd.read_csv(articles_csv, usecols=["article_id"] + METADATA_COLUMNS)

    # Fill NaN with "UNKNOWN" for categorical columns
    for col in METADATA_COLUMNS:
        df[col] = df[col].fillna("UNKNOWN").astype(str)

    # Build label maps for each categorical column
    category_maps = {}
    for col in METADATA_COLUMNS:
        unique_vals = sorted(df[col].unique())
        category_maps[col] = {v: i for i, v in enumerate(unique_vals)}

    # Index the dataframe by article_id for fast lookup
    df = df.set_index("article_id")

    # Build the one-hot encoded matrix row by row, aligned with article_ids_ordered
    total_one_hot_dim = sum(len(m) for m in category_maps.values())
    logger.info(
        "Encoding %d metadata columns → %d-dim one-hot vector.",
        len(METADATA_COLUMNS), total_one_hot_dim,
    )

    metadata_matrix = np.zeros((len(article_ids_ordered), total_one_hot_dim), dtype=np.float32)
    missing_count = 0

    for row_idx, aid in enumerate(article_ids_ordered):
        if aid not in df.index:
            # Article not in articles.csv — leave as zeros
            missing_count += 1
            continue

        article_row = df.loc[aid]
        # Handle duplicate article IDs (take first row if duplicated)
        if isinstance(article_row, pd.DataFrame):
            article_row = article_row.iloc[0]

        offset = 0
        for col in METADATA_COLUMNS:
            val = str(article_row[col])
            label = category_maps[col].get(val, 0)
            metadata_matrix[row_idx, offset + label] = 1.0
            offset += len(category_maps[col])

    if missing_count > 0:
        logger.warning("  %d articles not found in articles.csv (left as zero vectors).", missing_count)

    metadata_tensor = torch.from_numpy(metadata_matrix)
    logger.info("  Metadata tensor shape: %s", list(metadata_tensor.shape))
    return metadata_tensor, category_maps


# ──────────────────────────────────────────────────────────────
# Fusion + projection
# ──────────────────────────────────────────────────────────────

def fuse_and_project(
    visual_embeddings: torch.Tensor,
    metadata_embeddings: torch.Tensor,
    fused_dim: int,
) -> torch.Tensor:
    """
    Concatenate visual and metadata embeddings, then project down to a
    fixed fused_dim via a single linear layer.

    This is a one-time offline projection (not trained end-to-end), so we
    use a randomly initialised linear layer with fixed seed for reproducibility.

    Args:
        visual_embeddings:  (num_articles, 2048)
        metadata_embeddings: (num_articles, metadata_dim)
        fused_dim:          Target output dimension.

    Returns:
        Tensor of shape (num_articles, fused_dim)
    """
    concat = torch.cat([visual_embeddings, metadata_embeddings], dim=1)
    input_dim = concat.shape[1]

    logger.info(
        "Fusing: visual(%d) + metadata(%d) = %d → project to %d.",
        visual_embeddings.shape[1], metadata_embeddings.shape[1],
        input_dim, fused_dim,
    )

    # Deterministic projection
    torch.manual_seed(42)
    projector = nn.Linear(input_dim, fused_dim, bias=True)
    nn.init.xavier_uniform_(projector.weight)
    nn.init.zeros_(projector.bias)

    with torch.no_grad():
        fused = projector(concat)  # (num_articles, fused_dim)
        # Apply LayerNorm for stable downstream usage
        ln = nn.LayerNorm(fused_dim)
        fused = ln(fused)

    logger.info("  Fused embedding shape: %s", list(fused.shape))
    return fused


# ──────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────

def fuse_all(config: dict):
    """
    Full multimodal fusion pipeline: load visual embeddings, encode metadata,
    fuse and project, then save to disk.

    Args:
        config: Parsed config.yaml dict.
    """
    raw_data_dir = Path(config["paths"]["raw_data"])
    embeddings_dir = Path(config["paths"]["embeddings"])
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    output_path = embeddings_dir / "multimodal_embeddings.pt"

    # The fused dimension matches the hero model's hidden_dim so it can
    # be used directly as input embeddings without an extra projection.
    fused_dim = config.get("embedding", {}).get("dim", 2048)

    # ── Step 1: Load ResNet50 embeddings ──
    resnet_data = load_resnet_embeddings(embeddings_dir)
    visual_embeddings = resnet_data["embeddings"]      # (N, 2048)
    article_ids = resnet_data["article_ids"]           # list[int]

    # ── Step 2: Encode article metadata ──
    articles_csv = raw_data_dir / "articles.csv"
    metadata_tensor, category_maps = encode_article_metadata(articles_csv, article_ids)

    # ── Step 3: Fuse and project ──
    fused_embeddings = fuse_and_project(visual_embeddings, metadata_tensor, fused_dim)

    # ── Step 4: Save ──
    result = {
        "embeddings": fused_embeddings,           # (num_articles, fused_dim)
        "article_ids": article_ids,               # list[int], same row order
        "visual_dim": visual_embeddings.shape[1],  # 2048
        "metadata_dim": metadata_tensor.shape[1],  # total one-hot dim
        "fused_dim": fused_dim,
        "category_maps": category_maps,            # for reference / debugging
    }
    torch.save(result, output_path)

    logger.info("─" * 60)
    logger.info("Multimodal fusion complete.")
    logger.info("  Articles:       %d", len(article_ids))
    logger.info("  Visual dim:     %d", visual_embeddings.shape[1])
    logger.info("  Metadata dim:   %d", metadata_tensor.shape[1])
    logger.info("  Fused dim:      %d", fused_dim)
    logger.info("  Saved to:       %s", output_path)

    # ── Verification ──
    _verify_fused(output_path)


def _verify_fused(path: Path):
    """Reload and spot-check the fused embeddings."""
    logger.info("Running verification on %s ...", path)
    data = torch.load(path, weights_only=False)
    emb = data["embeddings"]
    aids = data["article_ids"]

    assert emb.shape[0] == len(aids), "Row count mismatch"
    assert emb.shape[1] == data["fused_dim"], "Dim mismatch"

    # Spot-check 10 random articles
    rng = np.random.RandomState(42)
    sample_indices = rng.choice(len(aids), size=min(10, len(aids)), replace=False)
    for idx in sample_indices:
        vec = emb[idx]
        assert vec.shape == (data["fused_dim"],), f"Bad shape at idx {idx}"
        assert vec.abs().sum() > 0, f"Zero vector at idx {idx}"

    logger.info("  ✓ Spot-check passed: 10 random fused embeddings are valid.")
    logger.info("  ✓ Verification complete.")


# ──────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = load_config()
    fuse_all(cfg)
