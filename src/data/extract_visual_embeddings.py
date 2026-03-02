"""
extract_visual_embeddings.py — ResNet50 Visual Feature Extraction
=================================================================
Team member: Ishan Biswas (covering for Elizabeth Coquillette)

PURPOSE:
    Pass all H&M product images through a frozen ResNet50 (ImageNet-pretrained)
    and save the 2048-dim average-pool output as a per-article embedding bank.

    Output is saved to ``data/embeddings/resnet50_embeddings.pt`` as a dict::

        {
            "embeddings": Tensor of shape (num_articles, 2048),
            "article_ids": list of int article IDs (same row order),
            "missing_ids": list of int article IDs that had no image,
        }

KEY FUNCTIONS:
    build_resnet_extractor  — loads frozen ResNet50, strips classification head
    article_id_to_image_path — maps an article ID to its JPEG path
    extract_all             — main loop: batch extraction on GPU, saves .pt

USAGE (from project root, venv activated):
    python -m src.data.extract_visual_embeddings
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from src.utils.helpers import load_config, get_device, setup_logging

logger = setup_logging()


# ──────────────────────────────────────────────────────────────
# Image transforms (ImageNet normalisation)
# ──────────────────────────────────────────────────────────────

IMAGE_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ──────────────────────────────────────────────────────────────
# ResNet50 extractor
# ──────────────────────────────────────────────────────────────

def build_resnet_extractor(device: torch.device) -> torch.nn.Module:
    """
    Load a frozen ResNet50, strip the final FC layer, and return
    a model that outputs (batch_size, 2048) feature vectors.

    Uses the ImageNet-pretrained weights. All parameters are frozen
    (no gradients needed — inference only).
    """
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    resnet = models.resnet50(weights=weights)

    # Remove the final fully-connected classification head.
    # After avgpool the output is (batch, 2048, 1, 1) — we flatten it.
    extractor = torch.nn.Sequential(*list(resnet.children())[:-1])

    extractor.eval()
    extractor.to(device)

    # Freeze all parameters
    for param in extractor.parameters():
        param.requires_grad = False

    logger.info("ResNet50 extractor loaded (frozen, %s).", device)
    return extractor


# ──────────────────────────────────────────────────────────────
# Image path resolution
# ──────────────────────────────────────────────────────────────

def article_id_to_image_path(article_id: int, images_root: Path) -> Path:
    """
    Map an article ID to its JPEG path under the H&M images directory.

    H&M naming convention:
        article_id  = 108775015
        zero-padded = 0108775015
        subfolder   = 010
        filename    = 0108775015.jpg
        full path   = images_root / 010 / 0108775015.jpg
    """
    padded = str(article_id).zfill(10)
    subfolder = padded[:3]
    return images_root / subfolder / f"{padded}.jpg"


# ──────────────────────────────────────────────────────────────
# PyTorch Dataset for image loading
# ──────────────────────────────────────────────────────────────

class ArticleImageDataset(Dataset):
    """
    Lazily loads article product images for batch extraction.

    For articles whose images are missing or unreadable, returns a
    zero tensor and sets a flag so the caller can record the miss.
    """

    def __init__(self, article_ids: list[int], images_root: Path):
        self.article_ids = article_ids
        self.images_root = images_root

    def __len__(self):
        return len(self.article_ids)

    def __getitem__(self, idx: int):
        article_id = self.article_ids[idx]
        img_path = article_id_to_image_path(article_id, self.images_root)

        try:
            # Read raw bytes → decode to tensor (H, W, C) uint8
            from torchvision.io import read_image, ImageReadMode
            img = read_image(str(img_path), mode=ImageReadMode.RGB)  # (3, H, W)
            img = IMAGE_TRANSFORMS(img)
            valid = True
        except Exception:
            # Missing / corrupt image → zero tensor
            img = torch.zeros(3, 224, 224, dtype=torch.float32)
            valid = False

        return img, article_id, valid


# ──────────────────────────────────────────────────────────────
# Main extraction loop
# ──────────────────────────────────────────────────────────────

def extract_all(config: dict):
    """
    Batch-extract ResNet50 embeddings for every article in articles.csv.

    Reads article IDs from the raw articles CSV, resolves image paths,
    runs batched forward passes on GPU, and saves the result to
    ``data/embeddings/resnet50_embeddings.pt``.

    Args:
        config: Parsed config.yaml dict.
    """
    raw_data_dir = Path(config["paths"]["raw_data"])
    embeddings_dir = Path(config["paths"]["embeddings"])
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    output_path = embeddings_dir / "resnet50_embeddings.pt"

    device_pref = config.get("embedding", {}).get("device", "cuda")
    batch_size = config.get("embedding", {}).get("batch_size", 64)
    device = get_device(device_pref)

    # ── Load all article IDs ──
    articles_csv = raw_data_dir / "articles.csv"
    logger.info("Reading article IDs from %s ...", articles_csv)
    df = pd.read_csv(articles_csv, usecols=["article_id"])
    article_ids = df["article_id"].tolist()
    num_articles = len(article_ids)
    logger.info("Total articles to process: %d", num_articles)

    # ── Build extractor ──
    extractor = build_resnet_extractor(device)

    # ── Build DataLoader ──
    images_root = raw_data_dir / "images"
    dataset = ArticleImageDataset(article_ids, images_root)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    # ── Extract ──
    all_embeddings = torch.zeros(num_articles, 2048, dtype=torch.float32)
    missing_ids = []
    processed = 0
    start_time = time.time()

    logger.info("Starting extraction (batch_size=%d, device=%s) ...", batch_size, device)

    with torch.no_grad():
        for batch_idx, (images, ids, valid_flags) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            features = extractor(images)          # (B, 2048, 1, 1)
            features = features.squeeze(-1).squeeze(-1)  # (B, 2048)
            features = features.cpu()

            # Write into the pre-allocated tensor at the correct global rows
            start_row = batch_idx * batch_size
            end_row = start_row + features.shape[0]
            all_embeddings[start_row:end_row] = features

            # Zero out missing images and record them
            for i, (aid, v) in enumerate(zip(ids.tolist(), valid_flags.tolist())):
                if not v:
                    all_embeddings[start_row + i] = 0.0
                    missing_ids.append(aid)

            processed += features.shape[0]
            if (batch_idx + 1) % 50 == 0 or processed == num_articles:
                elapsed = time.time() - start_time
                rate = processed / elapsed
                eta = (num_articles - processed) / rate if rate > 0 else 0
                logger.info(
                    "  [%d / %d] (%.1f%%)  %.0f img/s  ETA %.0fs",
                    processed, num_articles,
                    100.0 * processed / num_articles,
                    rate, eta,
                )

    elapsed_total = time.time() - start_time

    # ── Save ──
    result = {
        "embeddings": all_embeddings,           # (num_articles, 2048)
        "article_ids": article_ids,             # list[int], same row order
        "missing_ids": missing_ids,             # list[int]
    }
    torch.save(result, output_path)

    logger.info("─" * 60)
    logger.info("Extraction complete in %.1f s (%.0f img/s).", elapsed_total, num_articles / elapsed_total)
    logger.info("  Total articles:  %d", num_articles)
    logger.info("  Missing images:  %d (%.1f%%)", len(missing_ids), 100.0 * len(missing_ids) / num_articles)
    logger.info("  Embedding shape: %s", list(all_embeddings.shape))
    logger.info("  Saved to: %s", output_path)

    # ── Quick verification ──
    _verify_embeddings(output_path)


# ──────────────────────────────────────────────────────────────
# Post-extraction verification
# ──────────────────────────────────────────────────────────────

def _verify_embeddings(path: Path):
    """
    Reload the saved .pt file and run basic sanity checks:
    spot-check 10 random articles for correct shape and non-zero values.
    """
    logger.info("Running verification on %s ...", path)
    data = torch.load(path, weights_only=False)

    embeddings = data["embeddings"]
    article_ids = data["article_ids"]
    missing_ids = set(data["missing_ids"])

    assert embeddings.shape[1] == 2048, f"Expected dim 2048, got {embeddings.shape[1]}"
    assert embeddings.shape[0] == len(article_ids), "Row count mismatch"

    # Spot-check 10 random non-missing articles
    rng = np.random.RandomState(42)
    valid_indices = [i for i, aid in enumerate(article_ids) if aid not in missing_ids]
    if len(valid_indices) >= 10:
        sample_indices = rng.choice(valid_indices, size=10, replace=False)
        for idx in sample_indices:
            vec = embeddings[idx]
            assert vec.shape == (2048,), f"Bad shape at idx {idx}: {vec.shape}"
            assert vec.abs().sum() > 0, f"Zero vector at idx {idx} (article {article_ids[idx]})"
        logger.info("  ✓ Spot-check passed: 10 random embeddings are (2048,) and non-zero.")
    else:
        logger.warning("  ⚠ Fewer than 10 valid embeddings — skipping spot-check.")

    logger.info("  ✓ Verification complete.")


# ──────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = load_config()
    extract_all(cfg)
