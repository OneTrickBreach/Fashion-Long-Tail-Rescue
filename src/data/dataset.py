"""
dataset.py — PyTorch Dataset & DataLoader Definitions
======================================================
Team member: Ishan Biswas
Key classes: TransactionDataset
Key functions: build_dataloaders, build_id_maps

PURPOSE:
    Defines custom Dataset classes for the Villain (text-only sequences)
    and later the Hero (multimodal sequences with visual embeddings).

    The core abstraction: each sample is one user represented as a
    chronological sequence of item-index IDs.  We use a leave-one-out
    temporal split (last item → test, second-to-last → val, rest → train).

NOTES:
    - Raw article_id values are large 9-digit integers.  We remap them to
      contiguous 0-based indices via `build_id_maps()`.
    - Sequences are right-padded to `max_seq_len` with a dedicated PAD
      token (index 0).
    - Uses `config.yaml` paths to locate sampled CSVs.
"""

import os
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.utils.helpers import load_config

logger = logging.getLogger("seeing-the-unseen")

# Reserve index 0 for PAD token
PAD_IDX = 0


# ──────────────────────────────────────────────────────────────
# ID mapping
# ──────────────────────────────────────────────────────────────

def build_id_maps(articles_path: str) -> tuple[dict, dict]:
    """
    Build bidirectional mappings between raw article_id and
    contiguous 0-based indices.  Index 0 is reserved for PAD.

    Args:
        articles_path: Path to articles_sampled.csv.

    Returns:
        (id_to_idx, idx_to_id) — both are plain dicts.
    """
    articles = pd.read_csv(articles_path)
    unique_ids = sorted(articles["article_id"].unique())
    # Start from 1; index 0 = PAD
    id_to_idx = {aid: i + 1 for i, aid in enumerate(unique_ids)}
    idx_to_id = {v: k for k, v in id_to_idx.items()}
    idx_to_id[PAD_IDX] = -1  # sentinel for PAD
    logger.info(f"ID map built: {len(unique_ids)} articles → indices 1..{len(unique_ids)}")
    return id_to_idx, idx_to_id


# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────

class TransactionDataset(Dataset):
    """
    Sequential recommendation dataset using leave-one-out evaluation.

    For each user's chronologically sorted purchase history:
        - mode="train": input  = items[:-2], target = items[-3] … sliding window
        - mode="val":   input  = items[:-1], target = items[-2]
        - mode="test":  input  = items[:],   target = items[-1]

    Each sample returns:
        item_seq   (LongTensor): (max_seq_len,) — 0-padded item indices
        positions  (LongTensor): (max_seq_len,) — position indices [0..seq_len-1]
        target     (LongTensor): scalar — ground-truth next item index
        seq_len    (LongTensor): scalar — actual (unpadded) sequence length
    """

    def __init__(
        self,
        user_sequences: dict[str, list[int]],
        id_to_idx: dict[int, int],
        max_seq_len: int,
        mode: str = "train",
        num_items: int | None = None,
        seed: int = 42,
    ):
        """
        Args:
            user_sequences: {customer_id: [article_id, …]} in chronological order.
            id_to_idx:      Raw article_id → contiguous index mapping.
            max_seq_len:    Maximum sequence length (truncate/pad to this).
            mode:           "train", "val", or "test".
            num_items:      Total number of items (for negative sampling bounds).
            seed:           RNG seed for negative sampling.
        """
        super().__init__()
        assert mode in ("train", "val", "test"), f"Unknown mode: {mode}"
        self.mode = mode
        self.max_seq_len = max_seq_len
        self.id_to_idx = id_to_idx
        self.num_items = num_items or (max(id_to_idx.values()) + 1)
        self.rng = np.random.RandomState(seed)

        self.samples = self._build_samples(user_sequences)
        logger.info(f"TransactionDataset({mode}): {len(self.samples)} samples")

    # ── sample construction ──────────────────────────────────

    def _build_samples(self, user_sequences: dict[str, list[int]]) -> list[tuple]:
        """
        Convert raw user sequences into (input_indices, target_index) pairs.

        Leave-one-out protocol:
          train → all input subsequences up to items[:-2], targets are next items
          val   → input = items[:-1], target = second-to-last
          test  → input = all items[:-1], target = last item
        """
        samples = []
        for uid, raw_items in user_sequences.items():
            # Map to contiguous indices, skip unknown items
            idx_seq = [
                self.id_to_idx[a] for a in raw_items if a in self.id_to_idx
            ]
            if len(idx_seq) < 3:
                # Need at least 3 items for train/val/test split
                continue

            if self.mode == "test":
                inp = idx_seq[:-1]
                tgt = idx_seq[-1]
                samples.append((inp, tgt))
            elif self.mode == "val":
                inp = idx_seq[:-2]
                tgt = idx_seq[-2]
                samples.append((inp, tgt))
            else:  # train — sliding window over the prefix
                # For each position t (2 <= t <= len-2), predict item[t]
                prefix = idx_seq[:-2]  # exclude val & test items
                for t in range(1, len(prefix)):
                    inp = prefix[:t]
                    tgt = prefix[t]
                    samples.append((inp, tgt))
        return samples

    # ── PyTorch interface ────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        inp, tgt = self.samples[index]

        # Truncate to max_seq_len (keep the most recent items)
        if len(inp) > self.max_seq_len:
            inp = inp[-self.max_seq_len:]

        seq_len = len(inp)

        # Right-pad with PAD_IDX
        padded = inp + [PAD_IDX] * (self.max_seq_len - seq_len)

        item_seq = torch.tensor(padded, dtype=torch.long)
        positions = torch.arange(self.max_seq_len, dtype=torch.long)
        target = torch.tensor(tgt, dtype=torch.long)

        return {
            "item_seq": item_seq,
            "positions": positions,
            "target": target,
            "seq_len": torch.tensor(seq_len, dtype=torch.long),
        }


# ──────────────────────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────────────────────

def build_dataloaders(
    config: dict,
    mode: str = "villain",
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Factory function that returns train/val/test DataLoaders.

    Args:
        config: Parsed config.yaml.
        mode:   "villain" for text-only or "hero" for multimodal (future).

    Returns:
        (train_loader, val_loader, test_loader, metadata)
        where metadata = {"id_to_idx", "idx_to_id", "num_items"}.
    """
    from src.utils.helpers import setup_logging
    setup_logging()

    paths = config["paths"]
    model_cfg = config[mode]  # "villain" or "hero" section
    seed = config["project"]["seed"]

    sampled_dir = paths["sampled_data"]
    txn_path = os.path.join(sampled_dir, "transactions_sampled.csv")
    art_path = os.path.join(sampled_dir, "articles_sampled.csv")

    # ── Build ID maps ────────────────────────────────────────
    id_to_idx, idx_to_id = build_id_maps(art_path)
    num_items = max(id_to_idx.values()) + 1  # includes PAD at 0

    # ── Build per-user chronological sequences ───────────────
    logger.info("Building user sequences from sampled transactions …")
    txn = pd.read_csv(txn_path, parse_dates=["t_dat"])
    txn = txn.sort_values(["customer_id", "t_dat"])

    user_sequences: dict[str, list[int]] = {}
    for uid, grp in txn.groupby("customer_id"):
        user_sequences[uid] = grp["article_id"].tolist()

    max_seq_len = model_cfg["max_seq_len"]
    batch_size = model_cfg["batch_size"]

    # ── Create datasets ──────────────────────────────────────
    train_ds = TransactionDataset(
        user_sequences, id_to_idx, max_seq_len,
        mode="train", num_items=num_items, seed=seed,
    )
    val_ds = TransactionDataset(
        user_sequences, id_to_idx, max_seq_len,
        mode="val", num_items=num_items, seed=seed,
    )
    test_ds = TransactionDataset(
        user_sequences, id_to_idx, max_seq_len,
        mode="test", num_items=num_items, seed=seed,
    )

    # ── Create loaders ───────────────────────────────────────
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    metadata = {
        "id_to_idx": id_to_idx,
        "idx_to_id": idx_to_id,
        "num_items": num_items,
    }

    logger.info(
        f"DataLoaders ready: train={len(train_ds)}, val={len(val_ds)}, "
        f"test={len(test_ds)}, num_items={num_items}, "
        f"batch_size={batch_size}, max_seq_len={max_seq_len}"
    )
    return train_loader, val_loader, test_loader, metadata


# ──────────────────────────────────────────────────────────────
# CLI smoke test
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = load_config()
    train_ld, val_ld, test_ld, meta = build_dataloaders(cfg, mode="villain")

    batch = next(iter(train_ld))
    print("\n=== Smoke Test: first train batch ===")
    for k, v in batch.items():
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
    print(f"  num_items: {meta['num_items']}")
    print("  ✓ DataLoader smoke test passed")
