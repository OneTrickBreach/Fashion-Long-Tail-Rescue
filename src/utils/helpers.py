"""
helpers.py — Shared Utility Functions
=======================================
Team: Ishan, Elizabeth, Nishant

PURPOSE:
    Common utilities used across the entire project:
    - Config loading (YAML → dict)
    - Random seed setting (reproducibility)
    - Device selection (CPU / CUDA)
    - Logging setup
"""

import random
import yaml
import torch
import numpy as np


def load_config(path="config.yaml"):
    """Load and return the YAML configuration as a dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed=42):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preference="cuda"):
    """
    Return the best available torch device.

    Args:
        preference (str): "cuda" or "cpu".

    Returns:
        torch.device
    """
    if preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def setup_logging(name="seeing-the-unseen"):
    """Configure logging for the project."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format=f"[{name}] %(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)
