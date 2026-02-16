"""
dataset.py — PyTorch Dataset & DataLoader Definitions
======================================================
Team: Ishan, Elizabeth, Nishant

PURPOSE:
    Defines custom Dataset classes for both the Villain (text-only sequences)
    and the Hero (multimodal sequences with visual embeddings).

KEY CLASSES (to implement):
    - TransactionDataset:  Loads sampled transactions and builds user-item
                           interaction sequences with chronological ordering.
    - MultimodalDataset:   Extends TransactionDataset by attaching pre-computed
                           ResNet50 embeddings to each item in the sequence.

NOTES:
    - All datasets should support memory-mapped loading where possible to
      stay within local PC RAM limits.
    - Use `config.yaml` paths to locate sampled CSVs and embedding files.
"""


def build_dataloaders(config, mode="villain"):
    """
    Factory function that returns train/val/test DataLoaders.

    Args:
        config (dict): Parsed config.yaml.
        mode (str): "villain" for text-only or "hero" for multimodal.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    raise NotImplementedError("TODO: Implement DataLoader construction")
