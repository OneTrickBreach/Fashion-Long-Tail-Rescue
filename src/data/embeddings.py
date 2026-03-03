"""
embeddings.py — Convenience wrappers for visual & multimodal embeddings
========================================================================
Team: Ishan, Elizabeth, Nishant

PURPOSE:
    Thin delegation layer that forwards to the concrete implementations:
        - ``src.data.extract_visual_embeddings.extract_all``
        - ``src.data.fuse_multimodal_embeddings.fuse_all``
        - ``src.data.dataset.load_multimodal_embeddings``

    These wrappers exist so that ``run_all.py`` and other scripts can
    import from a single module if desired.
"""


def extract_embeddings(config: dict) -> None:
    """
    Extract ResNet50 visual embeddings for all product images, then
    fuse with categorical metadata.  Delegates to the standalone scripts.

    Args:
        config: Parsed config.yaml dict.
    """
    from src.data.extract_visual_embeddings import extract_all
    from src.data.fuse_multimodal_embeddings import fuse_all

    extract_all(config)
    fuse_all(config)


def load_embeddings(embeddings_path: str, id_to_idx: dict, num_items: int):
    """
    Load pre-computed multimodal embeddings aligned with the ID map.

    Delegates to ``src.data.dataset.load_multimodal_embeddings``.

    Args:
        embeddings_path: Path to multimodal_embeddings.pt.
        id_to_idx:       Raw article_id → contiguous index mapping.
        num_items:       Total vocabulary size (includes PAD at 0).

    Returns:
        torch.Tensor of shape (num_items, fused_dim).
    """
    from src.data.dataset import load_multimodal_embeddings

    return load_multimodal_embeddings(embeddings_path, id_to_idx, num_items)
