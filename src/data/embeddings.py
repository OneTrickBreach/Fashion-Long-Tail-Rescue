"""
embeddings.py — ResNet50 Visual Feature Extraction Pipeline
=============================================================
Team: Ishan, Elizabeth, Nishant

PURPOSE:
    Pre-computes 2048-dim visual embeddings for all product images using a
    pre-trained ResNet50 backbone. Embeddings are saved to disk so the Hero
    model can load them as a lookup table instead of running CNN inference
    during every training epoch.

KEY FUNCTIONS (to implement):
    - extract_embeddings():   Iterate over product images in batches, run
                              through ResNet50 (with final FC layer removed),
                              and save the output tensor to `data/embeddings/`.
    - load_embeddings():      Memory-map the saved .npy file for efficient
                              random access during training.

OUTPUT:
    - data/embeddings/article_embeddings.npy  → (num_articles, 2048) float32
    - data/embeddings/article_id_map.json     → row index → article_id mapping
"""

import torch


def extract_embeddings(config):
    """
    Extract ResNet50 embeddings for all product images and save to disk.

    Args:
        config (dict): Parsed config.yaml.
    """
    raise NotImplementedError("TODO: Implement ResNet50 embedding extraction")


def load_embeddings(config):
    """
    Load pre-computed embeddings as a memory-mapped numpy array.

    Args:
        config (dict): Parsed config.yaml.

    Returns:
        tuple: (embeddings_array, id_to_index_map)
    """
    raise NotImplementedError("TODO: Implement embedding loading")
