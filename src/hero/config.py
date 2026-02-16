"""
config.py — Hero-Specific Hyperparameter Overrides
====================================================
Team: Ishan, Elizabeth, Nishant

PURPOSE:
    Provides defaults and validation for Hero hyperparameters.
    Values in config.yaml take precedence; this module fills gaps.
"""

HERO_DEFAULTS = {
    "max_seq_len": 50,
    "hidden_dim": 128,
    "num_heads": 4,
    "num_layers": 3,
    "dropout": 0.1,
    "lr": 0.0005,
    "epochs": 50,
    "batch_size": 128,
    "contrastive": {
        "temperature": 0.07,
        "hard_negatives": 10,
        "weight": 0.3,
    },
}


def get_hero_config(global_config):
    """Merge global config with hero defaults."""
    raise NotImplementedError("TODO: Implement config merging")
