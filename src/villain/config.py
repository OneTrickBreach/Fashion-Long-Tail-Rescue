"""
config.py — Villain-Specific Hyperparameter Overrides
=======================================================
Team member: Ishan Biswas
Key functions: get_villain_config

PURPOSE:
    Provides defaults and validation for Villain hyperparameters.
    Values in config.yaml take precedence; this module fills gaps.
"""

VILLAIN_DEFAULTS = {
    "model_type": "sasrec",
    "max_seq_len": 50,
    "hidden_dim": 64,
    "num_heads": 2,
    "num_layers": 2,
    "dropout": 0.2,
    "lr": 0.001,
    "epochs": 30,
    "batch_size": 256,
}


def get_villain_config(global_config: dict) -> dict:
    """
    Merge global config's villain section with defaults.

    Values in config.yaml override VILLAIN_DEFAULTS.

    Args:
        global_config: Full parsed config.yaml dict.

    Returns:
        dict with all villain hyperparameters resolved.
    """
    merged = {**VILLAIN_DEFAULTS}
    if "villain" in global_config:
        merged.update(global_config["villain"])
    return merged
