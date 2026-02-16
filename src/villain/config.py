"""
config.py — Villain-Specific Hyperparameter Overrides
=======================================================
Team: Ishan, Elizabeth, Nishant

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


def get_villain_config(global_config):
    """Merge global config with villain defaults."""
    raise NotImplementedError("TODO: Implement config merging")
