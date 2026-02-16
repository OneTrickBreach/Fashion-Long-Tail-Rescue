"""
sampler.py — Stratified Long-Tail Sampler
==========================================
Team: Ishan, Elizabeth, Nishant

PURPOSE:
    Creates a memory-friendly subset of the full H&M dataset while preserving
    the long-tail distribution of item popularity. This is essential because
    training on the full 31M-row dataset is infeasible on a single local PC.

KEY FUNCTIONS (to implement):
    - create_sample():        Read raw CSVs, apply stratified sampling, and
                              write smaller CSVs to `data/sampled/`.
    - analyze_distribution(): Print/plot item frequency distribution before
                              and after sampling for verification.

SAMPLING STRATEGY:
    - Stratified by item popularity bins (head / torso / tail) so that the
      long-tail structure is preserved in the subsample.
    - Configurable via `config.yaml → sampling.fraction` and
      `sampling.min_interactions`.
"""


def create_sample(config):
    """
    Read raw data, apply stratified sampling, and save to data/sampled/.

    Args:
        config (dict): Parsed config.yaml.
    """
    raise NotImplementedError("TODO: Implement stratified sampling")
