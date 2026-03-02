"""
run_all.py — Master Pipeline Script
=====================================
Team: Ishan, Elizabeth, Nishant
CS7180 Applied Deep Learning

PURPOSE:
    Single entry point that runs the full pipeline end-to-end:

    1. Sample data (if not already sampled)
    2. Extract visual embeddings (if not already extracted)
    3. Train the Villain (baseline)
    4. Train the Hero (main model)
    5. Evaluate both on multi-objective metrics
    6. Generate comparison report & Pareto analysis

USAGE:
    python run_all.py --config config.yaml
    python run_all.py --config config.yaml --stage train_hero   # run a single stage
    python run_all.py --config config.yaml --skip-sampling      # skip data prep
"""

import argparse

from src.utils.helpers import load_config, set_seed, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Seeing the Unseen — Full Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--stage", type=str, default="all",
                        choices=["all", "sample", "embed", "train_villain", "train_hero", "evaluate"],
                        help="Run a specific stage only")
    parser.add_argument("--skip-sampling", action="store_true", help="Skip data sampling step")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logging()
    set_seed(config["project"]["seed"])

    logger.info("=" * 60)
    logger.info("  Seeing the Unseen — Multi-Objective Rescue of the Fashion Long Tail")
    logger.info("=" * 60)

    # Stage 1: Data Sampling
    if args.stage in ("all", "sample") and not args.skip_sampling:
        logger.info("[1/5] Sampling data...")
        from src.data.sampler import create_sample
        create_sample(config)

    # Stage 2: Visual Embedding Extraction
    if args.stage in ("all", "embed"):
        logger.info("[2/5] Extracting visual embeddings...")
        # Phase 2: from src.data.embeddings import extract_embeddings
        # extract_embeddings(config)
        logger.info("  (skipped — Phase 2 feature)")

    # Stage 3: Train Villain (Baseline)
    if args.stage in ("all", "train_villain"):
        logger.info("[3/5] Training the Villain (baseline)...")
        from src.villain.trainer import train_villain
        train_villain(config)

    # Stage 4: Train Hero (Main Model)
    if args.stage in ("all", "train_hero"):
        logger.info("[4/5] Training the Hero (main model)...")
        # Phase 2: from src.hero.trainer import train_hero
        # train_hero(config)
        logger.info("  (skipped — Phase 2 feature)")

    # Stage 5: Evaluate Both Models
    if args.stage in ("all", "evaluate"):
        logger.info("[5/5] Evaluating both models...")
        from src.villain.evaluate import evaluate_villain
        evaluate_villain(config)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
