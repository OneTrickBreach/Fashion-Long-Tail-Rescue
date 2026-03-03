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
        logger.info("[1/6] Sampling data...")
        from src.data.sampler import create_sample
        create_sample(config)

    # Stage 2: Visual Embedding Extraction + Multimodal Fusion
    if args.stage in ("all", "embed"):
        logger.info("[2/6] Extracting visual embeddings...")
        from src.data.extract_visual_embeddings import extract_all
        extract_all(config)

        logger.info("[2/6] Fusing multimodal embeddings...")
        from src.data.fuse_multimodal_embeddings import fuse_all
        fuse_all(config)

    # Stage 3: Train Villain (Baseline)
    if args.stage in ("all", "train_villain"):
        logger.info("[3/6] Training the Villain (baseline)...")
        from src.villain.trainer import train_villain
        train_villain(config)

    # Stage 4: Train Hero (Main Model)
    if args.stage in ("all", "train_hero"):
        logger.info("[4/6] Training the Hero (main model)...")
        from src.hero.trainer import train_hero
        train_hero(config)

    # Stage 5: Evaluate Both Models
    if args.stage in ("all", "evaluate"):
        logger.info("[5/6] Evaluating both models...")
        from src.villain.evaluate import evaluate_villain
        evaluate_villain(config)

        from src.hero.evaluate import evaluate_hero
        evaluate_hero(config)

    # Stage 6: Cold-Start Analysis
    if args.stage in ("all", "evaluate"):
        logger.info("[6/6] Running cold-start analysis...")
        from src.hero.evaluate_cold_start import run_cold_start_simulation
        run_cold_start_simulation(config)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
