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
                        choices=["all", "sample", "embed", "train_villain", "train_hero", "evaluate", "ablation", "pareto_sweep", "pareto_plot"],
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
        logger.info("[1/9] Sampling data...")
        from src.data.sampler import create_sample
        create_sample(config)

    # Stage 2: Visual Embedding Extraction + Multimodal Fusion
    if args.stage in ("all", "embed"):
        logger.info("[2/9] Extracting visual embeddings...")
        from src.data.extract_visual_embeddings import extract_all
        extract_all(config)

        logger.info("[2/9] Fusing multimodal embeddings...")
        from src.data.fuse_multimodal_embeddings import fuse_all
        fuse_all(config)

    # Stage 3: Train Villain (Baseline)
    if args.stage in ("all", "train_villain"):
        logger.info("[3/9] Training the Villain (baseline)...")
        from src.villain.trainer import train_villain
        train_villain(config)

    # Stage 4: Train Hero (Main Model)
    if args.stage in ("all", "train_hero"):
        logger.info("[4/9] Training the Hero (main model)...")
        from src.hero.trainer import train_hero
        train_hero(config)

    # Stage 5: Evaluate Both Models
    if args.stage in ("all", "evaluate"):
        logger.info("[5/9] Evaluating both models...")
        from src.villain.evaluate import evaluate_villain
        evaluate_villain(config)

        from src.hero.evaluate import evaluate_hero
        evaluate_hero(config)

    # Stage 6: Cold-Start Analysis
    if args.stage in ("all", "evaluate"):
        logger.info("[6/9] Running cold-start analysis...")
        from src.hero.evaluate_cold_start import run_cold_start_simulation
        run_cold_start_simulation(config)

    # Stage 7: Ablation Study — ID-only Hero (Phase 3)
    if args.stage in ("all", "ablation"):
        logger.info("[7/9] Running ablation study (ID-only Hero)...")
        from src.hero.ablation import run_ablation
        run_ablation(config)

    # Stage 8: Pareto λ-Discovery Sweep (Phase 3)
    if args.stage in ("all", "pareto_sweep"):
        logger.info("[8/9] Running Pareto λ-discovery sweep...")
        from src.hero.pareto_sweep import run_pareto_sweep
        run_pareto_sweep(config)

    # Stage 9: Pareto Front Visualisation (Phase 3)
    if args.stage in ("all", "pareto_plot"):
        logger.info("[9/9] Generating Pareto front visualisation...")
        from src.utils.pareto_plot import generate_pareto_plots
        generate_pareto_plots(config)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
