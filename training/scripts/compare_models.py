import os
import sys
import argparse

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from training.src.comparison import run_cross_model_comparison


def parse_args():
    parser = argparse.ArgumentParser(description="Run cross-model evaluation comparison and composite ranking.")
    parser.add_argument("--experiments_root", type=str, default="outputs/experiments", help="Path to experiments output directory.")
    parser.add_argument("--output_dir", type=str, default="outputs/comparison", help="Output directory for comparison results.")
    return parser.parse_args()


def main():
    args = parse_args()
    run_cross_model_comparison(
        experiments_root=os.path.abspath(args.experiments_root),
        output_dir=os.path.abspath(args.output_dir),
    )


if __name__ == "__main__":
    main()
