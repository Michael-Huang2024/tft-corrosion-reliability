"""
Run the complete reproducibility pipeline in manuscript order.

Pipeline:
1. Chloride diffusion simulation
2. Corrosion initiation labeling
3. TFT model training
4. Pf(t) inference
5. Manuscript figure generation
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run_step(script: str, extra_args: list[str]) -> None:
    command = [sys.executable, str(ROOT / "scripts" / script), *extra_args]
    print("\nRunning:", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full reproducibility pipeline.")
    parser.add_argument("--skip-training", action="store_true", help="Use an existing checkpoint for inference.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint path used when skipping training.")
    parser.add_argument("--n-series", type=int, default=1000, help="Number of simulated scenario series.")
    parser.add_argument("--max-epochs", type=int, default=40, help="Maximum TFT training epochs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_step("01_generate_data.py", ["--n-series", str(args.n_series)])
    run_step("02_label_onset.py", [])

    if not args.skip_training:
        run_step("03_train_model.py", ["--max-epochs", str(args.max_epochs)])

    infer_args: list[str] = []
    if args.checkpoint is not None:
        infer_args.extend(["--checkpoint", str(args.checkpoint)])
    run_step("04_infer.py", infer_args)
    run_step("05_make_figures.py", [])


if __name__ == "__main__":
    main()
