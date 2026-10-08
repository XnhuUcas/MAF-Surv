"""Train complete legacy-logic MAF-Surv models across lambda_orth settings.

This script deliberately reuses the ablation trainer so that every setting has
the same architecture, data split, hyperparameters, and checkpoint policy as
the Table 3 complete-model configuration.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import torch

from train_lgg_ablation_models import ABLATION_CONFIGS, run_configuration, validate_dataset
from utils import split_data_cv


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
FULL_CONFIG = next(config for config in ABLATION_CONFIGS if config.name == "maf_surv_total")


def directory_name(value: float) -> str:
    return "lambda_orth_" + format(value, ".3f").rstrip("0").rstrip(".").replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Five-fold LGG sensitivity training for the complete legacy-logic MAF-Surv model."
    )
    parser.add_argument("--data-path", type=Path, default=PROJECT_ROOT / "Datasets" / "LGG" / "original_data.pkl")
    parser.add_argument(
        "--output-dir", type=Path, default=SCRIPT_DIR / "lgg_lambda_orth_sensitivity_legacy_logic"
    )
    parser.add_argument("--lambda-orth-values", type=float, nargs="+", default=[0.0, 0.03, 0.05, 0.10])
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=70)
    parser.add_argument("--min-delta", type=float, default=0.005)
    parser.add_argument("--learning-rate", type=float, default=0.0004)
    parser.add_argument("--lambda-reg", type=float, default=1e-5)
    parser.add_argument("--lambda-recon", type=float, default=0.3)
    parser.add_argument("--lambda-r3gan", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if any(value < 0.0 for value in args.lambda_orth_values):
        raise ValueError("All lambda_orth values must be non-negative.")
    if len(set(args.lambda_orth_values)) != len(args.lambda_orth_values):
        raise ValueError("Duplicate lambda_orth values are not allowed.")
    if not args.data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.data_path}")

    device = torch.device(f"cuda:{args.gpu_id}") if args.gpu_id >= 0 and torch.cuda.is_available() else torch.device("cpu")
    with args.data_path.open("rb") as file:
        payload = pickle.load(file)
    if "datasets" not in payload:
        raise KeyError("The pickle must contain a top-level 'datasets' dictionary.")
    data = payload["datasets"]
    validate_dataset(data)
    splits = split_data_cv(data, n_splits=5)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "run_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "analysis_type": "lambda_orth external sensitivity training",
                "source_cohort": "LGG",
                "anchor_modality": "gene",
                "model_logic": "legacy",
                "lambda_orth_values": args.lambda_orth_values,
                "data_path": str(args.data_path.resolve()),
                "device": str(device),
            },
            file,
            indent=2,
        )

    for value in args.lambda_orth_values:
        setting_args = argparse.Namespace(**vars(args))
        setting_args.lambda_orth = float(value)
        setting_args.output_dir = args.output_dir / directory_name(float(value))
        setting_args.config = FULL_CONFIG.name
        print(f"\n{'=' * 80}\nTraining lambda_orth={value:.2f}\n{'=' * 80}")
        run_configuration(setting_args, FULL_CONFIG, splits, device)


if __name__ == "__main__":
    main()
