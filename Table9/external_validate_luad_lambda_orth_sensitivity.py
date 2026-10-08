"""Evaluate legacy-logic LGG lambda_orth checkpoints on the RF80 LUAD cohort."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models import MisaPTPGatedRec
from utils import CIndex_lifeline, auc_formula16


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def directory_name(value: float) -> str:
    return "lambda_orth_" + format(value, ".3f").rstrip("0").rstrip(".").replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="External LUAD sensitivity evaluation for legacy-logic LGG MAF-Surv models."
    )
    parser.add_argument(
        "--checkpoint-root", type=Path, default=SCRIPT_DIR / "lgg_lambda_orth_sensitivity_legacy_logic"
    )
    parser.add_argument("--luad-data-path", type=Path, default=PROJECT_ROOT / "Datasets" / "LUAD" / "RF80" / "luad.pkl")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "lgg_lambda_orth_sensitivity_legacy_logic" / "external_validation",
    )
    parser.add_argument("--lambda-orth-values", type=float, nargs="+", default=[0.0, 0.03, 0.05, 0.10])
    parser.add_argument("--gpu-id", type=int, default=0)
    return parser.parse_args()


def load_data(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"LUAD dataset is not found: {path}")
    with path.open("rb") as file:
        payload = pickle.load(file)
    data = payload.get("datasets")
    required = {"x_gene", "x_path", "x_cna", "censored", "survival"}
    if not isinstance(data, dict) or required.difference(data):
        raise KeyError("LUAD pickle must contain all required dataset fields.")
    sample_count = len(data["survival"])
    for key in required:
        if len(data[key]) != sample_count:
            raise ValueError(f"LUAD field {key} has an inconsistent sample count.")
    for key in ("x_gene", "x_path", "x_cna"):
        data[key] = np.asarray(data[key], dtype=np.float32)
        if data[key].shape != (sample_count, 80):
            raise ValueError(f"LUAD {key} must have shape ({sample_count}, 80), got {data[key].shape}.")
    return data


def evaluate(model: torch.nn.Module, data: dict, device: torch.device) -> tuple[float, float, np.ndarray]:
    with torch.no_grad():
        prediction, *_ = model(
            torch.as_tensor(data["x_gene"], dtype=torch.float32, device=device),
            torch.as_tensor(data["x_path"], dtype=torch.float32, device=device),
            torch.as_tensor(data["x_cna"], dtype=torch.float32, device=device),
        )
    risk = prediction.detach().cpu().numpy().reshape(-1)
    survival = np.asarray(data["survival"]).reshape(-1)
    censored = np.asarray(data["censored"]).reshape(-1)
    return float(CIndex_lifeline(risk, censored, survival)), float(auc_formula16(survival, censored, risk)), risk


def main() -> None:
    args = parse_args()
    if any(value < 0.0 for value in args.lambda_orth_values):
        raise ValueError("All lambda_orth values must be non-negative.")
    device = torch.device(f"cuda:{args.gpu_id}") if args.gpu_id >= 0 and torch.cuda.is_available() else torch.device("cpu")
    data = load_data(args.luad_data_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, prediction_frames = [], []

    for value in args.lambda_orth_values:
        setting_name = directory_name(float(value))
        setting_dir = args.checkpoint_root / setting_name / "maf_surv_total"
        setting_rows = []
        for fold in range(5):
            checkpoint_path = setting_dir / f"best_model_fold_{fold}.pt"
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Checkpoint is not found: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if checkpoint.get("ablation", {}).get("name") != "maf_surv_total":
                raise ValueError(f"Checkpoint is not a complete MAF-Surv model: {checkpoint_path}")
            actual_lambda = float(checkpoint.get("loss_weights", {}).get("lambda_orth", np.nan))
            if not np.isclose(actual_lambda, value):
                raise ValueError(f"Checkpoint lambda_orth={actual_lambda} does not match {value}: {checkpoint_path}")
            model = MisaPTPGatedRec(in_size=80, output_dim=1, hidden_size1=80).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            cindex, auc, risk = evaluate(model, data, device)
            row = {
                "lambda_orth": float(value), "setting": setting_name, "fold": fold,
                "checkpoint": str(checkpoint_path), "checkpoint_epoch": checkpoint.get("epoch"),
                "checkpoint_val_cindex": checkpoint.get("validation_cindex"), "cindex": cindex, "auc": auc,
            }
            rows.append(row)
            setting_rows.append(row)
            prediction_frames.append(pd.DataFrame({
                "lambda_orth": float(value), "setting": setting_name, "fold": fold,
                "survival": np.asarray(data["survival"]).reshape(-1),
                "censored": np.asarray(data["censored"]).reshape(-1), "risk_pred": risk,
            }))
        pd.DataFrame(setting_rows).to_csv(args.output_dir / f"{setting_name}_fold_results.csv", index=False)

    fold_df = pd.DataFrame(rows)
    summary_df = fold_df.groupby(["lambda_orth", "setting"], as_index=False).agg(
        cindex_mean=("cindex", "mean"), cindex_std=("cindex", "std"),
        auc_mean=("auc", "mean"), auc_std=("auc", "std"),
    ).sort_values("lambda_orth")
    fold_df.to_csv(args.output_dir / "external_lambda_orth_fold_results.csv", index=False)
    summary_df.to_csv(args.output_dir / "external_lambda_orth_summary.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(args.output_dir / "external_lambda_orth_predictions.csv", index=False)
    with (args.output_dir / "run_metadata.json").open("w", encoding="utf-8") as file:
        json.dump({
            "analysis_type": "lambda_orth external sensitivity evaluation", "source_cohort": "LGG",
            "target_cohort": "LUAD", "fine_tuning": False, "anchor_modality": "gene",
            "model_logic": "legacy", "lambda_orth_values": args.lambda_orth_values,
            "checkpoint_root": str(args.checkpoint_root.resolve()), "luad_data_path": str(args.luad_data_path.resolve()),
        }, file, indent=2)
    print(summary_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
