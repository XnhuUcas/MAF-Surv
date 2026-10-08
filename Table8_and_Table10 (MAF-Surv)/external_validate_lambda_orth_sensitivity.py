"""External LUAD evaluation of all legacy-logic LGG ablation checkpoints."""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from models import MisaPTPGatedRec
from utils import CIndex_lifeline, auc_formula16


CONFIG_ORDER = (
    "baseline_cox",
    "plus_recon",
    "plus_orth",
    "plus_r3gan",
    "plus_recon_orth",
    "plus_recon_r3gan",
    "plus_orth_r3gan",
    "maf_surv_total",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate all legacy-logic LGG ablation checkpoints on LUAD without fine-tuning."
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=SCRIPT_DIR / "lgg_ablation_legacy_logic",
        help="Root directory created by train_lgg_ablation_models.py.",
    )
    parser.add_argument(
        "--luad-data-path",
        type=Path,
        default=PROJECT_ROOT / "Datasets" / "LUAD" / "RF80" / "luad.pkl",
        help="Path to the 80-dimensional LUAD external-validation pickle.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "lgg_ablation_legacy_logic" / "external_validation",
        help="Directory for external predictions and metric summaries.",
    )
    parser.add_argument(
        "--config",
        choices=list(CONFIG_ORDER) + ["all"],
        default="all",
        help="Evaluate all ablation configurations or one configuration.",
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="CUDA device index; use -1 for CPU.")
    return parser.parse_args()


def set_seed(seed: int = 111) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset(path: Path) -> Dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"LUAD dataset is not found: {path}")
    with path.open("rb") as file:
        payload = pickle.load(file)
    if "datasets" not in payload:
        raise KeyError("The LUAD pickle must contain a top-level 'datasets' dictionary.")
    data = payload["datasets"]
    required = {"x_gene", "x_path", "x_cna", "censored", "survival"}
    missing = required.difference(data)
    if missing:
        raise KeyError(f"LUAD dataset misses required fields: {sorted(missing)}")
    sample_count = len(data["survival"])
    for key in required:
        if len(data[key]) != sample_count:
            raise ValueError(f"LUAD field {key} has {len(data[key])} rows, expected {sample_count}.")
    for key in ("x_gene", "x_path", "x_cna"):
        if np.asarray(data[key]).shape != (sample_count, 80):
            raise ValueError(f"LUAD {key} must have shape ({sample_count}, 80), got {np.asarray(data[key]).shape}.")
    return data


def load_model(
    checkpoint_path: Path,
    expected_config: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint is not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint does not contain model_state_dict: {checkpoint_path}")
    if checkpoint.get("source_cohort") != "LGG":
        raise ValueError(f"Checkpoint source cohort is not LGG: {checkpoint_path}")
    if checkpoint.get("anchor_modality") != "gene":
        raise ValueError(f"Checkpoint anchor is not gene: {checkpoint_path}")
    actual_config = checkpoint.get("ablation", {}).get("name")
    if actual_config != expected_config:
        raise ValueError(
            f"Checkpoint configuration={actual_config!r} does not match expected {expected_config!r}: {checkpoint_path}"
        )

    model = MisaPTPGatedRec(in_size=80, output_dim=1, hidden_size1=80).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def evaluate(model: torch.nn.Module, data: Dict[str, np.ndarray], device: torch.device) -> tuple[float, float, np.ndarray]:
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
    set_seed(111)
    device = torch.device(f"cuda:{args.gpu_id}") if args.gpu_id >= 0 and torch.cuda.is_available() else torch.device("cpu")
    data = load_dataset(args.luad_data_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_fold_rows: List[dict] = []
    all_prediction_frames: List[pd.DataFrame] = []
    total_start = time.time()
    selected_configs: Iterable[str] = CONFIG_ORDER if args.config == "all" else (args.config,)
    for config_name in selected_configs:
        setting_dir = args.checkpoint_root / config_name
        if not setting_dir.is_dir():
            raise FileNotFoundError(f"Configuration directory is not found: {setting_dir}")
        setting_start = time.time()
        setting_rows: List[dict] = []

        for fold in range(5):
            checkpoint_path = setting_dir / f"best_model_fold_{fold}.pt"
            model, checkpoint = load_model(checkpoint_path, config_name, device)
            cindex, auc, risk = evaluate(model, data, device)
            row = {
                "configuration": config_name,
                "fold": fold,
                "checkpoint": str(checkpoint_path),
                "checkpoint_epoch": checkpoint.get("epoch"),
                "checkpoint_val_cindex": checkpoint.get("validation_cindex"),
                "cindex": cindex,
                "auc": auc,
            }
            setting_rows.append(row)
            all_fold_rows.append(row)
            all_prediction_frames.append(
                pd.DataFrame(
                    {
                        "configuration": config_name,
                        "fold": fold,
                        "survival": np.asarray(data["survival"]).reshape(-1),
                        "censored": np.asarray(data["censored"]).reshape(-1),
                        "risk_pred": risk,
                    }
                )
            )

        setting_df = pd.DataFrame(setting_rows)
        setting_df.to_csv(args.output_dir / f"{config_name}_fold_results.csv", index=False)
        print(
            f"{config_name}: C-index={setting_df['cindex'].mean():.4f} +/- {setting_df['cindex'].std(ddof=1):.4f}; "
            f"AUC={setting_df['auc'].mean():.4f} +/- {setting_df['auc'].std(ddof=1):.4f}; "
            f"runtime={time.time() - setting_start:.2f}s"
        )

    fold_df = pd.DataFrame(all_fold_rows)
    summary_df = (
        fold_df.groupby("configuration", as_index=False)
        .agg(
            cindex_mean=("cindex", "mean"),
            cindex_std=("cindex", "std"),
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
        )
        .set_index("configuration")
        .reindex(list(selected_configs))
        .reset_index()
    )
    fold_df.to_csv(args.output_dir / "external_ablation_fold_results.csv", index=False)
    summary_df.to_csv(args.output_dir / "external_ablation_summary.csv", index=False)
    pd.concat(all_prediction_frames, ignore_index=True).to_csv(
        args.output_dir / "external_ablation_predictions.csv", index=False
    )
    with (args.output_dir / "run_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "analysis_type": "external ablation validation",
                "source_cohort": "LGG",
                "target_cohort": "LUAD",
                "fine_tuning": False,
                "anchor_modality": "gene",
                "configurations": list(selected_configs),
                "luad_data_path": str(args.luad_data_path.resolve()),
                "checkpoint_root": str(args.checkpoint_root.resolve()),
                "device": str(device),
                "total_runtime_seconds": round(time.time() - total_start, 2),
            },
            file,
            indent=2,
        )
    print("=" * 80)
    print("LGG-to-LUAD external ablation validation (legacy model logic)")
    print(summary_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
