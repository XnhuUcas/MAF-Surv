import argparse
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from networks import MisaLMFGatedRec
from utils import CIndex_lifeline


ROOT_DIR = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Use the 5-fold LGG CAMR models for external validation on LUAD."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Optional manual checkpoint path. If provided, only this checkpoint is used.",
    )
    parser.add_argument(
        "--luad-pkl",
        type=str,
        default="Datasets/LUAD/RF80/luad.pkl",
        help="Path to the LUAD pkl file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiment/LUAD/CAMR/RF80/external_validation",
        help="Output directory.",
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU id")
    parser.add_argument("--seed", type=int, default=111, help="Random seed")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def auto_find_camr_checkpoints() -> list[Path]:
    model_dir = ROOT_DIR / "experiment" / "LGG" / "CAMR" / "models" / "LGG" / "CAMR"
    checkpoints = sorted(model_dir.glob("CAMR_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    return checkpoints


def auc_formula16(times, events, scores) -> float:
    times = np.asarray(times, dtype=float).reshape(-1)
    events = np.asarray(events, dtype=int).reshape(-1)
    scores = np.asarray(scores, dtype=float).reshape(-1)

    event_times = np.sort(np.unique(times[events == 1]))
    good_total = 0
    pair_total = 0

    for current_time in event_times:
        pos_idx = (events == 1) & (times < current_time)
        neg_idx = times > current_time
        n_pos = int(np.sum(pos_idx))
        n_neg = int(np.sum(neg_idx))
        if n_pos == 0 or n_neg == 0:
            continue

        pos_scores = scores[pos_idx]
        neg_scores = scores[neg_idx]
        good_total += int(np.sum(np.greater.outer(pos_scores, neg_scores)))
        pair_total += int(n_pos * n_neg)

    if pair_total == 0:
        return float("nan")
    return good_total / pair_total


def load_dataset(pkl_path: Path) -> dict:
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    return obj["datasets"]


def validate_dataset(data: dict) -> None:
    lengths = [
        len(data["x_gene"]),
        len(data["x_path"]),
        len(data["x_cna"]),
        len(data["censored"]),
        len(data["survival"]),
    ]
    if len(set(lengths)) != 1:
        raise ValueError(f"LUAD data lengths are inconsistent: {lengths}")


def load_checkpoint(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        opt = checkpoint.get("opt", None)
    else:
        state_dict = checkpoint
        opt = None
    return state_dict, opt


def build_model(state_dict: dict, checkpoint_opt, device: torch.device) -> torch.nn.Module:
    if checkpoint_opt is not None and hasattr(checkpoint_opt, "input_size"):
        input_size = int(checkpoint_opt.input_size)
        label_dim = int(getattr(checkpoint_opt, "label_dim", 1))
        dropout_rate = float(getattr(checkpoint_opt, "dropout_rate", 0.3))
    else:
        first_weight = state_dict["common.0.weight"]
        input_size = int(first_weight.shape[1])
        label_dim = 1
        dropout_rate = 0.3

    model = MisaLMFGatedRec(
        in_size=input_size,
        output_dim=label_dim,
        hidden_size1=input_size,
        dropout=dropout_rate,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_risk(model: torch.nn.Module, data: dict, device: torch.device) -> np.ndarray:
    x_gene = torch.as_tensor(data["x_gene"], dtype=torch.float32, device=device)
    x_path = torch.as_tensor(data["x_path"], dtype=torch.float32, device=device)
    x_cna = torch.as_tensor(data["x_cna"], dtype=torch.float32, device=device)

    with torch.no_grad():
        pred, *_ = model(x_gene, x_path, x_cna)

    return pred.detach().cpu().numpy().reshape(-1)


def main():
    args = parse_args()
    set_seed(args.seed)

    checkpoint_paths = []
    if args.checkpoint:
        checkpoint_path = resolve_path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint is not found: {checkpoint_path}")
        checkpoint_paths = [checkpoint_path]
    else:
        checkpoint_paths = auto_find_camr_checkpoints()
        if not checkpoint_paths:
            raise FileNotFoundError("Failed to find the LGG CAMR checkpoints automatically.")

    luad_pkl_path = resolve_path(args.luad_pkl)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not luad_pkl_path.exists():
        raise FileNotFoundError(f"LUAD pkl is not found: {luad_pkl_path}")

    device = (
        torch.device(f"cuda:{args.gpu_id}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    start_time = time.time()

    data = load_dataset(luad_pkl_path)
    validate_dataset(data)

    survival = np.asarray(data["survival"]).reshape(-1)
    censored = np.asarray(data["censored"]).reshape(-1)
    fold_rows = []
    prediction_frames = []
    for fold_idx, checkpoint_path in enumerate(checkpoint_paths):
        state_dict, checkpoint_opt = load_checkpoint(checkpoint_path, device)
        model = build_model(state_dict, checkpoint_opt, device)

        risk_pred = predict_risk(model, data, device)
        cindex = CIndex_lifeline(risk_pred, censored, survival)
        auc = auc_formula16(survival, censored, risk_pred)
        prediction_frames.append(
            pd.DataFrame(
                {
                    "fold": fold_idx,
                    "risk_pred": risk_pred,
                    "os_time": survival,
                    "os_status": censored,
                }
            )
        )
        fold_rows.append(
            {
                "fold": fold_idx,
                "checkpoint": str(checkpoint_path),
                "cindex": float(cindex),
                "auc": float(auc),
            }
        )
    runtime_seconds = time.time() - start_time

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(output_dir / "external_validation_5fold_results_camr.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        output_dir / "external_validation_predictions_5fold_camr.csv", index=False
    )

    summary_df = pd.DataFrame(
        [
            {
                "dataset": str(luad_pkl_path),
                "num_folds": len(checkpoint_paths),
                "cindex_mean": float(fold_df["cindex"].mean()),
                "cindex_std": float(fold_df["cindex"].std(ddof=1)),
                "auc_mean": float(fold_df["auc"].mean()),
                "auc_std": float(fold_df["auc"].std(ddof=1)),
                "runtime_seconds": float(runtime_seconds),
            }
        ]
    )
    summary_csv = output_dir / "external_validation_summary_camr.csv"
    summary_df.to_csv(summary_csv, index=False)

    print("CAMR 5-fold external validation on LUAD")
    print(f"Dataset: {luad_pkl_path}")
    print(f"Folds: {len(checkpoint_paths)}")
    print(f"C-index: {fold_df['cindex'].mean():.4f} ± {fold_df['cindex'].std(ddof=1):.4f}")
    print(f"AUC: {fold_df['auc'].mean():.4f} ± {fold_df['auc'].std(ddof=1):.4f}")
    print(f"Runtime: {runtime_seconds:.2f} seconds")
    print(f"Saved summary to: {summary_csv}")


if __name__ == "__main__":
    main()
