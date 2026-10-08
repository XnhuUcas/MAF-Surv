import argparse
import pickle
import time
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch

from models import MisaPTPGatedRec
from utils import CIndex_lifeline, auc_formula16


ROOT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Use the 5-fold LGG gene-anchor MAF-Surv models for external validation on LUAD."
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
        "--gene-feature-csv",
        type=str,
        default="Datasets/LUAD/RF80/gene_top80.csv",
        help="CSV used to read gene feature names.",
    )
    parser.add_argument(
        "--path-feature-csv",
        type=str,
        default="Datasets/LUAD/RF80/path_top80.csv",
        help="CSV used to read path feature names.",
    )
    parser.add_argument(
        "--cna-feature-csv",
        type=str,
        default="Datasets/LUAD/RF80/cna_top80.csv",
        help="CSV used to read CNA feature names.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiment/LUAD/RF80+LGG_model_5fold",
        help="Output directory.",
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--background-size", type=int, default=50)
    parser.add_argument("--explain-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=111)
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


def auto_find_gene_anchor_checkpoints() -> list[Path]:
    metrics_candidates = sorted(
        ROOT_DIR.glob("experiment/LGG/**/test_anchor-gene/detailed_results.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    for metrics_csv in metrics_candidates:
        metrics_path_str = str(metrics_csv)
        if "change_tsne" in metrics_path_str:
            continue

        try:
            metrics_df = pd.read_csv(metrics_csv)
        except Exception:
            continue

        if "fold" not in metrics_df.columns:
            continue

        result_dir = metrics_csv.parent
        checkpoint_dir = Path(str(result_dir).replace("\\new_results\\", "\\new_models\\"))
        fold_checkpoints = []
        for fold in sorted(metrics_df["fold"].astype(int).unique()):
            checkpoint = checkpoint_dir / f"best_model_fold_{fold}.pt"
            if checkpoint.exists():
                fold_checkpoints.append(checkpoint)
        if fold_checkpoints:
            return fold_checkpoints

    return []


def load_dataset(pkl_path: Path) -> dict:
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    return obj["datasets"]


def validate_dataset_alignment(data: dict) -> None:
    lengths = {
        "x_gene": len(data["x_gene"]),
        "x_path": len(data["x_path"]),
        "x_cna": len(data["x_cna"]),
        "censored": len(data["censored"]),
        "survival": len(data["survival"]),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(
            "LUAD pkl has inconsistent sample counts: "
            + ", ".join([f"{k}={v}" for k, v in lengths.items()])
        )


def load_feature_names(csv_path: Path, feature_count: int, prefix: str) -> list[str]:
    if csv_path.exists():
        df = pd.read_csv(csv_path, nrows=1)
        feature_names = list(df.columns[1:])
        if len(feature_names) == feature_count:
            return feature_names
    return [f"{prefix}_{i}" for i in range(feature_count)]


class RiskOnlyModel(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, x_gene, x_path, x_cna):
        pred, *_ = self.base_model(x_gene, x_path, x_cna)
        return pred


def build_model(input_size: int, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = MisaPTPGatedRec(input_size, 1, hidden_size1=input_size).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_risk(
    model: torch.nn.Module,
    x_gene: np.ndarray,
    x_path: np.ndarray,
    x_cna: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    with torch.no_grad():
        pred, *_ = model(
            torch.as_tensor(x_gene, dtype=torch.float32, device=device),
            torch.as_tensor(x_path, dtype=torch.float32, device=device),
            torch.as_tensor(x_cna, dtype=torch.float32, device=device),
        )
    return pred.detach().cpu().numpy().reshape(-1)


def evaluate_external(model: torch.nn.Module, data: dict, device: torch.device):
    risk_pred = predict_risk(model, data["x_gene"], data["x_path"], data["x_cna"], device)
    survival = np.asarray(data["survival"]).reshape(-1)
    censored = np.asarray(data["censored"]).reshape(-1)
    cindex = CIndex_lifeline(risk_pred, censored, survival)
    auc = auc_formula16(survival, censored, risk_pred)
    return cindex, auc, risk_pred


def sample_indices(n_samples: int, sample_size: int, seed: int) -> np.ndarray:
    if sample_size >= n_samples:
        return np.arange(n_samples)
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(n_samples, size=sample_size, replace=False))


def normalize_shap_outputs(shap_values):
    if isinstance(shap_values, list) and len(shap_values) == 1 and isinstance(shap_values[0], list):
        shap_values = shap_values[0]

    normalized = []
    for values in shap_values:
        arr = np.asarray(values)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.squeeze(arr, axis=-1)
        normalized.append(arr)
    return normalized


def compute_shap_values(
    model: torch.nn.Module,
    data: dict,
    device: torch.device,
    background_size: int,
    explain_size: int,
    seed: int,
):
    background_idx = sample_indices(len(data["x_gene"]), background_size, seed)
    explain_idx = sample_indices(len(data["x_gene"]), explain_size, seed + 1)

    background = [
        torch.as_tensor(data["x_gene"][background_idx], dtype=torch.float32, device=device),
        torch.as_tensor(data["x_path"][background_idx], dtype=torch.float32, device=device),
        torch.as_tensor(data["x_cna"][background_idx], dtype=torch.float32, device=device),
    ]
    explain_inputs = [
        torch.as_tensor(data["x_gene"][explain_idx], dtype=torch.float32, device=device),
        torch.as_tensor(data["x_path"][explain_idx], dtype=torch.float32, device=device),
        torch.as_tensor(data["x_cna"][explain_idx], dtype=torch.float32, device=device),
    ]

    risk_model = RiskOnlyModel(model).to(device)
    explainer = shap.DeepExplainer(risk_model, background)
    try:
        shap_values = explainer.shap_values(explain_inputs, check_additivity=False)
    except TypeError:
        shap_values = explainer.shap_values(explain_inputs)

    shap_values = normalize_shap_outputs(shap_values)
    explain_arrays = [tensor.detach().cpu().numpy() for tensor in explain_inputs]
    return shap_values, explain_arrays, explain_idx


def save_topk_csv(shap_values: np.ndarray, feature_names: list[str], out_csv: Path, top_k: int) -> pd.DataFrame:
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    summary_df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    summary_df.to_csv(out_csv, index=False)
    return summary_df.head(top_k).reset_index(drop=True)


def save_summary_plot(
    shap_values: np.ndarray,
    feature_values: np.ndarray,
    feature_names: list[str],
    title: str,
    out_png: Path,
    top_k: int,
) -> None:
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        feature_values,
        feature_names=feature_names,
        max_display=top_k,
        show=False,
    )
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


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
        checkpoint_paths = auto_find_gene_anchor_checkpoints()
        if not checkpoint_paths:
            raise FileNotFoundError("Failed to find the LGG gene-anchor checkpoints automatically.")

    luad_pkl_path = resolve_path(args.luad_pkl)
    gene_csv_path = resolve_path(args.gene_feature_csv)
    path_csv_path = resolve_path(args.path_feature_csv)
    cna_csv_path = resolve_path(args.cna_feature_csv)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = (
        torch.device(f"cuda:{args.gpu_id}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    total_start = time.time()
    data = load_dataset(luad_pkl_path)
    validate_dataset_alignment(data)
    input_size = int(np.asarray(data["x_gene"]).shape[1])

    gene_feature_names = load_feature_names(gene_csv_path, input_size, "gene")
    path_feature_names = load_feature_names(path_csv_path, int(np.asarray(data["x_path"]).shape[1]), "path")
    cna_feature_names = load_feature_names(cna_csv_path, int(np.asarray(data["x_cna"]).shape[1]), "cna")

    fold_rows = []
    prediction_frames = []
    for fold_idx, checkpoint_path in enumerate(checkpoint_paths):
        model = build_model(input_size, checkpoint_path, device)
        eval_start = time.time()
        cindex, auc, risk_pred = evaluate_external(model, data, device)
        eval_runtime = time.time() - eval_start

        prediction_frames.append(
            pd.DataFrame(
                {
                    "fold": fold_idx,
                    "os_time": np.asarray(data["survival"]).reshape(-1),
                    "os_status": np.asarray(data["censored"]).reshape(-1),
                    "risk_pred": risk_pred,
                }
            )
        )
        fold_rows.append(
            {
                "fold": fold_idx,
                "checkpoint": str(checkpoint_path),
                "cindex": round(float(cindex), 6),
                "auc": round(float(auc), 6),
                "evaluation_runtime_seconds": round(float(eval_runtime), 2),
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(output_dir / "external_validation_5fold_results.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        output_dir / "luad_external_predictions_5fold.csv", index=False
    )

    total_runtime = time.time() - total_start

    summary_df = pd.DataFrame(
        [
            {
                "luad_pkl": str(luad_pkl_path),
                "num_samples": len(data["x_gene"]),
                "num_folds": len(checkpoint_paths),
                "cindex_mean": round(float(fold_df["cindex"].mean()), 6),
                "cindex_std": round(float(fold_df["cindex"].std(ddof=1)), 6),
                "auc_mean": round(float(fold_df["auc"].mean()), 6),
                "auc_std": round(float(fold_df["auc"].std(ddof=1)), 6),
                "total_runtime_seconds": round(float(total_runtime), 2),
            }
        ]
    )
    summary_df.to_csv(output_dir / "external_validation_summary.csv", index=False)

    print("=" * 80)
    print("LUAD External Validation (5-fold)")
    print("=" * 80)
    print(f"Dataset: {luad_pkl_path}")
    print(f"Folds: {len(checkpoint_paths)}")
    print(f"C-index: {fold_df['cindex'].mean():.4f} ± {fold_df['cindex'].std(ddof=1):.4f}")
    print(f"AUC: {fold_df['auc'].mean():.4f} ± {fold_df['auc'].std(ddof=1):.4f}")
    print(f"Total runtime: {total_runtime:.2f} seconds")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
