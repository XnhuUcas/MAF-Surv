"""Train all LGG MAF-Surv ablation configurations using the legacy model logic.

Every configuration is trained on the same five folds.  The checkpoint with the
highest validation C-index in each fold is saved for direct LUAD inference.
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
# The local models.py intentionally preserves the legacy implementation used in
# the submitted experiments; do not import the root-level model definition.
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from dataset import GraphFusionDatasetLoader
from losses import CoxLoss2, DiffLoss, MSE, regularize_weights
from models import Discriminator, MisaPTPGatedRec
from optimizers import define_optimizer, define_scheduler
from utils import CIndex_lifeline, auc_formula16, cox_log_rank, split_data_cv


@dataclass(frozen=True)
class AblationConfig:
    """One ablation setting; L_reg remains enabled in every configuration."""

    name: str
    use_recon: bool
    use_orth: bool
    use_r3gan: bool


ABLATION_CONFIGS: Tuple[AblationConfig, ...] = (
    AblationConfig("baseline_cox", False, False, False),
    AblationConfig("plus_recon", True, False, False),
    AblationConfig("plus_orth", False, True, False),
    AblationConfig("plus_r3gan", False, False, True),
    AblationConfig("plus_recon_orth", True, True, False),
    AblationConfig("plus_recon_r3gan", True, False, True),
    AblationConfig("plus_orth_r3gan", False, True, True),
    AblationConfig("maf_surv_total", True, True, True),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Five-fold LGG training for all MAF-Surv ablation configurations."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=PROJECT_ROOT / "Datasets" / "LGG" / "original_data.pkl",
        help="Path to the original 80-dimensional LGG pickle dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "lgg_ablation_legacy_logic",
        help="Directory for all ablation checkpoints, predictions, and summaries.",
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="CUDA device index; use -1 for CPU.")
    parser.add_argument("--epochs", type=int, default=80, help="Maximum epochs per fold.")
    parser.add_argument("--patience", type=int, default=70, help="Early-stopping patience in epochs.")
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.005,
        help="Minimum validation C-index improvement used only for early stopping.",
    )
    parser.add_argument("--learning-rate", type=float, default=0.0004)
    parser.add_argument("--lambda-reg", type=float, default=1e-5)
    # These are the loss weights recorded for the original LGG MAF-Surv run.
    parser.add_argument("--lambda-recon", type=float, default=0.3)
    parser.add_argument("--lambda-orth", type=float, default=0.1)
    parser.add_argument("--lambda-r3gan", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--config",
        choices=[config.name for config in ABLATION_CONFIGS] + ["all"],
        default="all",
        help="Run all configurations or one named configuration.",
    )
    return parser.parse_args()


def set_seed(seed: int = 111) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_anchor_samples(
    gene_common: torch.Tensor,
    path_common: torch.Tensor,
    cna_common: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use gene as the sole reference modality for this external-validation run."""

    return gene_common, path_common, cna_common


def zero_centered_gradient_penalty(samples: torch.Tensor, critics: torch.Tensor) -> torch.Tensor:
    gradients, = torch.autograd.grad(
        outputs=critics.sum(),
        inputs=samples,
        create_graph=True,
        retain_graph=True,
    )
    return gradients.square().sum(1)


def evaluate(
    model: nn.Module,
    split_data: Dict[str, np.ndarray],
    device: torch.device,
    num_workers: int,
) -> Dict[str, object]:
    """Evaluate one split using the model state currently loaded in memory."""

    model.eval()
    loader = DataLoader(
        GraphFusionDatasetLoader({"evaluation": split_data}, "evaluation"),
        batch_size=len(split_data["x_gene"]),
        shuffle=False,
        num_workers=num_workers,
    )
    all_risk, all_censor, all_survival = [], [], []
    cox_loss_sum = 0.0

    with torch.no_grad():
        for x_gene, x_path, x_cna, censor, survival in loader:
            x_gene = x_gene.view(x_gene.size(0), -1).to(device)
            x_path = x_path.view(x_path.size(0), -1).to(device)
            x_cna = x_cna.view(x_cna.size(0), -1).to(device)
            censor = censor.to(device)

            prediction, *_ = model(x_gene, x_path, x_cna)
            cox_loss_sum += float(CoxLoss2(survival, censor, prediction, device).item())
            all_risk.append(prediction.detach().cpu().numpy().reshape(-1))
            all_censor.append(censor.detach().cpu().numpy().reshape(-1))
            all_survival.append(survival.detach().cpu().numpy().reshape(-1))

    risk = np.concatenate(all_risk)
    censor = np.concatenate(all_censor)
    survival = np.concatenate(all_survival)
    return {
        "loss_cox": cox_loss_sum / len(loader.dataset),
        "cindex": float(CIndex_lifeline(risk, censor, survival)),
        "auc": float(auc_formula16(survival, censor, risk)),
        "logrank_pvalue": float(cox_log_rank(risk, censor, survival)),
        "risk": risk,
        "censored": censor,
        "survival": survival,
    }


def make_optimizer_options(args: argparse.Namespace) -> argparse.Namespace:
    """Provide the fields expected by the historical optimizer helpers."""

    return argparse.Namespace(
        optimizer_type="adam",
        lr=args.learning_rate,
        beta1=0.9,
        beta2=0.999,
        weight_decay=args.weight_decay,
        lr_policy="linear",
        epoch_count=1,
        niter=0,
        niter_decay=args.epochs,
        lr_decay_iters=50,
    )


def selected_weights(args: argparse.Namespace, config: AblationConfig) -> Dict[str, float]:
    return {
        "lambda_reg": args.lambda_reg,
        "lambda_recon": args.lambda_recon if config.use_recon else 0.0,
        "lambda_orth": args.lambda_orth if config.use_orth else 0.0,
        "lambda_r3gan": args.lambda_r3gan if config.use_r3gan else 0.0,
    }


def train_one_fold(
    args: argparse.Namespace,
    config: AblationConfig,
    fold_index: int,
    fold_data: Dict[str, Dict[str, np.ndarray]],
    device: torch.device,
    checkpoint_path: Path,
) -> Tuple[nn.Module, Dict[str, object], Dict[str, float]]:
    """Train one fold and save the exact validation-selected checkpoint."""

    set_seed(111)
    weights = selected_weights(args, config)
    optimizer_options = make_optimizer_options(args)

    # LGG and the RF80 LUAD external cohort have 80 features per modality.
    model = MisaPTPGatedRec(in_size=80, output_dim=1, hidden_size1=80).to(device)
    discriminator = Discriminator(60).to(device) if config.use_r3gan else None
    optimizer = define_optimizer(optimizer_options, model)
    scheduler = define_scheduler(optimizer_options, optimizer)
    discriminator_optimizer = (
        torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
        if discriminator is not None else None
    )
    reconstruction_loss = MSE()
    orthogonality_loss = DiffLoss()

    train_loader = DataLoader(
        GraphFusionDatasetLoader(fold_data, "train"),
        batch_size=len(fold_data["train"]["x_gene"]),
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Match the historical checkpoint policy: replace a saved checkpoint only
    # after a validation C-index improvement greater than ``min_delta``.
    best_val_cindex = 0.0
    best_epoch = None
    best_model_state = None
    best_discriminator_state = None
    epochs_without_improvement = 0
    history = []
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in tqdm(range(1, args.epochs + 1), desc=f"{config.name} fold {fold_index}", leave=False):
        model.train()
        if discriminator is not None:
            discriminator.train()

        for x_gene, x_path, x_cna, censor, survival in train_loader:
            x_gene = x_gene.view(x_gene.size(0), -1).to(device)
            x_path = x_path.view(x_path.size(0), -1).to(device)
            x_cna = x_cna.view(x_cna.size(0), -1).to(device)
            censor = censor.to(device)

            prediction, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, _ = model(
                x_gene, x_path, x_cna
            )

            if discriminator is not None:
                discriminator_optimizer.zero_grad()
                real, fake_1, fake_2 = get_anchor_samples(x7, x8, x9)
                real = real.detach().requires_grad_(True)
                fake_1 = fake_1.detach().requires_grad_(True)
                fake_2 = fake_2.detach().requires_grad_(True)
                real_logits = discriminator(real)
                fake_logits_1 = discriminator(fake_1)
                fake_logits_2 = discriminator(fake_2)
                discriminator_adversarial = (
                    nn.functional.softplus(-(fake_logits_1 - real_logits)).mean()
                    + nn.functional.softplus(-(fake_logits_2 - real_logits)).mean()
                ) / 2.0
                r1 = zero_centered_gradient_penalty(real, real_logits).mean()
                r2_1 = zero_centered_gradient_penalty(fake_1, fake_logits_1).mean()
                r2_2 = zero_centered_gradient_penalty(fake_2, fake_logits_2).mean()
                discriminator_loss = discriminator_adversarial + 5.0 * (r1 + r2_1 + r2_2)
                discriminator_loss.backward()
                discriminator_optimizer.step()

            optimizer.zero_grad()
            if discriminator is not None:
                real, fake_1, _ = get_anchor_samples(x7, x8, x9)
                generator_r3gan_loss = nn.functional.softplus(discriminator(fake_1) - discriminator(real)).mean()
            else:
                generator_r3gan_loss = torch.zeros((), device=device)

            reconstruction = reconstruction_loss(x1, x2) + reconstruction_loss(x3, x4) + reconstruction_loss(x5, x6)
            orthogonality = (orthogonality_loss(x7, x10) + orthogonality_loss(x8, x11) + orthogonality_loss(x9, x12)) / 3.0
            cox = CoxLoss2(survival, censor, prediction, device)
            regularization = regularize_weights(model)
            total = (
                cox
                + weights["lambda_reg"] * regularization
                + weights["lambda_recon"] * reconstruction
                + weights["lambda_orth"] * orthogonality
                + weights["lambda_r3gan"] * generator_r3gan_loss
            )
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()
        validation = evaluate(model, fold_data["val"], device, args.num_workers)
        history.append({"epoch": epoch, "val_cindex": validation["cindex"], "val_auc": validation["auc"]})

        if (
            np.isfinite(validation["cindex"])
            and validation["cindex"] > best_val_cindex + args.min_delta
        ):
            best_val_cindex = validation["cindex"]
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            best_discriminator_state = copy.deepcopy(discriminator.state_dict()) if discriminator is not None else None
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": best_model_state,
                    "discriminator_state_dict": best_discriminator_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "discriminator_optimizer_state_dict": (
                        discriminator_optimizer.state_dict() if discriminator_optimizer is not None else None
                    ),
                    "epoch": best_epoch,
                    "validation_cindex": best_val_cindex,
                    "anchor_modality": "gene",
                    "ablation": asdict(config),
                    "loss_weights": weights,
                    "source_cohort": "LGG",
                    "input_dimensions": {"gene": 80, "path": 80, "cna": 80},
                },
                checkpoint_path,
            )

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            break

    if best_model_state is None:
        raise RuntimeError(f"No finite validation C-index is produced for {config.name}, fold {fold_index}.")

    model.load_state_dict(best_model_state)
    final_test = evaluate(model, fold_data["test"], device, args.num_workers)
    fold_result = {
        "fold": fold_index,
        "best_epoch": best_epoch,
        "best_val_cindex": float(best_val_cindex),
        "test_cindex": final_test["cindex"],
        "test_auc": final_test["auc"],
        "test_logrank_pvalue": final_test["logrank_pvalue"],
        "epochs_completed": len(history),
        "history": history,
    }
    return model, fold_result, final_test


def validate_dataset(data: Dict[str, np.ndarray]) -> None:
    required = {"x_gene", "x_path", "x_cna", "censored", "survival"}
    missing = required.difference(data)
    if missing:
        raise KeyError(f"Dataset misses required fields: {sorted(missing)}")
    sample_count = len(data["survival"])
    for key in required:
        if len(data[key]) != sample_count:
            raise ValueError(f"Field {key} has {len(data[key])} rows, expected {sample_count}.")
    for key in ("x_gene", "x_path", "x_cna"):
        if np.asarray(data[key]).shape != (sample_count, 80):
            raise ValueError(f"{key} must have shape ({sample_count}, 80), got {np.asarray(data[key]).shape}.")


def run_configuration(
    args: argparse.Namespace,
    config: AblationConfig,
    splits: Dict[int, Dict[str, Dict[str, np.ndarray]]],
    device: torch.device,
) -> None:
    config_dir = args.output_dir / config.name
    config_dir.mkdir(parents=True, exist_ok=True)
    with (config_dir / "configuration.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "source_cohort": "LGG",
                "anchor_modality": "gene",
                "analysis_type": "external ablation training",
                "ablation": asdict(config),
                "loss_weights": selected_weights(args, config),
                "data_path": str(args.data_path.resolve()),
                "seed": 111,
                "fold_count": len(splits),
            },
            file,
            indent=2,
        )

    fold_rows = []
    all_predictions = []
    total_start = time.time()
    for fold_index, fold_data in splits.items():
        fold_start = time.time()
        checkpoint_path = config_dir / f"best_model_fold_{fold_index}.pt"
        _, fold_result, test_result = train_one_fold(
            args, config, fold_index, fold_data, device, checkpoint_path
        )
        fold_result["runtime_seconds"] = round(time.time() - fold_start, 2)
        fold_rows.append(fold_result)
        pd.DataFrame(
            {
                "fold": fold_index,
                "risk_pred": test_result["risk"],
                "survival": test_result["survival"],
                "censored": test_result["censored"],
            }
        ).to_csv(config_dir / f"test_predictions_fold_{fold_index}.csv", index=False)
        all_predictions.append(pd.read_csv(config_dir / f"test_predictions_fold_{fold_index}.csv"))
        print(
            f"{config.name} | fold {fold_index}: "
            f"C-index={fold_result['test_cindex']:.4f}, AUC={fold_result['test_auc']:.4f}, "
            f"best epoch={fold_result['best_epoch']}"
        )

    metrics = pd.DataFrame(fold_rows)
    metrics.to_csv(config_dir / "fold_metrics.csv", index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(config_dir / "test_predictions_5fold.csv", index=False)
    summary = {
        "analysis_type": "external ablation training",
        "configuration": config.name,
        "anchor_modality": "gene",
        "cindex_mean": float(metrics["test_cindex"].mean()),
        "cindex_std": float(metrics["test_cindex"].std(ddof=1)),
        "auc_mean": float(metrics["test_auc"].mean()),
        "auc_std": float(metrics["test_auc"].std(ddof=1)),
        "total_runtime_seconds": round(time.time() - total_start, 2),
    }
    with (config_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(
        f"{config.name} summary: C-index={summary['cindex_mean']:.4f} +/- {summary['cindex_std']:.4f}; "
        f"AUC={summary['auc_mean']:.4f} +/- {summary['auc_std']:.4f}"
    )


def main() -> None:
    args = parse_args()
    if not args.data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.data_path}")
    if args.gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.data_path.open("rb") as file:
        payload = pickle.load(file)
    if "datasets" not in payload:
        raise KeyError("The pickle must contain a top-level 'datasets' dictionary.")
    data = payload["datasets"]
    validate_dataset(data)
    splits = split_data_cv(data, n_splits=5)

    selected_configs: Iterable[AblationConfig]
    if args.config == "all":
        selected_configs = ABLATION_CONFIGS
    else:
        selected_configs = (next(item for item in ABLATION_CONFIGS if item.name == args.config),)

    run_metadata = {
        "source_cohort": "LGG",
        "data_path": str(args.data_path.resolve()),
        "device": str(device),
        "anchor_modality": "gene",
        "analysis_type": "external ablation training",
        "arguments": vars(args) | {"data_path": str(args.data_path), "output_dir": str(args.output_dir)},
    }
    with (args.output_dir / "run_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(run_metadata, file, indent=2)

    for config in selected_configs:
        run_configuration(args, config, splits, device)


if __name__ == "__main__":
    main()
