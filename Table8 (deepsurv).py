import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("THEANO_FLAGS", "device=cpu,floatX=float32,cxx=,openmp=False")
if not hasattr(np, "bool"):
    np.bool = bool

try:
    from theano.tensor.signal import pool as theano_pool

    downsample_module = types.ModuleType("downsample")
    if hasattr(theano_pool, "pool_2d"):
        downsample_module.max_pool_2d = theano_pool.pool_2d
    sys.modules["theano.tensor.signal.downsample"] = downsample_module
except Exception:
    pass

import lasagne

from data_from_pkl import load_fused_splits_from_pkl
from deep_surv import DeepSurv, load_model_from_json


ROOT_DIR = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DeepSurv on LGG and use all 5 fold models for external validation on LUAD."
    )
    parser.add_argument("--lgg-pkl", type=str, default="Datasets/LGG/original_data.pkl")
    parser.add_argument("--luad-pkl", type=str, default="Datasets/LUAD/RF80/luad.pkl")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiment/LUAD/Deepsurv/RF80/external_validation",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=111)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


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


def to_ds(x, t, e):
    return {
        "x": x.astype("float32"),
        "t": t.astype("float32"),
        "e": e.astype("int32"),
    }


def load_fused_dataset_with_scaler(pkl_path: Path, scaler) -> dict:
    import pickle

    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    data = obj["datasets"] if isinstance(obj, dict) and "datasets" in obj else obj

    x_gene = np.asarray(data["x_gene"])
    x_cna = np.asarray(data["x_cna"])
    x_path = np.asarray(data["x_path"])
    x_all = np.hstack([x_gene, x_cna, x_path]).astype(np.float32)
    x_all = scaler.transform(x_all).astype(np.float32)

    return {
        "x": x_all,
        "t": np.asarray(data["survival"]).astype(np.float32),
        "e": np.asarray(data["censored"]).astype(np.int32),
    }


def save_model_with_params(model: DeepSurv, params, model_json: Path, model_weights: Path):
    if params is not None:
        lasagne.layers.set_all_param_values(model.network, params, trainable=True)
    model.save_model(str(model_json), str(model_weights))


def main():
    args = parse_args()

    lgg_pkl = resolve_path(args.lgg_pkl)
    luad_pkl = resolve_path(args.luad_pkl)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not lgg_pkl.exists():
        raise FileNotFoundError(f"LGG pkl not found: {lgg_pkl}")
    if not luad_pkl.exists():
        raise FileNotFoundError(f"LUAD pkl not found: {luad_pkl}")

    total_start_time = time.time()
    fold_records = []
    external_rows = []
    fold_artifacts = []

    for pack in load_fused_splits_from_pkl(
        lgg_pkl,
        n_splits=args.n_splits,
        val_size=args.val_size,
        random_state=args.random_state,
        scaler="standard",
        print_info=True,
    ):
        fold = pack["fold"]
        train_data = to_ds(pack["X_tr"], pack["T_tr"], pack["E_tr"])
        valid_data = to_ds(pack["X_va"], pack["T_va"], pack["E_va"])
        test_data = to_ds(pack["X_te"], pack["T_te"], pack["E_te"])

        hyperparams = {
            "n_in": pack["X_tr"].shape[1],
            "learning_rate": args.learning_rate,
        }
        model = DeepSurv(**hyperparams)
        history = model.train(train_data, valid_data)

        best_params = history.get("best_params", None)
        if best_params is not None:
            lasagne.layers.set_all_param_values(model.network, best_params, trainable=True)

        val_cindex = float(model.get_concordance_index(**valid_data))
        test_cindex = float(model.get_concordance_index(**test_data))
        val_scores = model.predict_risk(valid_data["x"])
        test_scores = model.predict_risk(test_data["x"])
        val_auc = float(auc_formula16(valid_data["t"], valid_data["e"], val_scores))
        test_auc = float(auc_formula16(test_data["t"], test_data["e"], test_scores))

        fold_records.append(
            {
                "fold": fold,
                "val_cindex": val_cindex,
                "test_cindex": test_cindex,
                "val_auc": val_auc,
                "test_auc": test_auc,
            }
        )

        print(
            f"[Fold {fold}] val C-index: {val_cindex:.4f} | test C-index: {test_cindex:.4f} | "
            f"val AUC: {val_auc:.4f} | test AUC: {test_auc:.4f}"
        )

        fold_artifacts.append(
            {
                "fold": fold,
                "model": model,
                "best_params": best_params,
                "scaler": pack["scaler"],
                "val_cindex": val_cindex,
                "val_auc": val_auc,
                "test_cindex": test_cindex,
                "test_auc": test_auc,
            }
        )

    fold_df = pd.DataFrame(fold_records)
    fold_df.to_csv(output_dir / "lgg_5fold_results.csv", index=False)

    for artifact in fold_artifacts:
        fold = artifact["fold"]
        model_json = output_dir / f"deepsurv_fold_{fold}.json"
        model_weights = output_dir / f"deepsurv_fold_{fold}.h5"
        scaler_npz = output_dir / f"deepsurv_fold_{fold}_scaler.npz"
        meta_json = output_dir / f"deepsurv_fold_{fold}_metadata.json"

        save_model_with_params(artifact["model"], artifact["best_params"], model_json, model_weights)
        np.savez(
            scaler_npz,
            mean=artifact["scaler"].mean_,
            scale=artifact["scaler"].scale_,
        )
        with open(meta_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "fold": int(fold),
                    "val_cindex": float(artifact["val_cindex"]),
                    "test_cindex": float(artifact["test_cindex"]),
                    "val_auc": float(artifact["val_auc"]),
                    "test_auc": float(artifact["test_auc"]),
                    "lgg_pkl": str(lgg_pkl),
                    "luad_pkl": str(luad_pkl),
                },
                f,
                indent=2,
            )

        external_model = load_model_from_json(str(model_json), str(model_weights))
        scaler_payload = np.load(scaler_npz)
        external_model.offset = scaler_payload["mean"].astype(np.float32)
        external_model.scale = scaler_payload["scale"].astype(np.float32)

        luad_data = load_fused_dataset_with_scaler(luad_pkl, artifact["scaler"])
        luad_scores = external_model.predict_risk(luad_data["x"])
        luad_cindex = float(external_model.get_concordance_index(**luad_data))
        luad_auc = float(auc_formula16(luad_data["t"], luad_data["e"], luad_scores))

        pd.DataFrame(
            {
                "risk_pred": luad_scores,
                "os_time": luad_data["t"],
                "os_status": luad_data["e"],
            }
        ).to_csv(output_dir / f"luad_external_validation_predictions_fold_{fold}.csv", index=False)

        external_rows.append(
            {
                "fold": int(fold),
                "val_cindex": float(artifact["val_cindex"]),
                "val_auc": float(artifact["val_auc"]),
                "luad_cindex": float(luad_cindex),
                "luad_auc": float(luad_auc),
            }
        )

    total_runtime = time.time() - total_start_time
    external_df = pd.DataFrame(external_rows)
    external_df.to_csv(output_dir / "luad_external_validation_5fold_results.csv", index=False)
    summary_df = pd.DataFrame(
        [
            {
                "num_folds": len(external_df),
                "luad_cindex_mean": float(external_df["luad_cindex"].mean()),
                "luad_cindex_std": float(external_df["luad_cindex"].std(ddof=1)),
                "luad_auc_mean": float(external_df["luad_auc"].mean()),
                "luad_auc_std": float(external_df["luad_auc"].std(ddof=1)),
                "total_runtime_seconds": float(total_runtime),
            }
        ]
    )
    summary_df.to_csv(output_dir / "luad_external_validation_summary.csv", index=False)

    print("\nLGG 5-fold models for external validation")
    print(f"LUAD C-index: {external_df['luad_cindex'].mean():.4f} ± {external_df['luad_cindex'].std(ddof=1):.4f}")
    print(f"LUAD AUC: {external_df['luad_auc'].mean():.4f} ± {external_df['luad_auc'].std(ddof=1):.4f}")
    print(f"Total runtime: {total_runtime:.2f} seconds")


if __name__ == "__main__":
    main()
