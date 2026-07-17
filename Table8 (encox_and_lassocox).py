import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sksurv.linear_model import CoxnetSurvivalAnalysis

from encox_and_lassocox import (
    ALPHA_MIN_RATIO,
    L1_RATIO,
    MAX_ITER,
    N_ALPHAS,
    N_FOLDS,
    TOL,
    auc_formula16,
    cindex_formula15,
    ensure_dir,
    fit_final_with_alpha_backoff,
    fuse_modalities,
    load_data_from_pkl,
    make_y,
    model_name_from_ratio,
    pick_best_alpha_by_validation,
)
from utils import split_data_cv


LGG_PKL_PATH = Path("Datasets") / "LGG" / "original_data.pkl"
LUAD_PKL_PATH = Path("Datasets") / "LUAD" / "RF80" / "luad.pkl"

RESULTS_ROOT = Path("experiment") / "LUAD" / "encox and lassocox" / "RF80" / "external_validation"
EXP_NAME = "RF80+LGG_model"


def fit_one_fold_with_metadata(train_split, val_split, test_split, l1_ratio):
    x_train = fuse_modalities(train_split)
    x_val = fuse_modalities(val_split)
    x_test = fuse_modalities(test_split)

    t_train = np.asarray(train_split["survival"], dtype=float)
    e_train = np.asarray(train_split["censored"], dtype=int)
    t_val = np.asarray(val_split["survival"], dtype=float)
    e_val = np.asarray(val_split["censored"], dtype=int)
    t_test = np.asarray(test_split["survival"], dtype=float)
    e_test = np.asarray(test_split["censored"], dtype=int)

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(x_train)
    x_train_std = scaler.transform(x_train).astype(np.float32)
    x_val_std = scaler.transform(x_val).astype(np.float32)

    y_train = make_y(t_train, e_train)
    path_estimator = CoxnetSurvivalAnalysis(
        l1_ratio=l1_ratio,
        n_alphas=N_ALPHAS,
        alpha_min_ratio=ALPHA_MIN_RATIO,
        normalize=False,
        tol=TOL,
        max_iter=MAX_ITER,
        fit_baseline_model=False,
    ).fit(x_train_std, y_train)

    best_alpha, cindex_values = pick_best_alpha_by_validation(path_estimator, x_val_std, t_val, e_val)
    if best_alpha is None:
        raise ValueError("Failed to select a valid alpha on the validation split.")

    path_alphas = np.asarray(path_estimator.alphas_, dtype=float)
    best_idx = int(np.nanargmax(cindex_values))
    candidate_alphas = list(path_alphas[best_idx::-1])

    x_train_val = np.vstack([x_train, x_val]).astype(np.float32)
    t_train_val = np.concatenate([t_train, t_val]).astype(float)
    e_train_val = np.concatenate([e_train, e_val]).astype(int)

    scaler_final = StandardScaler().fit(x_train_val)
    x_train_val_std = scaler_final.transform(x_train_val).astype(np.float32)
    x_test_std = scaler_final.transform(x_test).astype(np.float32)
    x_val_final_std = scaler_final.transform(x_val).astype(np.float32)

    y_train_val = make_y(t_train_val, e_train_val)
    final_estimator, fitted_alpha = fit_final_with_alpha_backoff(
        x_train_val_std=x_train_val_std,
        y_train_val=y_train_val,
        candidate_alphas=candidate_alphas,
        l1_ratio=l1_ratio,
    )

    val_scores = final_estimator.predict(x_val_final_std)
    val_cindex = cindex_formula15(t_val, e_val, val_scores)
    val_auc = auc_formula16(t_val, e_val, val_scores)

    test_scores = final_estimator.predict(x_test_std)
    test_cindex = cindex_formula15(t_test, e_test, test_scores)
    test_auc = auc_formula16(t_test, e_test, test_scores)

    return {
        "estimator": final_estimator,
        "scaler": scaler_final,
        "risk_pred": np.asarray(test_scores, dtype=float),
        "os_time": np.asarray(t_test, dtype=float),
        "os_status": np.asarray(e_test, dtype=int),
        "cindex": float(test_cindex),
        "auc": float(test_auc),
        "val_cindex": float(val_cindex),
        "val_auc": float(val_auc),
        "selected_alpha": float(fitted_alpha),
        "initial_best_alpha": float(best_alpha),
    }


def evaluate_external(estimator, scaler, external_data):
    x_external = fuse_modalities(external_data)
    x_external_std = scaler.transform(x_external).astype(np.float32)
    risk_pred = estimator.predict(x_external_std)

    t_external = np.asarray(external_data["survival"], dtype=float)
    e_external = np.asarray(external_data["censored"], dtype=int)

    cindex = cindex_formula15(t_external, e_external, risk_pred)
    auc = auc_formula16(t_external, e_external, risk_pred)

    return {
        "risk_pred": np.asarray(risk_pred, dtype=float),
        "os_time": t_external,
        "os_status": e_external,
        "cindex": float(cindex),
        "auc": float(auc),
    }


def main():
    lgg_data = load_data_from_pkl(LGG_PKL_PATH)
    luad_data = load_data_from_pkl(LUAD_PKL_PATH)

    cv_splits = split_data_cv(lgg_data, n_splits=N_FOLDS)
    model_name = model_name_from_ratio(L1_RATIO)

    result_dir = RESULTS_ROOT / EXP_NAME / model_name
    ensure_dir(result_dir)

    total_start_time = time.time()
    fold_rows = []
    external_rows = []

    print("\n" + "=" * 80)
    print(f"RUNNING {model_name.upper()} LGG TRAINING + LUAD EXTERNAL VALIDATION")
    print("=" * 80)
    print(f"[INFO] LGG pkl = {LGG_PKL_PATH}")
    print(f"[INFO] LUAD pkl = {LUAD_PKL_PATH}")
    print(f"[INFO] l1_ratio = {L1_RATIO}")
    print(f"[INFO] result_dir = {result_dir}")

    for fold_idx, split_data_dict in cv_splits.items():
        fold_start_time = time.time()
        print("\n" + "-" * 80)
        print(f"Fold {fold_idx + 1}/{N_FOLDS}")
        print("-" * 80)

        fold_result = fit_one_fold_with_metadata(
            train_split=split_data_dict["train"],
            val_split=split_data_dict["val"],
            test_split=split_data_dict["test"],
            l1_ratio=L1_RATIO,
        )

        fold_runtime_seconds = time.time() - fold_start_time
        fold_rows.append(
            {
                "fold": fold_idx,
                "val_cindex": fold_result["val_cindex"],
                "val_auc": fold_result["val_auc"],
                "test_cindex": fold_result["cindex"],
                "test_auc": fold_result["auc"],
                "selected_alpha": fold_result["selected_alpha"],
                "initial_best_alpha": fold_result["initial_best_alpha"],
                "fold_runtime_seconds": round(fold_runtime_seconds, 2),
                "fold_runtime_minutes": round(fold_runtime_seconds / 60.0, 2),
            }
        )

        print(
            f"[Fold {fold_idx + 1}] "
            f"val C-index={fold_result['val_cindex']:.4f} | "
            f"val AUC={fold_result['val_auc']:.4f} | "
            f"test C-index={fold_result['cindex']:.4f} | "
            f"test AUC={fold_result['auc']:.4f} | "
            f"alpha={fold_result['selected_alpha']:.3e} | "
            f"runtime={fold_runtime_seconds:.2f}s"
        )

        external_result = evaluate_external(
            estimator=fold_result["estimator"],
            scaler=fold_result["scaler"],
            external_data=luad_data,
        )

        with open(result_dir / f"model_fold_{fold_idx}.pkl", "wb") as f:
            pickle.dump(fold_result["estimator"], f)
        with open(result_dir / f"scaler_fold_{fold_idx}.pkl", "wb") as f:
            pickle.dump(fold_result["scaler"], f)

        pd.DataFrame(
            {
                "os_time": external_result["os_time"],
                "os_status": external_result["os_status"],
                "risk_pred": external_result["risk_pred"],
            }
        ).to_csv(result_dir / f"luad_external_predictions_fold_{fold_idx}.csv", index=False)

        external_rows.append(
            {
                "fold": fold_idx,
                "val_cindex": fold_result["val_cindex"],
                "val_auc": fold_result["val_auc"],
                "luad_cindex": external_result["cindex"],
                "luad_auc": external_result["auc"],
            }
        )

    detailed_df = pd.DataFrame(fold_rows)
    detailed_df.to_csv(result_dir / "lgg_5fold_results.csv", index=False)

    cindex_mean = float(np.mean(detailed_df["test_cindex"].to_numpy(dtype=float)))
    cindex_std = float(np.std(detailed_df["test_cindex"].to_numpy(dtype=float), ddof=1))
    auc_mean = float(np.mean(detailed_df["test_auc"].to_numpy(dtype=float)))
    auc_std = float(np.std(detailed_df["test_auc"].to_numpy(dtype=float), ddof=1))

    pd.DataFrame(
        [
            {"metric": "test_cindex", "mean": round(cindex_mean, 6), "std": round(cindex_std, 6)},
            {"metric": "test_auc", "mean": round(auc_mean, 6), "std": round(auc_std, 6)},
        ]
    ).to_csv(result_dir / "lgg_5fold_summary.csv", index=False)

    total_runtime_seconds = time.time() - total_start_time
    external_df = pd.DataFrame(external_rows)
    external_df.to_csv(result_dir / "luad_external_validation_5fold_results.csv", index=False)

    summary_df = pd.DataFrame(
        [
            {
                "model_name": model_name,
                "l1_ratio": L1_RATIO,
                "num_folds": len(external_df),
                "luad_cindex_mean": round(float(external_df["luad_cindex"].mean()), 6),
                "luad_cindex_std": round(float(external_df["luad_cindex"].std(ddof=1)), 6),
                "luad_auc_mean": round(float(external_df["luad_auc"].mean()), 6),
                "luad_auc_std": round(float(external_df["luad_auc"].std(ddof=1)), 6),
                "total_runtime_seconds": round(total_runtime_seconds, 2),
                "total_runtime_minutes": round(total_runtime_seconds / 60.0, 2),
            }
        ]
    )
    summary_df.to_csv(result_dir / "luad_external_validation_summary.csv", index=False)

    print("\n" + "=" * 80)
    print("LGG 5-FOLD SUMMARY")
    print("=" * 80)
    print(f"Test C-index: {cindex_mean:.4f} ± {cindex_std:.4f}")
    print(f"Test AUC: {auc_mean:.4f} ± {auc_std:.4f}")
    print("\n" + "=" * 80)
    print("LUAD 5-FOLD EXTERNAL VALIDATION")
    print("=" * 80)
    print(f"C-index: {external_df['luad_cindex'].mean():.4f} ± {external_df['luad_cindex'].std(ddof=1):.4f}")
    print(f"AUC: {external_df['luad_auc'].mean():.4f} ± {external_df['luad_auc'].std(ddof=1):.4f}")
    print(f"Total runtime: {total_runtime_seconds:.2f} seconds ({total_runtime_seconds / 60.0:.2f} minutes)")
    print(f"Results saved to: {result_dir}")


if __name__ == "__main__":
    main()
