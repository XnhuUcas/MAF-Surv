import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv

from utils import auc_formula16, km_plot_and_save, split_data_cv


# 默认读取 LUSC 的 pkl
PKL_PATH = Path("Datasets") / "LUSC" / "SIS200"/ "newdataset_200.pkl"

# 结果保存在 experiments/LUSC/encox and lassocox 下
RESULTS_ROOT = Path("experiment") / "LUSC" / "encox and lassocox"
EXP_NAME = "1"

# EN-Cox=0.5，Lasso-Cox=1.0
L1_RATIO = 1.0

# 与主方法一致：5-fold
N_FOLDS = 5

# Coxnet 配置
N_ALPHAS = 200
ALPHA_MIN_RATIO = 1e-7
TOL = 1e-8
MAX_ITER = 200_000


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def model_name_from_ratio(l1_ratio: float) -> str:
    return "encox" if abs(l1_ratio - 0.5) < 1e-12 else "lassocox"


def make_y(survival_time, event):
    return Surv.from_arrays(
        event=np.asarray(event).astype(bool),
        time=np.asarray(survival_time).astype(float),
    )


def cindex_formula15(survival_time, event, risk_scores) -> float:
    survival_time = np.asarray(survival_time, float).ravel()
    event = np.asarray(event, int).ravel()
    risk_scores = np.asarray(risk_scores, float).ravel()

    order = np.argsort(survival_time)
    good = 0
    total = 0
    for i in order:
        if event[i] != 1:
            continue
        time_i = survival_time[i]
        score_i = risk_scores[i]
        comparable_idx = np.where(survival_time > time_i)[0]
        if comparable_idx.size == 0:
            continue
        score_j = risk_scores[comparable_idx]
        good += np.sum(score_i > score_j)
        total += comparable_idx.size

    if total == 0:
        return float("nan")
    return float(good) / float(total)


def load_data_from_pkl(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    data = obj["datasets"]

    x_gene = np.asarray(data["x_gene"], dtype=np.float32)
    x_path = np.asarray(data["x_path"], dtype=np.float32)
    x_cna = np.asarray(data["x_cna"], dtype=np.float32)
    censored = np.asarray(data["censored"], dtype=np.float32)
    survival = np.asarray(data["survival"], dtype=np.float32)

    print(f"[INFO] Loaded pkl: {pkl_path}")
    print(f"[INFO] samples: {len(survival)}")
    print(f"[INFO] feature dims: gene={x_gene.shape[1]}, path={x_path.shape[1]}, cna={x_cna.shape[1]}")
    print(f"[INFO] event rate: {float(np.mean(censored)):.4f}")

    return {
        "x_gene": x_gene,
        "x_path": x_path,
        "x_cna": x_cna,
        "censored": censored,
        "survival": survival,
    }


def fuse_modalities(split_data):
    return np.hstack(
        [
            np.asarray(split_data["x_gene"], dtype=np.float32),
            np.asarray(split_data["x_path"], dtype=np.float32),
            np.asarray(split_data["x_cna"], dtype=np.float32),
        ]
    ).astype(np.float32)


def pick_best_alpha_by_validation(estimator, x_val, t_val, e_val):
    alphas = getattr(estimator, "alphas_", None)
    if alphas is None or len(alphas) == 0:
        scores = estimator.predict(x_val)
        return None, [cindex_formula15(t_val, e_val, scores)]

    cindex_values = []
    for alpha in alphas:
        scores = estimator.predict(x_val, alpha=alpha)
        cindex_values.append(cindex_formula15(t_val, e_val, scores))

    best_idx = int(np.nanargmax(cindex_values))
    return float(alphas[best_idx]), cindex_values


def fit_final_with_alpha_backoff(x_train_val_std, y_train_val, candidate_alphas, l1_ratio):
    last_error = None
    for alpha in candidate_alphas:
        try:
            estimator = CoxnetSurvivalAnalysis(
                l1_ratio=l1_ratio,
                alphas=[float(alpha)],
                normalize=False,
                tol=TOL,
                max_iter=MAX_ITER,
                fit_baseline_model=False,
            ).fit(x_train_val_std, y_train_val)
            return estimator, float(alpha)
        except ArithmeticError as error:
            last_error = error
            continue
    raise last_error


def fit_one_fold(train_split, val_split, test_split, l1_ratio):
    x_train = fuse_modalities(train_split)
    x_val = fuse_modalities(val_split)
    x_test = fuse_modalities(test_split)

    t_train = np.asarray(train_split["survival"], dtype=float)
    e_train = np.asarray(train_split["censored"], dtype=int)
    t_val = np.asarray(val_split["survival"], dtype=float)
    e_val = np.asarray(val_split["censored"], dtype=int)
    t_test = np.asarray(test_split["survival"], dtype=float)
    e_test = np.asarray(test_split["censored"], dtype=int)

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

    # 如果最优 alpha 过小导致最终拟合数值不稳定，则按更强正则依次回退
    path_alphas = np.asarray(path_estimator.alphas_, dtype=float)
    best_idx = int(np.nanargmax(cindex_values))
    candidate_alphas = list(path_alphas[best_idx::-1])

    x_train_val = np.vstack([x_train, x_val]).astype(np.float32)
    t_train_val = np.concatenate([t_train, t_val]).astype(float)
    e_train_val = np.concatenate([e_train, e_val]).astype(int)

    scaler_final = StandardScaler().fit(x_train_val)
    x_train_val_std = scaler_final.transform(x_train_val).astype(np.float32)
    x_test_std = scaler_final.transform(x_test).astype(np.float32)

    y_train_val = make_y(t_train_val, e_train_val)
    final_estimator, fitted_alpha = fit_final_with_alpha_backoff(
        x_train_val_std=x_train_val_std,
        y_train_val=y_train_val,
        candidate_alphas=candidate_alphas,
        l1_ratio=l1_ratio,
    )

    test_scores = final_estimator.predict(x_test_std)
    test_cindex = cindex_formula15(t_test, e_test, test_scores)
    test_auc = auc_formula16(t_test, e_test, test_scores)

    return {
        "risk_pred": np.asarray(test_scores, dtype=float),
        "os_time": np.asarray(t_test, dtype=float),
        "os_status": np.asarray(e_test, dtype=int),
        "cindex": float(test_cindex),
        "auc": float(test_auc),
        "selected_alpha": float(fitted_alpha),
        "initial_best_alpha": float(best_alpha),
    }


def main():
    data = load_data_from_pkl(PKL_PATH)
    cv_splits = split_data_cv(data, n_splits=N_FOLDS)

    model_name = model_name_from_ratio(L1_RATIO)
    result_dir = RESULTS_ROOT / EXP_NAME / model_name
    ensure_dir(result_dir)

    total_start_time = time.time()
    fold_rows = []
    merged_rows = []

    print("\n" + "=" * 80)
    print(f"RUNNING {model_name.upper()} ON LUSC")
    print("=" * 80)
    print(f"[INFO] l1_ratio = {L1_RATIO}")
    print(f"[INFO] result_dir = {result_dir}")

    for fold_idx, split_data_dict in cv_splits.items():
        fold_start_time = time.time()
        print("\n" + "-" * 80)
        print(f"Fold {fold_idx + 1}/{N_FOLDS}")
        print("-" * 80)

        fold_result = fit_one_fold(
            train_split=split_data_dict["train"],
            val_split=split_data_dict["val"],
            test_split=split_data_dict["test"],
            l1_ratio=L1_RATIO,
        )

        fold_runtime_seconds = time.time() - fold_start_time
        fold_dir = result_dir / f"{fold_idx}_fold"
        ensure_dir(fold_dir)

        pred_tuple = (
            fold_result["risk_pred"],
            fold_result["os_time"],
            fold_result["os_status"],
        )
        with open(fold_dir / f"{model_name}_{fold_idx}pred_test.pkl", "wb") as f:
            pickle.dump(pred_tuple, f)

        pd.DataFrame(
            {
                "os_time": fold_result["os_time"],
                "os_status": fold_result["os_status"],
                "risk_pred": fold_result["risk_pred"],
            }
        ).to_csv(result_dir / f"{fold_idx}-fold_pred.csv", index=False)

        pd.DataFrame(
            [
                {
                    "fold": fold_idx,
                    "cindex": fold_result["cindex"],
                    "auc": fold_result["auc"],
                    "selected_alpha": fold_result["selected_alpha"],
                    "initial_best_alpha": fold_result["initial_best_alpha"],
                    "fold_runtime_seconds": round(fold_runtime_seconds, 2),
                    "fold_runtime_minutes": round(fold_runtime_seconds / 60.0, 2),
                }
            ]
        ).to_csv(fold_dir / f"{model_name}_{fold_idx}_metrics.csv", index=False)

        print(
            f"[Fold {fold_idx + 1}] C-index={fold_result['cindex']:.4f} | "
            f"AUC={fold_result['auc']:.4f} | "
            f"alpha={fold_result['selected_alpha']:.3e} "
            f"(initial={fold_result['initial_best_alpha']:.3e}) | "
            f"runtime={fold_runtime_seconds:.2f}s"
        )

        fold_rows.append(
            {
                "fold": fold_idx,
                "cindex": fold_result["cindex"],
                "auc": fold_result["auc"],
                "selected_alpha": fold_result["selected_alpha"],
                "initial_best_alpha": fold_result["initial_best_alpha"],
                "fold_runtime_seconds": round(fold_runtime_seconds, 2),
                "fold_runtime_minutes": round(fold_runtime_seconds / 60.0, 2),
            }
        )
        merged_rows.append(
            pd.DataFrame(
                {
                    "os_time": fold_result["os_time"],
                    "os_status": fold_result["os_status"],
                    "risk_pred": fold_result["risk_pred"],
                }
            )
        )

    detailed_results = pd.DataFrame(fold_rows)
    total_runtime_seconds = time.time() - total_start_time
    detailed_results["total_runtime_seconds"] = round(total_runtime_seconds, 2)
    detailed_results["total_runtime_minutes"] = round(total_runtime_seconds / 60.0, 2)
    detailed_results.to_csv(result_dir / "detailed_results.csv", index=False)

    merged_df = pd.concat(merged_rows, ignore_index=True)
    merged_df.to_csv(result_dir / "out_pred_5fold.csv", index=False)

    cindex_values = detailed_results["cindex"].to_numpy(dtype=float)
    auc_values = detailed_results["auc"].to_numpy(dtype=float)
    fold_runtime_seconds_values = detailed_results["fold_runtime_seconds"].to_numpy(dtype=float)
    fold_runtime_minutes_values = detailed_results["fold_runtime_minutes"].to_numpy(dtype=float)
    alpha_values = detailed_results["selected_alpha"].to_numpy(dtype=float)
    initial_alpha_values = detailed_results["initial_best_alpha"].to_numpy(dtype=float)

    cindex_mean = float(np.mean(cindex_values))
    cindex_std = float(np.std(cindex_values, ddof=1))
    auc_mean = float(np.mean(auc_values))
    auc_std = float(np.std(auc_values, ddof=1))

    summary_metrics_df = pd.DataFrame(
        [
            {"metric": "cindex", "mean": round(cindex_mean, 6), "std": round(cindex_std, 6)},
            {"metric": "auc", "mean": round(auc_mean, 6), "std": round(auc_std, 6)},
            {
                "metric": "fold_runtime_seconds",
                "mean": round(float(np.mean(fold_runtime_seconds_values)), 6),
                "std": round(float(np.std(fold_runtime_seconds_values, ddof=1)), 6),
            },
            {
                "metric": "fold_runtime_minutes",
                "mean": round(float(np.mean(fold_runtime_minutes_values)), 6),
                "std": round(float(np.std(fold_runtime_minutes_values, ddof=1)), 6),
            },
            {
                "metric": "selected_alpha",
                "mean": round(float(np.mean(alpha_values)), 6),
                "std": round(float(np.std(alpha_values, ddof=1)), 6),
            },
            {
                "metric": "initial_best_alpha",
                "mean": round(float(np.mean(initial_alpha_values)), 6),
                "std": round(float(np.std(initial_alpha_values, ddof=1)), 6),
            },
        ]
    )
    summary_metrics_df.to_csv(result_dir / "summary_metrics.csv", index=False)

    final_results = {
        "model_name": model_name,
        "l1_ratio": L1_RATIO,
        "cindex_mean": round(cindex_mean, 4),
        "cindex_std": round(cindex_std, 4),
        "auc_mean": round(auc_mean, 4),
        "auc_std": round(auc_std, 4),
        "fold_runtime_seconds_mean": round(float(np.mean(fold_runtime_seconds_values)), 4),
        "fold_runtime_seconds_std": round(float(np.std(fold_runtime_seconds_values, ddof=1)), 4),
        "fold_runtime_minutes_mean": round(float(np.mean(fold_runtime_minutes_values)), 4),
        "fold_runtime_minutes_std": round(float(np.std(fold_runtime_minutes_values, ddof=1)), 4),
        "selected_alpha_mean": round(float(np.mean(alpha_values)), 6),
        "selected_alpha_std": round(float(np.std(alpha_values, ddof=1)), 6),
        "initial_best_alpha_mean": round(float(np.mean(initial_alpha_values)), 6),
        "initial_best_alpha_std": round(float(np.std(initial_alpha_values, ddof=1)), 6),
        "cindex_values": [round(x, 4) for x in cindex_values],
        "auc_values": [round(x, 4) for x in auc_values],
        "total_runtime_seconds": round(total_runtime_seconds, 2),
        "total_runtime_minutes": round(total_runtime_seconds / 60.0, 2),
    }
    with open(result_dir / "final_results.pkl", "wb") as f:
        pickle.dump(final_results, f)

    pd.DataFrame(
        [
            {
                "total_runtime_seconds": round(total_runtime_seconds, 2),
                "total_runtime_minutes": round(total_runtime_seconds / 60.0, 2),
            }
        ]
    ).to_csv(result_dir / "runtime_summary.csv", index=False)

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"C-index: {cindex_mean:.4f} ± {cindex_std:.4f}")
    print(f"AUC: {auc_mean:.4f} ± {auc_std:.4f}")
    print(
        "Fold runtime (seconds): "
        f"{float(np.mean(fold_runtime_seconds_values)):.2f} ± {float(np.std(fold_runtime_seconds_values, ddof=1)):.2f}"
    )
    print(
        "Fold runtime (minutes): "
        f"{float(np.mean(fold_runtime_minutes_values)):.2f} ± {float(np.std(fold_runtime_minutes_values, ddof=1)):.2f}"
    )
    print(
        "Selected alpha: "
        f"{float(np.mean(alpha_values)):.3e} ± {float(np.std(alpha_values, ddof=1)):.3e}"
    )

    km_title = "EN-Cox" if model_name == "encox" else "Lasso-Cox"
    km_png = result_dir / f"km_curve_{model_name}.png"
    p_value = km_plot_and_save(merged_df, km_png, risk_threshold="median", title_prefix=km_title)

    print(f"KM curve saved: {km_png} | Log-rank p={p_value:.2e}")
    print(f"Total runtime: {total_runtime_seconds:.2f} seconds ({total_runtime_seconds / 60.0:.2f} minutes)")
    print(f"Results saved to: {result_dir}")


if __name__ == "__main__":
    main()
