# deepsurv_pkl.py
import os, sys
os.environ.setdefault("THEANO_FLAGS", "device=cpu,floatX=float32,cxx=,openmp=False")

import numpy as np
import pandas as pd
import time
from pathlib import Path

# 确保优先导入本项目的 deepsurv 实现
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deep_surv import DeepSurv                      # 你的 Theano+Lasagne 实现
from data_from_pkl import load_fused_splits_from_pkl  # 你前面写好的加载器（event=1）

# ===== 路径 =====
PKL_PATH = r"D:\MAF-Surv Experiments\Datasets\LGG\original_data.pkl"
from utils import km_plot_and_save


def auc_formula16(T, E, scores):
    T = np.asarray(T, float).ravel()
    E = np.asarray(E, int).ravel()
    S = np.asarray(scores, float).ravel()

    event_times = np.sort(np.unique(T[E == 1]))
    good_total = 0
    pair_total = 0

    for t in event_times:
        pos_idx = (E == 1) & (T < t)
        neg_idx = (T > t)

        n_pos = int(np.sum(pos_idx))
        n_neg = int(np.sum(neg_idx))
        if n_pos == 0 or n_neg == 0:
            continue

        s_pos = S[pos_idx]
        s_neg = S[neg_idx]
        comp = np.greater.outer(s_pos, s_neg)
        good_total += int(np.sum(comp))
        pair_total += int(n_pos * n_neg)

    if pair_total == 0:
        return float("nan")
    return good_total / pair_total

def main():
    total_start_time = time.time()
    val_cidx, test_cidx = [], []
    val_auc_list, test_auc_list = [], []
    all_test_scores, all_test_times, all_test_events = [], [], []
    dataset_name = Path(PKL_PATH).parent.name.upper()
    result_dir = Path("experiment") / dataset_name / "Deepsurv"
    result_dir.mkdir(parents=True, exist_ok=True)

    for pack in load_fused_splits_from_pkl(
        PKL_PATH,
        n_splits=5,
        val_size=0.15,
        random_state=111,
        scaler="standard",     # 若加载器已统一标准化，则这里把 DeepSurv 的 standardize 设为 False
        print_info=True,
    ):
        X_tr, X_va, X_te = pack["X_tr"], pack["X_va"], pack["X_te"]
        T_tr, T_va, T_te = pack["T_tr"], pack["T_va"], pack["T_te"]
        E_tr, E_va, E_te = pack["E_tr"], pack["E_va"], pack["E_te"]
        fold = pack["fold"]

        # DeepSurv 接口期望字典键：x(特征 float32)、t(时间 float32)、e(事件 int)
        to_ds = lambda X, T, E: {
            "x": X.astype("float32"),
            "t": T.astype("float32"),
            "e": E.astype("int32"),
        }
        train_data = to_ds(X_tr, T_tr, E_tr)
        valid_data = to_ds(X_va, T_va, E_va)
        test_data  = to_ds(X_te, T_te, E_te)

        if fold == 1:
            print(f"[INFO] DeepSurv 输入维度 n_in={X_tr.shape[1]} | "
                  f"train/val/test = {X_tr.shape} / {X_va.shape} / {X_te.shape}")

        
        hyperparams = dict(
            n_in=X_tr.shape[1],
            learning_rate=1e-3,  # 这是你需要自己定的；常用 1e-3
        )

        model = DeepSurv(**hyperparams)
        #_ = model.train(train_data, valid_data, n_epochs=300, validation_frequency=1)
        _ = model.train(train_data, valid_data)

        val_c = float(model.get_concordance_index(**valid_data))
        te_c  = float(model.get_concordance_index(**test_data))
        val_scores = model.predict_risk(valid_data["x"])
        test_scores = model.predict_risk(test_data["x"])
        val_auc = float(auc_formula16(valid_data["t"], valid_data["e"], val_scores))
        test_auc = float(auc_formula16(test_data["t"], test_data["e"], test_scores))
        print(
            f"[Fold {fold}] val C-index: {val_c:.4f} | test C-index: {te_c:.4f} | "
            f"val AUC: {val_auc:.4f} | test AUC: {test_auc:.4f}"
        )

        val_cidx.append(val_c)
        test_cidx.append(te_c)
        val_auc_list.append(val_auc)
        test_auc_list.append(test_auc)
        all_test_scores.extend(np.asarray(test_scores).reshape(-1).tolist())
        all_test_times.extend(np.asarray(test_data["t"]).reshape(-1).tolist())
        all_test_events.extend(np.asarray(test_data["e"]).reshape(-1).tolist())

    print(f"5-fold mean val C-index:  {np.mean(val_cidx):.4f} ± {np.std(val_cidx):.4f}")
    print(f"5-fold mean test C-index: {np.mean(test_cidx):.4f} ± {np.std(test_cidx):.4f}")

    print(f"5-fold mean val AUC:      {np.mean(val_auc_list):.4f} ± {np.std(val_auc_list):.4f}")
    print(f"5-fold mean test AUC:     {np.mean(test_auc_list):.4f} ± {np.std(test_auc_list):.4f}")

    km_df = pd.DataFrame({
        "os_time": all_test_times,
        "os_status": all_test_events,
        "risk_pred": all_test_scores,
    })
    km_df.to_csv(result_dir / "deepsurv_5fold_predictions.csv", index=False)
    km_pvalue = km_plot_and_save(
        km_df,
        result_dir / f"km_curve_deepsurv_{dataset_name.lower()}.png",
        risk_threshold="median",
        title_prefix="DeepSurv",
    )
    print(f"5-fold KM log-rank p-value: {km_pvalue:.2e}")

    total_elapsed_seconds = time.time() - total_start_time
    print(f"Total runtime: {total_elapsed_seconds:.2f} seconds ({total_elapsed_seconds / 60.0:.2f} minutes)")

if __name__ == "__main__":
    main()
