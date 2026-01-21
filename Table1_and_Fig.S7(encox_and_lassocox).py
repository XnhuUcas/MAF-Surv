# # enc_cx_nestedcv.py —— EN/Lasso-Cox：嵌套CV选α，外层5折评估 + 合并KM + AUC(公式16)
#
# import os
# from pathlib import Path
# import pickle
# import numpy as np
# import pandas as pd
# import warnings
#
# from sklearn.model_selection import StratifiedKFold
# from sklearn.preprocessing import StandardScaler
#
# from sksurv.linear_model import CoxnetSurvivalAnalysis
# from sksurv.util import Surv
#
# import matplotlib.pyplot as plt
# from lifelines import KaplanMeierFitter
# from lifelines.statistics import logrank_test
#
#
# # ========== 路径与配置 ==========
# PKL_PATH    = r"D:\CAMR_paper\multimodal_survival_prediction\multimodal_survival_prediction\new datasets\sis_200\newdataset_200.pkl"
# RESULTS_DIR = r"D:\CAMR_paper\multimodal_survival_prediction\multimodal_survival_prediction\CAMR_results"
# EXP_NAME    = "sis200"
# MODEL_NAME  = "en-cox"          # 改名也可以
# EVENT_KEY   = "event"           # 你的 pkl 里事件字段名（1=事件，0=删失）
# L1_RATIO    = 0.5               # EN-Cox=0.5 ；Lasso-Cox=1.0
# N_OUTER_FOLDS = 5               # 外层折数
# N_INNER_FOLDS = 5               # 内层折数（用于选α）
# CV_RANDOM_STATE = 111           # 外/内层统一随机种子
#
#
# # ========== 小工具 ==========
# def ensure_dir(p: Path):
#     p.mkdir(parents=True, exist_ok=True)
#
# def make_y(T, E):
#     """把 (time, event) 转为 sksurv 的结构；E: 1=事件, 0=删失。"""
#     return Surv.from_arrays(event=E.astype(bool), time=T.astype(float))
#
# # === NEW: 严格按公式(15) 实现 C-index（不做任何方向翻转） ===
# def cindex_formula15(T, E, scores) -> float:
#     """
#     公式(15)：C-index = (1/N) * sum_{i} sum_{j>i} I( h(x_i) > h(x_j) )
#     在生存里可比对定义为：E_i=1 且 T_i < T_j。
#     当 s_i == s_j 时 I=0（不计半分）。
#     """
#     T = np.asarray(T, float).ravel()
#     E = np.asarray(E, int).ravel()
#     S = np.asarray(scores, float).ravel()
#
#     order = np.argsort(T)  # 时间从小到大
#     good = 0
#     total = 0
#     for idx_i in order:
#         if E[idx_i] != 1:
#             continue
#         ti, si = T[idx_i], S[idx_i]
#         js = np.where(T > ti)[0]  # 更晚时间的样本
#         if js.size == 0:
#             continue
#         sj = S[js]
#         good += np.sum(si > sj)      # ties 计 0
#         total += js.size
#     if total == 0:
#         return float("nan")
#     return float(good) / float(total)
#
# # === CHANGED: 严格按公式(16) 实现“动态 AUC”
# def auc_formula16(T, E, scores) -> float:
#     """
#     公式(16)：AUC = (1/num) * sum_{t∈T_event} sum_{y_i<t, δ_i=1} sum_{y_j>t} I(h(x_i) > h(x_j))
#     这里 T_event 为测试集中所有发生事件样本的观测时间集合（唯一值）。
#     与固定 t 的 ROC-AUC 不同，这是对所有 t 的区分能力按“可比较对数量”加权平均。
#     """
#     T = np.asarray(T, float).ravel()
#     E = np.asarray(E, int).ravel()
#     S = np.asarray(scores, float).ravel()
#
#     # 所有事件时刻（唯一、升序）
#     event_times = np.sort(np.unique(T[E == 1]))
#     good_total = 0
#     pair_total = 0
#
#     for t in event_times:
#         pos_idx = (E == 1) & (T < t)   # 在 t 前已发生事件
#         neg_idx = (T > t)              # 在 t 时仍在风险集中
#
#         n_pos = int(np.sum(pos_idx))
#         n_neg = int(np.sum(neg_idx))
#         if n_pos == 0 or n_neg == 0:
#             continue
#
#         s_pos = S[pos_idx]
#         s_neg = S[neg_idx]
#         # 两两比较，ties 计 0
#         # 使用外积比较以加速：shape (n_pos, n_neg)
#         comp = np.greater.outer(s_pos, s_neg)
#         good_total += int(np.sum(comp))
#         pair_total += int(n_pos * n_neg)
#
#     if pair_total == 0:
#         return float("nan")
#     return good_total / pair_total
#
# def save_fold_pred(results_root: Path, fold: int, risk_pred, survtime, event):
#     """
#     保存每折测试集的 (risk_pred, time, event)，便于KM脚本复用
#     <root>/<fold>_fold/<MODEL_NAME>_<fold>pred_test.pkl
#     """
#     fold_dir = results_root / f"{fold}_fold"
#     ensure_dir(fold_dir)
#     out_pkl = fold_dir / f"{MODEL_NAME}_{fold}pred_test.pkl"
#     with open(out_pkl, "wb") as f:
#         pickle.dump((np.asarray(risk_pred), np.asarray(survtime), np.asarray(event)), f)
#     return out_pkl
#
# def km_plot_and_save(df: pd.DataFrame, out_png: Path, title_prefix=""):
#     """
#     合并5折后的 KM 图：以风险分数中位数划分高/低风险
#     """
#     thr = float(np.median(df["risk_pred"]))
#     hi = df["risk_pred"] > thr
#     lo = ~hi
#
#     kmf = KaplanMeierFitter()
#     plt.figure(figsize=(8, 6))
#     kmf.fit(df.loc[hi, "os_time"], df.loc[hi, "os_status"], label=f"High risk (n={int(hi.sum())})")
#     kmf.plot(ci_show=True, linewidth=2)
#     kmf.fit(df.loc[lo, "os_time"], df.loc[lo, "os_status"], label=f"Low risk (n={int(lo.sum())})")
#     kmf.plot(ci_show=True, linewidth=2)
#
#     res = logrank_test(
#         df.loc[hi, "os_time"], df.loc[lo, "os_time"],
#         df.loc[hi, "os_status"], df.loc[lo, "os_status"]
#     )
#
#     plt.title(f"{title_prefix} KM (Log-rank p={res.p_value:.2e})")
#     plt.xlabel("Time")
#     plt.ylabel("Survival probability")
#     plt.grid(alpha=0.3)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(out_png, dpi=300)
#     plt.close()
#     return float(res.p_value)
#
# def load_from_pkl(pkl_path: str):
#     """
#     读取 .pkl，早期融合为特征矩阵 X，并返回 (X, T, E)。
#     """
#     with open(pkl_path, "rb") as f:
#         obj = pickle.load(f)
#     data = obj.get("datasets", obj)
#
#     X_gene = np.asarray(data["x_gene"], dtype=np.float32)
#     X_cna  = np.asarray(data["x_cna"],  dtype=np.float32)
#     X_path = np.asarray(data["x_path"], dtype=np.float32)
#     T = np.asarray(data["survival"], dtype=float)
#
#     if EVENT_KEY in data:
#         E = np.asarray(data[EVENT_KEY], dtype=int)
#     elif "event" in data:
#         E = np.asarray(data["event"], dtype=int)
#     elif "censored" in data:
#         E = np.asarray(data["censored"], dtype=int)
#     else:
#         raise KeyError("pkl 未找到事件标签（event/censored）。")
#
#     X = np.hstack([X_gene, X_cna, X_path]).astype(np.float32)
#     print(f"[INFO] Loaded: n={X.shape[0]}, p={X.shape[1]}, event_rate={E.mean():.3f}")
#     return X, T, E
#
# def pick_best_alpha_by_validation(est: CoxnetSurvivalAnalysis, X_va, T_va, E_va):
#     """
#     给定已在 inner-train 上拟合好的 Coxnet（带有一条 alphas_ 路径），
#     在验证集上沿路径计算 C-index(公式15)，返回 “最佳 alpha”。
#     """
#     alphas = getattr(est, "alphas_", None)
#     if alphas is None or len(alphas) == 0:
#         s = est.predict(X_va)
#         c = cindex_formula15(T_va, E_va, s)   # === CHANGED: 用公式(15)
#         return None, [c]
#     c_list = []
#     for a in alphas:
#         s = est.predict(X_va, alpha=a)
#         c = cindex_formula15(T_va, E_va, s)   # === CHANGED: 用公式(15)
#         c_list.append(c)
#     best_idx = int(np.nanargmax(c_list))
#     return float(alphas[best_idx]), c_list
#
#
# # ========== 主流程（嵌套CV） ==========
# def main():
#     # 数据与输出目录
#     X, T, E = load_from_pkl(PKL_PATH)
#     results_root = Path(RESULTS_DIR) / EXP_NAME / MODEL_NAME
#     ensure_dir(results_root)
#
#     # 外层 5 折
#     outer = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
#
#     test_cidx_list = []
#     auc16_list = []
#     merged_rows = []
#
#     print(f"\n[INFO] Nested CV starting: outer={N_OUTER_FOLDS}, inner={N_INNER_FOLDS}, l1_ratio={L1_RATIO}\n")
#
#     for fold, (tr_idx_all, te_idx) in enumerate(outer.split(X, E), start=1):
#         print(f"\n===== Outer Fold {fold} =====")
#         X_tr_all, X_te = X[tr_idx_all], X[te_idx]
#         T_tr_all, T_te = T[tr_idx_all], T[te_idx]
#         E_tr_all, E_te = E[tr_idx_all], E[te_idx]
#
#         # ---------- 内层：在外层训练集上选“本折最佳 alpha” ----------
#         inner = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
#         best_alphas = []
#
#         for k, (tr_idx, va_idx) in enumerate(inner.split(X_tr_all, E_tr_all), start=1):
#             X_tr, X_va = X_tr_all[tr_idx], X_tr_all[va_idx]
#             T_tr, T_va = T_tr_all[tr_idx], T_tr_all[va_idx]
#             E_tr, E_va = E_tr_all[tr_idx], E_tr_all[va_idx]
#
#             # 标准化：只用 inner-train 拟合
#             scaler = StandardScaler().fit(X_tr)
#             X_tr = scaler.transform(X_tr).astype(np.float32)
#             X_va = scaler.transform(X_va).astype(np.float32)
#
#             y_tr = make_y(T_tr, E_tr)
#             est = CoxnetSurvivalAnalysis(
#                 l1_ratio=L1_RATIO,
#                 n_alphas=200,
#                 alpha_min_ratio=1e-7,   # 如遇数值问题，可调大到 1e-6
#                 normalize=False,
#                 tol=1e-8,
#                 max_iter=200_000,
#                 fit_baseline_model=False
#             ).fit(X_tr, y_tr)
#
#             a_best, _ = pick_best_alpha_by_validation(est, X_va, T_va, E_va)
#             best_alphas.append(a_best)
#
#         # 该外层折采用“内层 best_alpha 的中位数”作为稳健选择
#         alpha_sel = float(np.median(best_alphas))
#         print(f"[Inner] best_alphas (5 folds) = {np.array(best_alphas)}  ->  selected α(median) = {alpha_sel:.3e}")
#
#         # ---------- 用外层训练集(全部)在 α_sel 上重训，测试 ----------
#         scaler = StandardScaler().fit(X_tr_all)
#         X_tr_all_std = scaler.transform(X_tr_all).astype(np.float32)
#         X_te_std     = scaler.transform(X_te).astype(np.float32)
#
#         y_tr_all = make_y(T_tr_all, E_tr_all)
#         est_final = CoxnetSurvivalAnalysis(
#             l1_ratio=L1_RATIO,
#             alphas=[alpha_sel],
#             normalize=False,
#             tol=1e-8,
#             max_iter=200_000,
#             fit_baseline_model=False
#         ).fit(X_tr_all_std, y_tr_all)
#
#         s_te = est_final.predict(X_te_std)
#
#         # === CHANGED: 评估严格按公式(15)/(16)
#         te_cidx = cindex_formula15(T_te, E_te, s_te)   # 公式(15)
#         auc16   = auc_formula16(T_te, E_te, s_te)      # 公式(16)
#
#         test_cidx_list.append(te_cidx)
#         auc16_list.append(auc16)
#
#         # 保存每折预测（KM合并用）
#         out_pkl = save_fold_pred(results_root, fold, s_te, T_te, E_te)
#         print(f"[Outer {fold}]  Test C-index={te_cidx:.4f} | AUC(16)={auc16:.4f} | saved: {out_pkl}")
#
#         merged_rows.append(pd.DataFrame({
#             "risk_pred": s_te.astype(float),
#             "os_time":   T_te.astype(float),
#             "os_status": E_te.astype(int),
#             "fold":      fold
#         }))
#
#     # ---------- 汇总 ----------
#     cidx_mean, cidx_std = np.nanmean(test_cidx_list), np.nanstd(test_cidx_list)
#     auc_mean,  auc_std  = np.nanmean(auc16_list), np.nanstd(auc16_list)
#
#     print("\n===== Summary over outer 5 folds =====")
#     print(f"C-index (test, formula 15): {cidx_mean:.4f} ± {cidx_std:.4f}")
#     print(f"AUC (test, formula 16):     {auc_mean:.4f} ± {auc_std:.4f}")
#
#     # 保存 AUC(16) 的每折值，方便与其它方法汇总画箱线图
#     auc_table = pd.DataFrame({"fold": np.arange(1, N_OUTER_FOLDS+1), "AUC_formula16": auc16_list})
#     out_auc_csv = Path(RESULTS_DIR) / EXP_NAME / MODEL_NAME / f"AUC_formula16_{MODEL_NAME}.csv"
#     ensure_dir(out_auc_csv.parent)
#     auc_table.to_csv(out_auc_csv, index=False)
#     print(f"AUC (formula 16) per fold saved to: {out_auc_csv}")
#
#     # 合并并保存预测（与既有KM脚本兼容）
#     merged_df = pd.concat(merged_rows, ignore_index=True)
#     out_pred_csv = Path(RESULTS_DIR) / EXP_NAME / MODEL_NAME / "out_pred_5fold.csv"
#     merged_df.to_csv(out_pred_csv, index=False)
#     print(f"Merged test predictions saved to: {out_pred_csv}")
#
#     # 画“5折合并”的 KM
#     out_km_png = Path(RESULTS_DIR) / EXP_NAME / MODEL_NAME / "KM_5fold_merged.png"
#     p = km_plot_and_save(merged_df, out_km_png, title_prefix=f"En-Cox")
#     print(f"KM curve saved: {out_km_png} | Log-rank p={p:.2e}")
#
#
# if __name__ == "__main__":
#     warnings.filterwarnings("ignore", category=UserWarning)
#     main()


# en_cox_repeat_runs_meanKM.py
# EN/Lasso-Cox：单次划分选参并评估；换不同随机种子重复多次；
# 逐 run 画 KM，插值到公共时间网格后做均值±标准差（阴影带）

# import os
# from pathlib import Path
# import pickle
# import numpy as np
# import pandas as pd
#
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
#
# from sksurv.linear_model import CoxnetSurvivalAnalysis
# from sksurv.util import Surv
#
# import matplotlib.pyplot as plt
# from lifelines import KaplanMeierFitter
# from lifelines.statistics import logrank_test
#
# # ========== 路径与配置 ==========
# PKL_PATH     = r"D:\CAMR_paper\multimodal_survival_prediction\multimodal_survival_prediction\new datasets\sis_200\newdataset_200.pkl"
# RESULTS_DIR  = r"D:\CAMR_paper\multimodal_survival_prediction\multimodal_survival_prediction\CAMR_results"
# EXP_NAME     = "sis200"
# MODEL_NAME   = "en-cox"
# EVENT_KEY    = "event"      # 你的 pkl 里事件字段名（1=事件，0=删失）
# L1_RATIO     = 0.5          # EN-Cox=0.5；Lasso-Cox=1.0
# TEST_SIZE    = 0.2
# VAL_SIZE     = 0.15
# SEEDS        = [111, 222, 333, 444, 666]   # 5 个随机种子
# N_ALPHAS     = 200
# ALPHA_MIN_RATIO = 1e-7
#
# # ========== 小工具 ==========
# def ensure_dir(p: Path):
#     p.mkdir(parents=True, exist_ok=True)
#
# def make_y(T, E):
#     """把 (time, event) 转为 sksurv 的结构；E:1=事件,0=删失。"""
#     return Surv.from_arrays(event=E.astype(bool), time=T.astype(float))
#
# # —— 公式(15)：严格 c-index（不做方向保护、ties 计 0）
# def cindex_formula15(T, E, scores) -> float:
#     T = np.asarray(T, float).ravel()
#     E = np.asarray(E, int).ravel()
#     S = np.asarray(scores, float).ravel()
#     order = np.argsort(T)
#     good = 0
#     total = 0
#     for i in order:
#         if E[i] != 1:
#             continue
#         ti, si = T[i], S[i]
#         js = np.where(T > ti)[0]
#         if js.size == 0:
#             continue
#         sj = S[js]
#         good += np.sum(si > sj)   # ties 计 0
#         total += js.size
#     if total == 0:
#         return float("nan")
#     return float(good) / float(total)
#
# # —— 公式(16)：严格 AUC（与15的可比对与判别一致，单独保留函数名以对应原文）
# def auc_formula16(T, E, scores) -> float:
#     T = np.asarray(T, float).ravel()
#     E = np.asarray(E, int).ravel()
#     S = np.asarray(scores, float).ravel()
#     order = np.argsort(T)
#     good = 0
#     total = 0
#     for i in order:
#         if E[i] != 1:
#             continue
#         ti, si = T[i], S[i]
#         js = np.where(T > ti)[0]
#         if js.size == 0:
#             continue
#         sj = S[js]
#         good += np.sum(si > sj)   # ties 计 0
#         total += js.size
#     if total == 0:
#         return float("nan")
#     return float(good) / float(total)
#
# def load_from_pkl(pkl_path: str):
#     """读取 .pkl，早期融合为特征矩阵 X，并返回 (X, T, E)。"""
#     with open(pkl_path, "rb") as f:
#         obj = pickle.load(f)
#     data = obj.get("datasets", obj)
#
#     X_gene = np.asarray(data["x_gene"], dtype=np.float32)
#     X_cna  = np.asarray(data["x_cna"],  dtype=np.float32)
#     X_path = np.asarray(data["x_path"], dtype=np.float32)
#     T = np.asarray(data["survival"], dtype=float)
#
#     if EVENT_KEY in data:
#         E = np.asarray(data[EVENT_KEY], dtype=int)
#     elif "event" in data:
#         E = np.asarray(data["event"], dtype=int)
#     elif "censored" in data:
#         E = np.asarray(data["censored"], dtype=int)
#     else:
#         raise KeyError("pkl 未找到事件标签（event/censored）。")
#
#     X = np.hstack([X_gene, X_cna, X_path]).astype(np.float32)
#     print(f"[INFO] Loaded: n={X.shape[0]}, p={X.shape[1]}, event_rate={E.mean():.3f}")
#     return X, T, E
#
# def pick_best_alpha_by_validation(est: CoxnetSurvivalAnalysis, X_va, T_va, E_va):
#     """给定已在 train 上 fit 出整条路径的 est，在 val 上沿路径选公式(15)的最佳 alpha。"""
#     alphas = getattr(est, "alphas_", None)
#     if alphas is None or len(alphas) == 0:
#         s = est.predict(X_va)
#         return None, [cindex_formula15(T_va, E_va, s)]
#     c_list = []
#     for a in alphas:
#         s = est.predict(X_va, alpha=a)
#         c = cindex_formula15(T_va, E_va, s)
#         c_list.append(c)
#     best_idx = int(np.nanargmax(c_list))
#     return float(alphas[best_idx]), c_list
#
# # —— 每个 run 的 KM 曲线（高/低风险）插值到公共时间网格
# def compute_km_curves_on_grid(T, E, scores, t_grid):
#     T = np.asarray(T, float).ravel()
#     E = np.asarray(E, int).ravel()
#     s = np.asarray(scores, float).ravel()
#     thr = float(np.median(s))
#     hi = s > thr
#     lo = ~hi
#
#     kmf = KaplanMeierFitter()
#
#     # High risk
#     kmf.fit(T[hi], E[hi])
#     t_hi = kmf.survival_function_.index.values
#     S_hi = kmf.survival_function_["KM_estimate"].values
#     # Low risk
#     kmf.fit(T[lo], E[lo])
#     t_lo = kmf.survival_function_.index.values
#     S_lo = kmf.survival_function_["KM_estimate"].values
#
#     # 插值到公共 t_grid（左侧=1.0，右侧=末值）
#     S_hi_grid = np.interp(t_grid, t_hi, S_hi, left=1.0, right=S_hi[-1])
#     S_lo_grid = np.interp(t_grid, t_lo, S_lo, left=1.0, right=S_lo[-1])
#
#     # 本 run 的 log-rank（用于报告均值/方差）
#     res = logrank_test(T[hi], T[lo], E[hi], E[lo])
#     p_value = float(res.p_value)
#
#     return S_hi_grid, S_lo_grid, p_value
#
# # ========== 主流程 ==========
# def main():
#     X, T, E = load_from_pkl(PKL_PATH)
#
#     results_root = Path(RESULTS_DIR) / EXP_NAME / MODEL_NAME
#     ensure_dir(results_root)
#
#     # 收集 5 次 run 的指标与 KM 曲线（插值后）
#     cidx_runs = []
#     auc_runs  = []
#     p_runs    = []
#
#     # 为“均值KM”准备的公共时间网格：覆盖所有 run 的测试集时间范围
#     all_test_T_max = []
#
#     # 先做一遍 seed 预扫描来确定全局时间上界（也可以在每次 run 后追加，再最后取 max）
#     for seed in SEEDS:
#         _, X_te, _, T_te, _, _ = train_test_split(X, T, E, test_size=TEST_SIZE,
#                                                  random_state=seed, stratify=E)
#         all_test_T_max.append(np.max(T_te))
#     t_grid = np.linspace(0.0, float(np.max(all_test_T_max)), 200)  # 200 个点的公共网格
#
#     # 存放每个 run 在 t_grid 上的 S(t)
#     all_S_hi = []
#     all_S_lo = []
#
#     # 也保存每个 run 的测试集预测（可选，兼容你之前的 KM 工具）
#     merged_rows = []
#
#     print(f"\n[INFO] Repeating single-split pipeline for {len(SEEDS)} seeds: {SEEDS}\n")
#
#     for run_id, seed in enumerate(SEEDS, start=1):
#         print(f"===== Run {run_id} | seed={seed} =====")
#
#         # 1) 单次划分 train/val/test
#         X_trva, X_te, T_trva, T_te, E_trva, E_te = train_test_split(
#             X, T, E, test_size=TEST_SIZE, random_state=seed, stratify=E
#         )
#         # 2) 从 train+val 再划出验证集
#         idx = np.arange(len(T_trva))
#         tr_idx, va_idx = train_test_split(
#             idx, test_size=VAL_SIZE, random_state=seed, stratify=E_trva
#         )
#         X_tr, X_va = X_trva[tr_idx], X_trva[va_idx]
#         T_tr, T_va = T_trva[tr_idx], T_trva[va_idx]
#         E_tr, E_va = E_trva[tr_idx], E_trva[va_idx]
#
#         # 3) 标准化（仅用训练集拟合）
#         scaler = StandardScaler().fit(X_tr)
#         X_tr = scaler.transform(X_tr).astype(np.float32)
#         X_va = scaler.transform(X_va).astype(np.float32)
#         X_te_std = scaler.transform(X_te).astype(np.float32)
#         X_trva_std = scaler.transform(X_trva).astype(np.float32)
#
#         # 4) 训练整条 λ 路径
#         y_tr = make_y(T_tr, E_tr)
#         est = CoxnetSurvivalAnalysis(
#             l1_ratio=L1_RATIO,
#             n_alphas=N_ALPHAS,
#             alpha_min_ratio=ALPHA_MIN_RATIO,
#             normalize=False,
#             tol=1e-8,
#             max_iter=200_000,
#             fit_baseline_model=False
#         ).fit(X_tr, y_tr)
#         print(f"  Path: |alphas|={len(est.alphas_)}  range=[{est.alphas_[0]:.3e}, {est.alphas_[-1]:.3e}]")
#
#         # 5) 在 val 上按 公式(15) 选最佳 alpha
#         best_alpha, _ = pick_best_alpha_by_validation(est, X_va, T_va, E_va)
#         print(f"  Best alpha(from val) = {best_alpha:.3e}")
#
#         # 6) 用 train+val 合并，锁定 α 重训练
#         y_trva = make_y(T_trva, E_trva)
#         est_final = CoxnetSurvivalAnalysis(
#             l1_ratio=L1_RATIO,
#             alphas=[best_alpha],
#             normalize=False,
#             tol=1e-8,
#             max_iter=200_000,
#             fit_baseline_model=False
#         ).fit(X_trva_std, y_trva)
#
#         # 7) 测试集风险分数
#         s_te = est_final.predict(X_te_std)
#
#         # 8) 评估（严格公式）
#         cidx = cindex_formula15(T_te, E_te, s_te)
#         aucv = auc_formula16(T_te, E_te, s_te)
#         print(f"  Test C-index(15)={cidx:.4f} | AUC(16)={aucv:.4f}")
#
#         cidx_runs.append(cidx)
#         auc_runs.append(aucv)
#
#         # 9) 本 run 的 KM（高/低风险）插值到公共网格
#         S_hi_grid, S_lo_grid, p_val = compute_km_curves_on_grid(T_te, E_te, s_te, t_grid)
#         all_S_hi.append(S_hi_grid)
#         all_S_lo.append(S_lo_grid)
#         p_runs.append(p_val)
#
#         # 10) 可选：保存每 run 的预测（兼容既有 KM 工具）
#         merged_rows.append(pd.DataFrame({
#             "risk_pred": s_te.astype(float),
#             "os_time":   T_te.astype(float),
#             "os_status": E_te.astype(int),
#             "run":       run_id
#         }))
#
#     # ========== 汇总并出图 ==========
#     cidx_mean, cidx_std = float(np.nanmean(cidx_runs)), float(np.nanstd(cidx_runs))
#     auc_mean,  auc_std  = float(np.nanmean(auc_runs)),  float(np.nanstd(auc_runs))
#     p_mean,    p_std    = float(np.nanmean(p_runs)),    float(np.nanstd(p_runs))
#
#     print("\n===== Summary over 5 runs =====")
#     print(f"C-index (formula 15): mean={cidx_mean:.4f} ± {cidx_std:.4f}")
#     print(f"AUC      (formula 16): mean={auc_mean:.4f} ± {auc_std:.4f}")
#     print(f"Log-rank p (KM split by median): mean={p_mean:.3e} ± {p_std:.3e}")
#
#     # 保存 per-run AUC(16)（后续与其它方法汇总画箱线图）
#     out_dir = Path(RESULTS_DIR) / EXP_NAME / MODEL_NAME
#     ensure_dir(out_dir)
#     pd.DataFrame({
#         "run": np.arange(1, len(SEEDS)+1),
#         "C_index_formula15": cidx_runs,
#         "AUC_formula16": auc_runs,
#         "logrank_p": p_runs
#     }).to_csv(out_dir / "per_run_metrics.csv", index=False)
#
#     # 保存合并预测（如果你还需要用到既有的 pooled KM 脚本）
#     merged_df = pd.concat(merged_rows, ignore_index=True)
#     merged_df.to_csv(out_dir / "out_pred_5runs.csv", index=False)
#
#     # —— 画“平均 KM（均值±标准差阴影带）”
#     all_S_hi = np.vstack(all_S_hi)   # shape: (n_runs, len(t_grid))
#     all_S_lo = np.vstack(all_S_lo)
#
#     S_hi_mean, S_hi_std = all_S_hi.mean(axis=0), all_S_hi.std(axis=0)
#     S_lo_mean, S_lo_std = all_S_lo.mean(axis=0), all_S_lo.std(axis=0)
#
#     plt.figure(figsize=(8, 6))
#     plt.plot(t_grid, S_hi_mean, label="High risk (mean)", lw=2)
#     plt.fill_between(t_grid,
#                      np.clip(S_hi_mean - S_hi_std, 0, 1),
#                      np.clip(S_hi_mean + S_hi_std, 0, 1),
#                      alpha=0.2)
#     plt.plot(t_grid, S_lo_mean, label="Low risk (mean)", lw=2)
#     plt.fill_between(t_grid,
#                      np.clip(S_lo_mean - S_lo_std, 0, 1),
#                      np.clip(S_lo_mean + S_lo_std, 0, 1),
#                      alpha=0.2)
#     plt.xlabel("Time")
#     plt.ylabel("Survival probability")
#     plt.title(f"EN-Cox mean KM over {len(SEEDS)} runs\n"
#               f"C-index(15): {cidx_mean:.3f}±{cidx_std:.3f} | "
#               f"AUC(16): {auc_mean:.3f}±{auc_std:.3f}")
#     plt.legend()
#     plt.grid(alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(out_dir / "KM_mean_over_runs.png", dpi=300)
#     plt.close()
#
#     print(f"\nArtifacts saved to: {out_dir}")
#     print(" - per_run_metrics.csv  （每次 run 的 C-index、AUC、logrank p）")
#     print(" - out_pred_5runs.csv   （每次 run 的测试集预测汇总，若你还想用 pooled KM）")
#     print(" - KM_mean_over_runs.png（平均KM曲线，附±1 SD 阴影带）")
#
#
# if __name__ == "__main__":
#     main()
# enc_cox_single_split_multiseed.py
# EN/Lasso-Cox：一次划分 + 单验证集选参 + 在同一次 test 上评估
# 用不同随机种子重复多次；输出 C-index(公式15) 与 AUC(公式16) 的均值±标准差；
# 并绘制 5 次测试样本“合并”的 KM 曲线。

import os
from pathlib import Path
import pickle
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv

import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


# ===== 路径与配置 =====
PKL_PATH    = r"D:\CAMR_paper\multimodal_survival_prediction\multimodal_survival_prediction\Datasets\original_data.pkl"
RESULTS_DIR = r"D:\CAMR_paper\论文图片"
EXP_NAME    = "LGG"         # 改成你的数据集名
MODEL_NAME  = "lasso_cox"
EVENT_KEY   = "event"        # 你的 pkl 里事件字段名（1=事件，0=删失）
L1_RATIO    = 1.0         # EN-Cox=0.5；Lasso-Cox=1.0

# 单次划分的比例
TEST_SIZE = 0.2
VAL_SIZE  = 0.15

# 要重复的随机种子
SEEDS = [111, 222, 333, 444, 666]


# ===== 小工具 =====
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def make_y(T, E):
    """把 (time, event) 转为 sksurv 的结构；E: 1=事件, 0=删失。"""
    return Surv.from_arrays(event=E.astype(bool), time=T.astype(float))

# —— 公式(15)：严格可比对 (E_i=1 且 T_i<T_j)，ties 记 0 分
def cindex_formula15(T, E, scores) -> float:
    T = np.asarray(T, float).ravel()
    E = np.asarray(E, int).ravel()
    S = np.asarray(scores, float).ravel()
    order = np.argsort(T)  # 按时间从小到大
    good = 0
    total = 0
    for i in order:
        if E[i] != 1:
            continue
        ti, si = T[i], S[i]
        js = np.where(T > ti)[0]
        if js.size == 0:
            continue
        sj = S[js]
        good += np.sum(si > sj)   # si==sj 计 0
        total += js.size
    if total == 0:
        return float("nan")
    return good / total

def auc_formula16(T, E, scores) -> float:
    """
    公式(16)：AUC = (1/num) * sum_{t∈T_event} sum_{y_i<t, δ_i=1} sum_{y_j>t} I(h(x_i) > h(x_j))
    这里 T_event 为测试集中所有发生事件样本的观测时间集合（唯一值）。
    与固定 t 的 ROC-AUC 不同，这是对所有 t 的区分能力按“可比较对数量”加权平均。
    """
    T = np.asarray(T, float).ravel()
    E = np.asarray(E, int).ravel()
    S = np.asarray(scores, float).ravel()

    # 所有事件时刻（唯一、升序）
    event_times = np.sort(np.unique(T[E == 1]))
    good_total = 0
    pair_total = 0

    for t in event_times:
        pos_idx = (E == 1) & (T < t)   # 在 t 前已发生事件
        neg_idx = (T > t)              # 在 t 时仍在风险集中

        n_pos = int(np.sum(pos_idx))
        n_neg = int(np.sum(neg_idx))
        if n_pos == 0 or n_neg == 0:
            continue

        s_pos = S[pos_idx]
        s_neg = S[neg_idx]
        # 两两比较，ties 计 0
        # 使用外积比较以加速：shape (n_pos, n_neg)
        comp = np.greater.outer(s_pos, s_neg)
        good_total += int(np.sum(comp))
        pair_total += int(n_pos * n_neg)

    if pair_total == 0:
        return float("nan")
    return good_total / pair_total

def save_fold_pred(results_root: Path, run_id: int, risk_pred, survtime, event):
    """
    保存每次运行的 test 预测：<root>/run_<id>/<MODEL_NAME>_run<id>_pred_test.pkl
    内容：(risk_pred, survtime, event)
    """
    run_dir = results_root / f"run_{run_id}"
    ensure_dir(run_dir)
    out_pkl = run_dir / f"{MODEL_NAME}_run{run_id}_pred_test.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump((np.asarray(risk_pred), np.asarray(survtime), np.asarray(event)), f)
    return out_pkl

def km_plot_and_save(df: pd.DataFrame, out_png: Path, title_prefix=""):
    """
    “合并（pooled）”KM：把 5 次 run 的 test 记录合并后，
    用 risk_pred 的中位数划分高/低风险，绘一条 KM。
    """
    thr = float(np.median(df["risk_pred"]))
    hi = df["risk_pred"] > thr
    lo = ~hi

    kmf = KaplanMeierFitter()
    plt.rcParams.update({'font.size': 16})  # 全局字体大小
    plt.figure(figsize=(8, 6))
    kmf.fit(df.loc[hi, "os_time"], df.loc[hi, "os_status"], label=f"High risk (n={int(hi.sum())})")
    kmf.plot(ci_show=True, linewidth=2)
    kmf.fit(df.loc[lo, "os_time"], df.loc[lo, "os_status"], label=f"Low risk (n={int(lo.sum())})")
    kmf.plot(ci_show=True, linewidth=2)

    res = logrank_test(
        df.loc[hi, "os_time"], df.loc[lo, "os_time"],
        df.loc[hi, "os_status"], df.loc[lo, "os_status"]
    )

    plt.title(f"{title_prefix} KM (Log-rank p={res.p_value:.2e})")
    plt.xlabel("Time")
    plt.ylabel("Survival probability")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    return float(res.p_value)

def load_from_pkl(pkl_path: str):
    """读取 .pkl，早期融合为特征矩阵 X，并返回 (X, T, E)。"""
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    data = obj.get("datasets", obj)

    X_gene = np.asarray(data["x_gene"], dtype=np.float32)
    X_cna  = np.asarray(data["x_cna"],  dtype=np.float32)
    X_path = np.asarray(data["x_path"], dtype=np.float32)
    T = np.asarray(data["survival"], dtype=float)

    if EVENT_KEY in data:
        E = np.asarray(data[EVENT_KEY], dtype=int)
    elif "event" in data:
        E = np.asarray(data["event"], dtype=int)
    elif "censored" in data:
        E = np.asarray(data["censored"], dtype=int)
    else:
        raise KeyError("pkl 未找到事件标签（event/censored）。")

    X = np.hstack([X_gene, X_cna, X_path]).astype(np.float32)
    print(f"[INFO] Loaded: n={X.shape[0]}, p={X.shape[1]}, event_rate={E.mean():.3f}")
    return X, T, E

def pick_best_alpha_by_validation(est: CoxnetSurvivalAnalysis, X_va, T_va, E_va):
    """
    给定已在训练集上拟合好的 Coxnet（带有一条 alphas_ 路径），
    在验证集上沿路径计算 C-index(公式15)，返回最佳 alpha。
    """
    alphas = getattr(est, "alphas_", None)
    if alphas is None or len(alphas) == 0:
        s = est.predict(X_va)
        c = cindex_formula15(T_va, E_va, s)
        return None, [c]
    c_list = []
    for a in alphas:
        s = est.predict(X_va, alpha=a)
        c = cindex_formula15(T_va, E_va, s)
        c_list.append(c)
    best_idx = int(np.nanargmax(c_list))
    return float(alphas[best_idx]), c_list


# ===== 主流程：单次划分 + 多种子重复 =====
def main():
    X, T, E = load_from_pkl(PKL_PATH)

    results_root = Path(RESULTS_DIR) / EXP_NAME / MODEL_NAME
    ensure_dir(results_root)

    all_runs_rows = []         # 合并 KM 用
    cidx_list, auc_list = [], []

    for r, seed in enumerate(SEEDS, start=1):
        print(f"\n========== Run {r} / seed={seed} ==========")

        # 1) 一次性划分 train+val / test
        X_trva, X_te, T_trva, T_te, E_trva, E_te = train_test_split(
            X, T, E, test_size=TEST_SIZE, random_state=seed, stratify=E
        )
        print(f"[SPLIT] train+val={X_trva.shape}, test={X_te.shape} | "
              f"event_rate trva/test = {E_trva.mean():.3f}/{E_te.mean():.3f}")

        # 2) 从 train+val 再划出验证集
        idx = np.arange(len(T_trva))
        tr_idx, va_idx = train_test_split(
            idx, test_size=VAL_SIZE, random_state=seed, stratify=E_trva
        )
        X_tr, X_va = X_trva[tr_idx], X_trva[va_idx]
        T_tr, T_va = T_trva[tr_idx], T_trva[va_idx]
        E_tr, E_va = E_trva[tr_idx], E_trva[va_idx]

        # 3) 标准化（只用训练集拟合），并变换 val/test 与 train+val
        scaler = StandardScaler().fit(X_tr)
        X_tr = scaler.transform(X_tr).astype(np.float32)
        X_va = scaler.transform(X_va).astype(np.float32)
        X_te_std = scaler.transform(X_te).astype(np.float32)
        X_trva_std = scaler.transform(X_trva).astype(np.float32)

        print(f"[STD] X shapes: train={X_tr.shape}, val={X_va.shape}, test={X_te_std.shape}")

        # 4) 在训练集上拟合整条 λ 路径（固定 l1_ratio）
        y_tr = make_y(T_tr, E_tr)
        est = CoxnetSurvivalAnalysis(
            l1_ratio=L1_RATIO,
            n_alphas=200,
            alpha_min_ratio=1e-7,
            normalize=False,      # 已标准化
            tol=1e-8,
            max_iter=200_000,
            fit_baseline_model=False
        ).fit(X_tr, y_tr)
        print(f"[FIT] path size={len(est.alphas_)} | alphas range=[{est.alphas_[0]:.3e}, {est.alphas_[-1]:.3e}]")

        # 5) 在验证集上选最佳 alpha（λ）——用公式(15)的 C-index
        best_alpha, c_list = pick_best_alpha_by_validation(est, X_va, T_va, E_va)
        print(f"[VAL] C-index along path: min={np.min(c_list):.4f}, max={np.max(c_list):.4f}, "
              f"best_alpha={best_alpha:.3e}")

        # 6) 用 train+val 合并，锁定最佳 α，重训练一次
        y_trva = make_y(T_trva, E_trva)
        est_final = CoxnetSurvivalAnalysis(
            l1_ratio=L1_RATIO,
            alphas=[best_alpha],  # 只用最佳 λ
            normalize=False,
            tol=1e-8,
            max_iter=200_000,
            fit_baseline_model=False
        ).fit(X_trva_std, y_trva)

        # 7) 在 test 上评估（严格按公式）
        s_te = est_final.predict(X_te_std)
        te_cidx = cindex_formula15(T_te, E_te, s_te)   # 公式(15)
        te_auc  = auc_formula16(T_te, E_te, s_te)      # 公式(16)

        cidx_list.append(te_cidx)
        auc_list.append(te_auc)

        # 保存本 run 的预测（KM 合并用）
        out_pkl = save_fold_pred(results_root, r, s_te, T_te, E_te)
        print(f"[Run {r}] Test C-index={te_cidx:.4f} | AUC(16)={te_auc:.4f} | saved: {out_pkl}")

        all_runs_rows.append(pd.DataFrame({
            "risk_pred": s_te.astype(float),
            "os_time":   T_te.astype(float),
            "os_status": E_te.astype(int),
            "run":       r,
            "seed":      seed
        }))

    # —— 汇总结果
    cidx_mean, cidx_std = float(np.nanmean(cidx_list)), float(np.nanstd(cidx_list))
    auc_mean,  auc_std  = float(np.nanmean(auc_list)),  float(np.nanstd(auc_list))

    print("\n===== Summary over 5 runs (single-split protocol) =====")
    print(f"C-index (formula 15): {cidx_mean:.4f} ± {cidx_std:.4f}")
    print(f"AUC     (formula 16): {auc_mean:.4f} ± {auc_std:.4f}")

    # 保存每次的 C/AUC
    results_root = Path(RESULTS_DIR) / EXP_NAME / MODEL_NAME
    ensure_dir(results_root)
    out_metrics_csv = results_root / "metrics_5runs.csv"
    pd.DataFrame({
        "run": list(range(1, len(SEEDS)+1)),
        "seed": SEEDS,
        "cindex_f15": cidx_list,
        "auc_f16": auc_list
    }).to_csv(out_metrics_csv, index=False)
    print(f"Per-run metrics saved to: {out_metrics_csv}")

    # —— 合并 5 次 test 的记录，画“pooled” KM
    merged_df = pd.concat(all_runs_rows, ignore_index=True)
    out_pred_csv = results_root / "out_pred_5runs_pooled.csv"
    merged_df.to_csv(out_pred_csv, index=False)
    print(f"Pooled test predictions saved to: {out_pred_csv}")

    out_km_png = results_root / "lassocox_KM_LGG.png"
    p = km_plot_and_save(merged_df, out_km_png, title_prefix=f"Lasso-Cox")
    print(f"KM (pooled) saved: {out_km_png} | Log-rank p={p:.2e}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()

