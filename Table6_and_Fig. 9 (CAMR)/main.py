# -*- coding: utf-8 -*-
import os
import copy

import logging
import numpy as np
import random
import pickle
import time
import matplotlib.pyplot as plt
import torch
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Env
from data_loaders import *
from options import parse_args
from train_test import train, test

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


def set_random_seed(seed):
    # 固定主流程随机性，保证划分、训练和可视化尽量可复现
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(p: str):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def get_dataset_display_name(dataroot: str) -> str:
    # 优先从路径中识别真实数据集名称，避免标题显示成 SIS200 这类子目录名
    normalized = os.path.normpath(dataroot).upper()
    for part in normalized.split(os.sep):
        if part in {"LGG", "LUSC"}:
            return part
    return os.path.basename(normalized)

# def auc_formula16(os_time, os_status, risk_scores) -> float:
#     """
#     严格按 CAMR 文中“式(16)”实现：遍历所有可比对 (i,j)，其中：
#       - i 为事件样本 (os_status[i] == 1)
#       - os_time[i] < os_time[j]
#     记 I( s_i > s_j )，AUC = good / total。ties 计 0。
#     """
#     T = np.asarray(os_time, float).ravel()
#     E = np.asarray(os_status, int).ravel()
#     S = np.asarray(risk_scores, float).ravel()
#     n = len(T)
#     if n == 0:
#         return float("nan")
#
#     order = np.argsort(T)  # 从小到大
#     good = 0
#     total = 0
#     for idx_i in order:
#         if E[idx_i] != 1:
#             continue
#         ti, si = T[idx_i], S[idx_i]
#         # 与更晚时间的所有样本配对
#         js = np.where(T > ti)[0]
#         if js.size == 0:
#             continue
#         sj = S[js]
#         good += np.sum(si > sj)  # ties 计 0
#         total += js.size
#
#     if total == 0:
#         return float("nan")
#     return float(good) / float(total)

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

def km_plot_and_save(df: pd.DataFrame, out_png: str, title_prefix: str = "") -> float:
    """
    合并5折后的 KM 图：以风险分数中位数划分高/低风险。
    返回 log-rank p 值。
    """
    thr = float(np.median(df["risk_pred"]))
    hi = df["risk_pred"] > thr
    lo = ~hi

    kmf = KaplanMeierFitter()
    plt.figure(figsize=(8, 6))
    kmf.fit(df.loc[hi, "os_time"], df.loc[hi, "os_status"], label=f"High risk (n={int(hi.sum())})")
    kmf.plot(ci_show=True, linewidth=2)
    kmf.fit(df.loc[lo, "os_time"], df.loc[lo, "os_status"], label=f"Low risk (n={int(lo.sum())})")
    kmf.plot(ci_show=True, linewidth=2)

    res = logrank_test(
        df.loc[hi, "os_time"], df.loc[lo, "os_time"],
        df.loc[hi, "os_status"], df.loc[lo, "os_status"]
    )

    plt.title(f"{title_prefix} KM (Log-rank p={res.p_value:.2e})", fontsize=22)
    plt.xlabel("Time", fontsize=20)
    plt.ylabel("Survival probability", fontsize=20)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=17)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    return float(res.p_value)


def extract_aligned_features(model, data, device, split=None):
    model.eval()
    split_data = data[split] if split is not None else data
    x_gene = torch.as_tensor(split_data['x_gene'], dtype=torch.float32, device=device)
    x_path = torch.as_tensor(split_data['x_path'], dtype=torch.float32, device=device)
    x_cna = torch.as_tensor(split_data['x_cna'], dtype=torch.float32, device=device)
    with torch.no_grad():
        _, _, _, _, _, _, _, x7, x8, x9, _, _, _ = model(x_gene, x_path, x_cna)
    return {
        'gene': x7.detach().cpu().numpy(),
        'path': x8.detach().cpu().numpy(),
        'cna': x9.detach().cpu().numpy(),
    }


def save_tsne_plot(modality_features, title, save_path):
    color_map = {
        'gene': '#FF0000',
        'path': '#00CC00',
        'cna': '#0000FF',
    }
    rng = np.random.RandomState(42)
    modality_order = ['gene', 'path', 'cna']
    feature_list = [np.asarray(modality_features[name], dtype=np.float32) for name in modality_order]
    modality_sizes = [len(features) for features in feature_list]
    all_features = np.concatenate(feature_list, axis=0)

    n_samples = len(all_features)
    base_point_size = 30 if n_samples > 1000 else 50
    perplexity = max(30, min(100, n_samples // 4))
    pca = PCA(n_components=min(50, all_features.shape[1]))
    features_pca = pca.fit_transform(all_features)
    embedding = TSNE(
        n_components=2,
        random_state=42,
        perplexity=perplexity,
        n_iter=3000,
        learning_rate=500,
        metric='cosine',
    ).fit_transform(features_pca)

    overall_center = np.mean(embedding, axis=0)
    plt.figure(figsize=(10, 8))
    plotted_points = []
    start_idx = 0
    for modality_name, modality_size in zip(modality_order, modality_sizes):
        points = embedding[start_idx:start_idx + modality_size].copy()
        modality_center = np.mean(points, axis=0)
        direction = overall_center - modality_center
        points = points + direction * 0.8
        noise_scale = np.std(points, axis=0) * 0.05
        points = points + rng.normal(0, noise_scale, points.shape)
        plotted_points.append(points)
        plt.scatter(
            points[:, 0],
            points[:, 1],
            c=color_map[modality_name],
            label=modality_name,
            alpha=0.7,
            s=40,
            edgecolors='white',
            linewidth=0.2,
        )
        start_idx += modality_size

    all_points = np.vstack(plotted_points)
    x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
    y_min, y_max = np.min(all_points[:, 1]), np.max(all_points[:, 1])
    x_margin = (x_max - x_min) * 0.08
    y_margin = (y_max - y_min) * 0.08
    plt.xlim(x_min - x_margin, x_max + x_margin)
    plt.ylim(y_min - y_margin, y_max + y_margin)

    plt.title(title, fontsize=22)
    plt.xlabel("t-SNE Component 1", fontsize=20)
    plt.ylabel("t-SNE Component 2", fontsize=20)
    plt.legend(fontsize=17)
    plt.grid(True, alpha=0.2)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def generate_fold0_tsne(opt, full_data, device, out_root, model):
    # 与当前 MAF-Surv 保持一致：使用第一折最优模型直接在完整数据集上提特征
    aligned_features = extract_aligned_features(model, full_data, device, split=None)
    dataset_name = get_dataset_display_name(opt.dataroot)
    save_tsne_plot(
        aligned_features,
        f'CAMR Aligned Features ({dataset_name})',
        os.path.join(out_root, f'tsne_camr_aligned_features_{dataset_name.lower()}.png'),
    )

opt = parse_args()
set_random_seed(opt.seed)
device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
print("Using device:", device)
print(torch.cuda.device_count())  # 打印gpu数量
if not os.path.exists(os.path.join(opt.model_save, opt.exp_name, opt.model_name)):
        os.makedirs(os.path.join(opt.model_save, opt.exp_name, opt.model_name))

def split_data_cv(data, n_splits=5):
    # 使用分层K折交叉验证分割数据
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=111)
    cv_splits = {}
    all_indices = np.arange(len(data['x_gene']))

    for i, (train_val_idx, test_idx) in enumerate(kf.split(all_indices, data['censored'])):
        # 进一步划分训练集和验证集
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=0.15, random_state=111,
            stratify=data['censored'][train_val_idx])

        # 存储分割后的数据
        cv_splits[i] = {
            'train': {
                'x_gene': data['x_gene'][train_idx],
                'x_path': data['x_path'][train_idx],
                'x_cna': data['x_cna'][train_idx],
                'censored': data['censored'][train_idx],
                'survival': data['survival'][train_idx]
            },
            'val': {
                'x_gene': data['x_gene'][val_idx],
                'x_path': data['x_path'][val_idx],
                'x_cna': data['x_cna'][val_idx],
                'censored': data['censored'][val_idx],
                'survival': data['survival'][val_idx]
            },
            'test': {
                'x_gene': data['x_gene'][test_idx],
                'x_path': data['x_path'][test_idx],
                'x_cna': data['x_cna'][test_idx],
                'censored': data['censored'][test_idx],
                'survival': data['survival'][test_idx]
            }
        }
    return cv_splits

total_start_time = time.time()
data_cv_path = '%s%s' % (opt.dataroot, opt.datatype)
print("Loading %s" % data_cv_path)
data_cv = pickle.load(open(data_cv_path, 'rb'))['datasets']
data_cv_splits = split_data_cv(data_cv)
results=[]

average_results = []
best_results = []
os_time,os_status,risk_pred = [],[],[]
best_os_time,best_os_status,best_risk_pred = [],[],[]

code_pred =[]
label_pred = []
auc_list = []
fold0_model = None
out_root = os.path.join(opt.results, opt.exp_name, opt.model_name)
print("[SAVE ROOT]", out_root)
ensure_dir(out_root)
### 3. Sets-Up Main Loop
for k, data in data_cv_splits.items():
    print("*******************************************")
    print("************** SPLIT (%d/%d) **************" % (k, len(data_cv_splits.items())))
    print("*******************************************")

    ### ### ### ### ### ### ### ### ###创建文件夹存储结果### ### ### ### ### ### ### ### ### ###
    # if not os.path.exists(os.path.join(opt.results,opt.exp_name, opt.model_name,'%d_fold'%(k))): os.makedirs(
    #     os.path.join(opt.results,opt.exp_name, opt.model_name,'%d_fold'%(k)))
    fold_dir = os.path.join(opt.results, opt.exp_name, opt.model_name, '%d_fold' % (k))
    print("[FOLD SAVE]", fold_dir)
    ensure_dir(fold_dir)

    ### 3.1 Trains Model
    model, optimizer, metric_logger = train(opt, data, device, k)
    if k == 0:
        # 保留第一折验证集最优模型，后续直接在完整数据集上做 t-SNE
        fold0_model = copy.deepcopy(model)
    epochs_list = range(opt.epoch_count, opt.niter+opt.niter_decay+1)
    ### 3.2 Evalutes Train + Test Error, and Saves Model
    loss_train, cindex_train, pvalue_train, surv_acc_train, pred_train= test(opt,model,data, 'train',device)
    loss_test, cindex_test, pvalue_test, surv_acc_test, pred_test=test(opt, model, data, 'test', device)

   
    print("[Final] Apply model to training set: C-Index: %.10f, P-Value: %.10e" % (cindex_train, pvalue_train))
    logging.info("[Final] Apply model to training set: C-Index: %.10f, P-Value: %.10e" % (cindex_train, pvalue_train))
    print("[Final] Apply model to testing set: C-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
    logging.info("[Final] Apply model to testing set: cC-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
  
    average_results.append(cindex_test)

    s_te = np.asarray(pred_test[0])
    t_te = np.asarray(pred_test[1])
    e_te = np.asarray(pred_test[2])
    auc16_k = auc_formula16(t_te, e_te, s_te)
    auc_list.append(float(auc16_k))
    with open(os.path.join(fold_dir, "AUC_formula16.txt"), "w", encoding="utf-8") as f:
        f.write(f"{auc16_k:.10f}\n")
    print(f"[Fold {k}] AUC (formula 16) = {auc16_k:.6f}")

    ### 3.3 Saves Model
    if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
        model_state_dict = model.state_dict()
    else:
        model_state_dict = model.state_dict()
    torch.save({
        'split':k,
        'opt': opt,
        'epoch': opt.niter+opt.niter_decay,
        'data': data,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metric_logger},
        os.path.join(opt.model_save, opt.exp_name, opt.model_name, '%s_%d.pt' % (opt.model_name, k))
    )
    print()

    # pickle.dump(pred_train, open(os.path.join(opt.results,opt.exp_name, opt.model_name,'%d_fold'%(k), '%s_%dpred_train.pkl' % (opt.model_name, k)), 'wb'))
    # pickle.dump(pred_test, open(os.path.join(opt.results,opt.exp_name, opt.model_name,'%d_fold'%(k), '%s_%dpred_test.pkl' % (opt.model_name, k)), 'wb'))
    # df = pd.DataFrame({'os_time': pred_test[1], 'os_status': pred_test[2], 'risk_pred': pred_test[0]})
    # df.to_csv(opt.results + "%d-fold_pred.csv" % (k), index=0, header=1)
    pickle.dump(pred_train, open(os.path.join(fold_dir, '%s_%dpred_train.pkl' % (opt.model_name, k)), 'wb'))
    pickle.dump(pred_test, open(os.path.join(fold_dir, '%s_%dpred_test.pkl' % (opt.model_name, k)), 'wb'))
    df_fold = pd.DataFrame({'os_time': pred_test[1], 'os_status': pred_test[2], 'risk_pred': pred_test[0]})
    # === MOD: 每折 CSV 存到每折目录
    df_fold.to_csv(os.path.join(fold_dir, "%d-fold_pred.csv" % k), index=0, header=1)

    PI = pred_test[0] > np.median(pred_test[0])
    #np.savetxt(opt.results + "%d-fold_label_test.csv" % (k), PI + 0, delimiter=",")
    np.savetxt(os.path.join(fold_dir, "%d-fold_label_test.csv" % k), PI + 0, delimiter=",")
    risk_pred.extend(pred_test[0])
    os_time.extend(pred_test[1])
    os_status.extend(pred_test[2])
    label_pred.extend(PI + 0)

# df = pd.DataFrame({'os_time':os_time,'os_status':os_status,'risk_pred':risk_pred})
# df.to_csv(opt.results + "out_pred_5fold.csv", index=0, header=1)
#
# df2 = pd.DataFrame({'os_time':os_time,'os_status':os_status,'risk_pred':label_pred})
# df2.to_csv(opt.results + "risk_pred_5fold.csv", index=0, header=1)
#
# np.savetxt(opt.results + "label_test.csv", label_pred, fmt="%d",delimiter=",")
# np.savetxt(opt.results + "split_average_results.csv", average_results, delimiter=",")
# print('Split Average Results:', average_results)
# print('Split Best Results:', best_results)
# print("Average_results:", np.array(average_results).mean()," std: ", np.std(average_results,ddof = 0))
# pickle.dump(average_results, open(os.path.join(opt.results, opt.exp_name, opt.model_name, '%s_results.pkl' % opt.model_name), 'wb'))
# === MOD: 合并后的 CSV 都放到固定输出根目录 ===
df = pd.DataFrame({'os_time':os_time,'os_status':os_status,'risk_pred':risk_pred})
df.to_csv(os.path.join(out_root, "out_pred_5fold.csv"), index=0, header=1)

df2 = pd.DataFrame({'os_time':os_time,'os_status':os_status,'risk_pred':label_pred})
df2.to_csv(os.path.join(out_root, "risk_pred_5fold.csv"), index=0, header=1)

np.savetxt(os.path.join(out_root, "label_test.csv"), label_pred, fmt="%d",delimiter=",")
np.savetxt(os.path.join(out_root, "split_average_results.csv"), average_results, delimiter=",")
print('Split Average Results:', average_results)
print('Split Best Results:', best_results)
print("Average_results:", np.array(average_results).mean()," std: ", np.std(average_results,ddof = 0))
pickle.dump(average_results, open(os.path.join(out_root, '%s_results.pkl' % opt.model_name), 'wb'))

# === NEW: 输出 AUC(式16) 的 5 折均值/标准差，并保存 CSV（便于与其它方法汇总画箱线图）
auc_mean = float(np.nanmean(auc_list)) if len(auc_list) else float("nan")
auc_std  = float(np.nanstd(auc_list, ddof=1)) if len(auc_list) > 1 else 0.0
print(f"AUC (formula 16) over 5 folds: {auc_mean:.4f} ± {auc_std:.4f}")
pd.DataFrame({"fold": np.arange(len(auc_list)), "AUC_formula16": auc_list}).to_csv(
    os.path.join(out_root, "AUC_formula16_CAMR.csv"), index=False
)
print(f"AUC (formula 16, ddof=1) over 5 folds: {auc_mean:.4f} ± {auc_std:.4f}")
pd.DataFrame([{
    "metric": "auc_formula16",
    "mean": auc_mean,
    "std": auc_std,
}]).to_csv(os.path.join(out_root, "summary_metrics.csv"), index=False)

# === NEW: 画“5 折合并”的 KM 曲线（与其它方法保持一致）
km_png = os.path.join(out_root, "KM_5fold_merged.png")
p_val = km_plot_and_save(df, km_png, title_prefix="CAMR")
print(f"KM saved: {km_png} | Log-rank p={p_val:.2e}")
if fold0_model is not None:
    generate_fold0_tsne(opt, data_cv, device, out_root, fold0_model)
total_elapsed_seconds = time.time() - total_start_time
print(f"C-index over 5 folds: {np.mean(average_results):.4f} ± {np.std(average_results, ddof=1) if len(average_results) > 1 else 0.0:.4f}")
print(f"AUC over 5 folds: {auc_mean:.4f} ± {auc_std:.4f}")
print(f"Total runtime: {total_elapsed_seconds:.2f} seconds ({total_elapsed_seconds / 60.0:.2f} minutes)")
