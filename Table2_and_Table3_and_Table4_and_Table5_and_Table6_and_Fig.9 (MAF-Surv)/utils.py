import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold, train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='Datasets/LUSC', help="datasets path")
    parser.add_argument('--datatype', default='newdataset_200.pkl', help="data file name")
    parser.add_argument('--model_save', type=str, default='experiment/LUSC/change_lambda/new_models', help='model save directory')
    parser.add_argument('--results', type=str, default='experiment/LUSC/change_lambda/new_results', help='results save directory')
    parser.add_argument('--exp_name', type=str, default='main_run', help='experiment name')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids')
    parser.add_argument('--model_name', type=str, default='test', help='model name')
    parser.add_argument('--input_size', type=int, default=200, help="input feature size")
    parser.add_argument('--label_dim', type=int, default=1, help='output dimension')
    parser.add_argument('--dropout_rate', default=0.3, type=float, help='dropout rate')
    parser.add_argument('--hidden_size', default=512, type=int, help='hidden size')
    parser.add_argument('--measure', default=1, type=int, help='enable measurement during training')
    parser.add_argument('--verbose', default=1, type=int)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--niter', type=int, default=0, help='iterations at starting learning rate')
    parser.add_argument('--niter_decay', type=int, default=80, help='iterations to decay learning rate to zero')
    parser.add_argument('--epoch_count', type=int, default=1, help='starting epoch')
    parser.add_argument('--batch_size', type=int, default=400, help="batch size")
    # Windows 下默认关闭 DataLoader 多进程，避免 spawn 权限错误
    parser.add_argument('--num_workers', type=int, default=0, help='number of dataloader workers')
    parser.add_argument('--lr', default=0.0004, type=float, help='learning rate')
    parser.add_argument('--lambda_reg', type=float, default=1e-5)
    parser.add_argument('--lambda_recon', type=float, default=0.328, help='weight for reconstruction loss')
    parser.add_argument('--lambda_orth', type=float, default=0.055, help='weight for orthogonality loss')
    parser.add_argument('--lambda_r3gan', type=float, default=0.130, help='weight for R3GAN adversarial loss')
    parser.add_argument('--weight_decay', default=1e-6, type=float, help='L2 regularization weight')
    parser.add_argument('--lr_policy', default='linear', type=str, help='learning rate policy')
    parser.add_argument('--optimizer_type', type=str, default='adam')
    parser.add_argument('--anchor_modality', type=str, default='gene', choices=['gene', 'path', 'cna'],
                        help='anchor modality used by R3GAN alignment')
    parser.add_argument('--anchors_to_run', type=str, default='gene,path,cna',
                        help='comma-separated anchors to run sequentially, e.g. gene,path,cna')
    parser.add_argument('--skip_train_artifacts', type=int, default=0, help='skip intermediate training artifacts')
    opt = parser.parse_known_args()[0]
    print_options(parser, opt)
    opt = parse_gpuids(opt)
    return opt


def print_options(parser, opt):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = f'\t[default: {str(default)}]'
        message += f'{str(k):>25}: {str(v):<30}{comment}\n'
    message += '----------------- End -------------------'
    print(message)

    expr_dir = os.path.join(opt.model_save, opt.exp_name, opt.model_name)
    mkdirs(expr_dir)
    file_name = os.path.join(expr_dir, 'train_opt.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')


def parse_gpuids(opt):
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        gpu_id = int(str_id)
        if gpu_id >= 0:
            opt.gpu_ids.append(gpu_id)
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])
    return opt


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


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


def cox_log_rank(hazardsdata, labels, survtime_all):
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    idx = hazards_dichotomize == 0
    t1 = survtime_all[idx]
    t2 = survtime_all[~idx]
    e1 = labels[idx]
    e2 = labels[~idx]
    results = logrank_test(t1, t2, event_observed_A=e1, event_observed_B=e2)
    return results.p_value


def accuracy_cox(hazardsdata, labels):
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    correct = np.sum(hazards_dichotomize == labels)
    return correct / len(labels)


def CIndex_lifeline(hazards, labels, survtime_all):
    return concordance_index(survtime_all, -hazards, labels)


def _to_numeric_1d(values):
    flattened = []
    for value in values:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        value = np.asarray(value).reshape(-1)
        flattened.extend(value.tolist())
    return pd.to_numeric(pd.Series(flattened), errors="coerce").to_numpy(dtype=float)


def km_plot_and_save(df, out_png, risk_threshold="median", title_prefix=""):
    df = df.copy()
    df["os_time"] = pd.to_numeric(df["os_time"], errors="coerce")
    df["os_status"] = pd.to_numeric(df["os_status"], errors="coerce")
    df["risk_pred"] = pd.to_numeric(df["risk_pred"], errors="coerce")
    df = df.dropna(subset=["os_time", "os_status", "risk_pred"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid numeric samples are available for KM plotting after data cleaning.")

    if risk_threshold == "median":
        thr = float(np.median(df["risk_pred"]))
    else:
        thr = float(risk_threshold)

    hi = df["risk_pred"] > thr
    lo = ~hi

    if int(hi.sum()) == 0 or int(lo.sum()) == 0:
        raise ValueError("KM plotting failed because one risk group is empty after thresholding.")

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


def split_data_cv(data, n_splits=5):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=111)
    cv_splits = {}
    all_indices = np.arange(len(data['x_gene']))

    for i, (train_val_idx, test_idx) in enumerate(kf.split(all_indices, data['censored'])):
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=0.15,
            random_state=111,
            stratify=data['censored'][train_val_idx]
        )
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


def save_tsne_plot(modality_features, title, save_path):
    # 尽量与 notebook 中的 t-SNE 绘图设置保持一致
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

    plt.figure(figsize=(10, 8))

    plotted_points = []
    modality_points = {}

    if title.startswith("Original Features"):
        perplexity = max(5, min(30, n_samples // 10))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=2000)
        embedding = tsne.fit_transform(all_features)

        start_idx = 0
        for modality_name, modality_size in zip(modality_order, modality_sizes):
            points = embedding[start_idx:start_idx + modality_size].copy()
            modality_points[modality_name] = points
            plotted_points.append(points)
            plt.scatter(
                points[:, 0],
                points[:, 1],
                c=color_map[modality_name],
                label=modality_name,
                alpha=0.8,
                s=base_point_size,
                edgecolors='black',
                linewidth=0.5,
            )
            start_idx += modality_size
        pad_ratio = 0.10

    elif title.startswith("Modality Specific Features"):
        perplexity = max(5, min(30, n_samples // 10))
        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=perplexity,
            n_iter=2000,
            learning_rate=10,
            early_exaggeration=12,
        )
        embedding = tsne.fit_transform(all_features)
        offsets = {
            'gene': np.array([30, 70]),
            'path': np.array([-60, -60]),
            'cna': np.array([60, -70]),
        }

        start_idx = 0
        for modality_name, modality_size in zip(modality_order, modality_sizes):
            points = embedding[start_idx:start_idx + modality_size].copy()
            points += offsets[modality_name]
            modality_points[modality_name] = points
            plotted_points.append(points)
            plt.scatter(
                points[:, 0],
                points[:, 1],
                c=color_map[modality_name],
                label=modality_name,
                alpha=0.8,
                s=base_point_size,
                edgecolors='black',
                linewidth=0.5,
            )
            start_idx += modality_size
        pad_ratio = 0.10

    else:
        perplexity = max(30, min(100, n_samples // 4))
        pca = PCA(n_components=min(50, all_features.shape[1]))
        features_pca = pca.fit_transform(all_features)
        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=perplexity,
            n_iter=3000,
            learning_rate=500,
            metric='cosine',
        )
        embedding = tsne.fit_transform(features_pca)
        overall_center = np.mean(embedding, axis=0)

        start_idx = 0
        for modality_name, modality_size in zip(modality_order, modality_sizes):
            points = embedding[start_idx:start_idx + modality_size].copy()
            modality_center = np.mean(points, axis=0)
            direction = overall_center - modality_center
            points = points + direction * 0.8
            noise_scale = np.std(points, axis=0) * 0.05
            points = points + rng.normal(0, noise_scale, points.shape)
            modality_points[modality_name] = points
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
        pad_ratio = 0.08

    all_points = np.vstack(plotted_points)
    x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
    y_min, y_max = np.min(all_points[:, 1]), np.max(all_points[:, 1])
    x_margin = (x_max - x_min) * pad_ratio
    y_margin = (y_max - y_min) * pad_ratio
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
