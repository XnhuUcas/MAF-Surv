# utils.py
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
import random
import gc

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='/dataroot/', help="datasets path")
    parser.add_argument('--datatype', default='original_data.pkl', help="data file name")
    parser.add_argument('--model_save', type=str, default='/dataroot/mm/', help='model save directory')
    parser.add_argument('--results', type=str, default='/dataroot/rr/', help='results save directory')
    parser.add_argument('--exp_name', type=str, default='1007', help='experiment name')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids')
    parser.add_argument('--model_name', type=str, default='test', help='model name')
    parser.add_argument('--input_size', type=int, default=80, help="input feature size")
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
    parser.add_argument('--lr', default=0.0004, type=float, help='learning rate')
    parser.add_argument('--lambda_reg', type=float, default=0.000002)
    parser.add_argument('--weight_decay', default=1e-6, type=float, help='L2 regularization weight')
    parser.add_argument('--lr_policy', default='linear', type=str, help='learning rate policy')
    parser.add_argument('--optimizer_type', type=str, default='adam')
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
    file_name = os.path.join(expr_dir, '{}_opt.txt'.format('train'))
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')

def parse_gpuids(opt):
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)
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

def km_plot_and_save(df, out_png, risk_threshold="median", title_prefix=""):
    if risk_threshold == "median":
        thr = float(np.median(df["risk_pred"]))
    else:
        thr = float(risk_threshold)

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

    plt.title(f"{title_prefix} KM (Log-rank p={res.p_value:.2e})")
    plt.xlabel("Time")
    plt.ylabel("Survival probability")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    return float(res.p_value)

def split_data_cv(data, n_splits=5):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=111)
    cv_splits = {}
    all_indices = np.arange(len(data['x_gene']))
    
    for i, (train_val_idx, test_idx) in enumerate(kf.split(all_indices, data['censored'])):
        train_idx, val_idx = train_test_split(train_val_idx, test_size=0.15, random_state=111, stratify=data['censored'][train_val_idx])
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