# setup.py
from setuptools import setup, find_packages

setup(
    name="mafsurv",
    version="1.0.0",
    description="MAF-Surv: Enhanced Cancer Survival PredictionFramework with Multimodal Data Fusion",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.0",
        "lifelines>=0.26.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.62.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)

# dataset.py
import torch
from torch.utils.data.dataset import Dataset

class GraphFusionDatasetLoader(Dataset):
    def __init__(self, data, split):
        self.X_gene = data[split]['x_gene']
        self.X_path = data[split]['x_path']
        self.X_cna = data[split]['x_cna']
        self.censored = data[split]['censored']
        self.survival = data[split]['survival']

    def __getitem__(self, index):
        single_censored = torch.tensor(self.censored[index]).type(torch.FloatTensor)
        single_survival = torch.tensor(self.survival[index]).type(torch.FloatTensor)
        single_X_gene = torch.tensor(self.X_gene[index]).type(torch.FloatTensor)
        single_X_path = torch.tensor(self.X_path[index]).type(torch.FloatTensor)
        single_X_cna = torch.tensor(self.X_cna[index]).type(torch.FloatTensor)

        return single_X_gene, single_X_path, single_X_cna, single_censored, single_survival

    def __len__(self):
        return len(self.X_gene)


# models.py
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_
from torch.nn import Parameter
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, in_size):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_size, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, z):
        return self.model(z)


class PTP(nn.Module):
    def __init__(self, input_dim, rank, output_dim=60, poly_order=2):
        super(PTP, self).__init__()
        self.input_dim = input_dim
        self.rank = rank
        self.poly_order = poly_order
        self.output_dim = output_dim

        self.expanded_dim = input_dim + 1
        self.poly_dim = self.expanded_dim * poly_order

        self.factors = nn.ParameterList([
            nn.Parameter(torch.Tensor(self.poly_dim, rank))
            for _ in range(3)
        ])
        self.fusion_weights = nn.Parameter(torch.Tensor(rank, output_dim))
        self.fusion_bias = nn.Parameter(torch.Tensor(output_dim))

        for factor in self.factors:
            xavier_normal_(factor)
        xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def forward(self, audio_x, video_x, text_x):
        batch_size = audio_x.shape[0]
        device = audio_x.device

        def _expand(x):
            x = torch.cat([torch.ones(batch_size, 1).to(device), x], dim=1)
            return torch.cat([x ** (i + 1) for i in range(self.poly_order)], dim=1)

        audio_poly = _expand(audio_x)
        video_poly = _expand(video_x)
        text_poly = _expand(text_x)

        fusion_audio = torch.matmul(audio_poly, self.factors[0])
        fusion_video = torch.matmul(video_poly, self.factors[1])
        fusion_text = torch.matmul(text_poly, self.factors[2])

        fusion_zy = fusion_audio * fusion_video * fusion_text
        output = torch.matmul(fusion_zy, self.fusion_weights) + self.fusion_bias

        return output


class MisaPTPGatedRec(nn.Module):
    def __init__(self, in_size, output_dim, hidden_size=20, hidden_size1=80, dropout=0.1):
        super(MisaPTPGatedRec, self).__init__()

        self.common = nn.Sequential(nn.Linear(in_size, 320), nn.ReLU(),
                                    nn.Linear(320, 128), nn.ReLU(),
                                    nn.Linear(128, 60), nn.ReLU())
        self.ptp = PTP(input_dim=60, rank=16, output_dim=60, poly_order=2)
        self.unique1 = nn.Sequential(nn.Linear(in_size, 320), nn.ReLU(),
                                     nn.Linear(320, 60), nn.ReLU())
        self.unique2 = nn.Sequential(nn.Linear(in_size, 256), nn.ReLU(),
                                     nn.Linear(256, 128), nn.ReLU(),
                                     nn.Linear(128, 60), nn.ReLU())
        self.unique3 = nn.Sequential(nn.Linear(in_size, 260), nn.ReLU(),
                                     nn.Linear(260, 130), nn.ReLU(),
                                     nn.Linear(130, 60), nn.ReLU())

        encoder1 = nn.Sequential(nn.Linear(60 * 4, 900), nn.ReLU(), nn.Dropout(p=dropout))
        encoder2 = nn.Sequential(nn.Linear(900, 512), nn.ReLU(), nn.Dropout(p=dropout))
        encoder3 = nn.Sequential(nn.Linear(512, 64), nn.ReLU(), nn.Dropout(p=dropout))
        encoder4 = nn.Sequential(nn.Linear(64, 15), nn.ReLU(), nn.Dropout(p=dropout))
        self.encoder = nn.Sequential(encoder1, encoder2, encoder3, encoder4)
        self.classifier = nn.Sequential(nn.Linear(15, output_dim), nn.Sigmoid())
        self.output_range = Parameter(torch.FloatTensor([8]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-4]), requires_grad=False)

        self.linear_h1 = nn.Sequential(nn.Linear(60, 60), nn.ReLU())
        self.linear_z1 = nn.Bilinear(60, 120, 60)
        self.linear_o1 = nn.Sequential(nn.Linear(60, 60), nn.ReLU(), nn.Dropout(p=dropout))

        self.linear_h2 = nn.Sequential(nn.Linear(60, 60), nn.ReLU())
        self.linear_z2 = nn.Bilinear(60, 120, 60)
        self.linear_o2 = nn.Sequential(nn.Linear(60, 60), nn.ReLU(), nn.Dropout(p=dropout))

        self.linear_h3 = nn.Sequential(nn.Linear(60, 60), nn.ReLU())
        self.linear_z3 = nn.Bilinear(60, 120, 60)
        self.linear_o3 = nn.Sequential(nn.Linear(60, 60), nn.ReLU(), nn.Dropout(p=dropout))

        self.rec1 = nn.Sequential(nn.Linear(120, 256), nn.ReLU(),
                                  nn.Linear(256, hidden_size1), nn.ReLU())
        self.rec2 = nn.Sequential(nn.Linear(120, 256), nn.ReLU(),
                                  nn.Linear(256, hidden_size1), nn.ReLU())
        self.rec3 = nn.Sequential(nn.Linear(120, 256), nn.ReLU(),
                                  nn.Linear(256, hidden_size1), nn.ReLU())

    def forward(self, x_gene, x_path, x_can):
        x_gene_common = self.common(x_gene)
        x_path_common = self.common(x_path)
        x_can_common = self.common(x_can)

        h1 = self.linear_h1(x_gene_common)
        vec31 = torch.cat((x_path_common, x_can_common), dim=1)
        z1 = self.linear_z1(x_gene_common, vec31)
        o1 = self.linear_o1(nn.Sigmoid()(z1) * h1)

        h2 = self.linear_h1(x_path_common)
        vec32 = torch.cat((x_gene_common, x_can_common), dim=1)
        z2 = self.linear_z1(x_path_common, vec32)
        o2 = self.linear_o1(nn.Sigmoid()(z2) * h2)

        h3 = self.linear_h1(x_can_common)
        vec33 = torch.cat((x_gene_common, x_path_common), dim=1)
        z3 = self.linear_z1(x_path_common, vec33)
        o3 = self.linear_o1(nn.Sigmoid()(z3) * h3)

        ptp = self.ptp(o1, o2, o3)

        x_gene_unique = self.unique1(x_gene)
        x_path_unique = self.unique2(x_path)
        x_can_unique = self.unique3(x_can)

        out_fusion = torch.cat((ptp, x_gene_unique, x_path_unique, x_can_unique), dim=1)
        encoder = self.encoder(out_fusion)
        out = self.classifier(encoder)
        out = out * self.output_range + self.output_shift

        gene_rec = self.rec1(torch.cat((ptp, x_gene_unique), dim=1))
        path_rec = self.rec2(torch.cat((ptp, x_path_unique), dim=1))
        can_rec = self.rec3(torch.cat((ptp, x_can_unique), dim=1))

        return out, x_gene, gene_rec, x_path, path_rec, x_can, can_rec, \
            x_gene_common, x_path_common, x_can_common, \
            x_gene_unique, x_path_unique, x_can_unique, ptp


# losses.py
import torch
import torch.nn as nn
import numpy as np

def R_set(x):
    n_sample = x.size(0)
    matrix_ones = torch.ones(n_sample, n_sample)
    indicator_matrix = torch.tril(matrix_ones)
    return indicator_matrix

def regularize_weights(model, reg_type=None):
    l1_reg = None
    for W in model.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            l1_reg = l1_reg + torch.abs(W).sum()
    return l1_reg

def cox_log_rank(hazardsdata, labels, survtime_all):
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    idx = hazards_dichotomize == 0
    T1 = survtime_all[idx]
    T2 = survtime_all[~idx]
    E1 = labels[idx]
    E2 = labels[~idx]
    from lifelines.statistics import logrank_test
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    return results.p_value

def accuracy_cox(hazardsdata, labels):
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    correct = np.sum(hazards_dichotomize == labels)
    return correct / len(labels)

def CIndex_lifeline(hazards, labels, survtime_all):
    from lifelines.utils import concordance_index
    return concordance_index(survtime_all, -hazards, labels)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def CoxLoss2(survtime, censor, hazard_pred, device):
    n_observed = censor.sum(0) + 1
    ytime_indicator = R_set(survtime)
    ytime_indicator = torch.FloatTensor(ytime_indicator).to(device)
    risk_set_sum = ytime_indicator.mm(torch.exp(hazard_pred))
    diff = hazard_pred - torch.log(risk_set_sum)
    sum_diff_in_observed = torch.transpose(diff, 0, 1).mm(censor.unsqueeze(1))
    cost = (- (sum_diff_in_observed / n_observed)).reshape((-1,))
    return cost

class DiffLoss(nn.Module):
    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):
        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        input1_mean = torch.mean(input1, dim=0, keepdims=True)
        input2_mean = torch.mean(input2, dim=0, keepdims=True)
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))
        return diff_loss

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n - torch.sum(diffs.pow(2)) / (n * n)
        return mse


# optimizers.py
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

def define_optimizer(opt, model):
    if opt.optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=opt.weight_decay)
    elif opt.optimizer_type == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay, initial_accumulator_value=0.1)
    elif opt.optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=opt.weight_decay)
    elif opt.optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        raise NotImplementedError(f'optimizer {opt.optimizer_type} not implemented')
    return optimizer

def define_scheduler(opt, optimizer):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'exp':
        scheduler = lr_scheduler.ExponentialLR(optimizer, 0.1, last_epoch=-1)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        raise NotImplementedError(f'learning rate policy {opt.lr_policy} not implemented')
    return scheduler


# train.py
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import pickle
from tqdm import tqdm

from models import MisaPTPGatedRec, Discriminator
from dataset import GraphFusionDatasetLoader
from losses import CoxLoss2, DiffLoss, MSE, regularize_weights
from optimizers import define_optimizer, define_scheduler
from utils import auc_formula16, CIndex_lifeline, cox_log_rank, accuracy_cox


class AdversarialTraining:
    def __init__(self, Generator, Discriminator):
        self.Generator = Generator
        self.Discriminator = Discriminator

    @staticmethod
    def ZeroCenteredGradientPenalty(Samples, Critics):
        if not Samples.requires_grad:
            Samples.requires_grad_(True)

        Gradient, = torch.autograd.grad(
            outputs=Critics.sum(),
            inputs=Samples,
            create_graph=True,
            retain_graph=True
        )
        return Gradient.square().sum(1)


def train(opt, data, device, k):
    cudnn.deterministic = True
    torch.cuda.manual_seed_all(111)
    torch.manual_seed(111)

    model = MisaPTPGatedRec(opt.input_size, opt.label_dim).to(device)
    diff = DiffLoss()
    mse = MSE()
    discr = Discriminator(60).to(device)

    optimizer = define_optimizer(opt, model)
    optimizer_d = torch.optim.Adam(discr.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
    scheduler = define_scheduler(opt, optimizer)

    print(model)
    print(f"Number of Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    custom_data_loader = GraphFusionDatasetLoader(data, split='train')
    train_loader = DataLoader(dataset=custom_data_loader, batch_size=len(custom_data_loader), num_workers=4,
                              shuffle=False)

    metric_logger = {
        'train': {'loss': [], 'pvalue': [], 'cindex': [], 'surv_acc': [], 'auc': []},
        'test': {'loss': [], 'pvalue': [], 'cindex': [], 'surv_acc': [], 'auc': []}
    }

    c_index_best = 0
    patience = 70
    min_delta = 0.005
    patience_counter = 0
    best_val_cindex = 0
    best_model_state = None
    early_stop_counter = 0

    for epoch in tqdm(range(opt.epoch_count, opt.niter + opt.niter_decay + 1)):
        model.train()
        discr.train()
        risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])

        for batch_idx, (x_gene, x_path, x_cna, censor, survtime) in enumerate(train_loader):
            censor = censor.to(device)
            x_gene = x_gene.view(x_gene.size(0), -1).to(device)
            x_path = x_path.view(x_path.size(0), -1).to(device)
            x_cna = x_cna.view(x_cna.size(0), -1).to(device)

            pred, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, ptp = model(
                x_gene.to(device), x_path.to(device), x_cna.to(device)
            )

            optimizer_d.zero_grad()
            real_samples = x7.detach().requires_grad_(True)
            fake_samples_1 = x8.detach().requires_grad_(True)
            fake_samples_2 = x9.detach().requires_grad_(True)

            real_logits = discr(real_samples)
            fake_logits_1 = discr(fake_samples_1)
            fake_logits_2 = discr(fake_samples_2)

            relativistic_logits_1 = fake_logits_1 - real_logits
            relativistic_logits_2 = fake_logits_2 - real_logits

            adversarial_loss_1 = nn.functional.softplus(-relativistic_logits_1).mean()
            adversarial_loss_2 = nn.functional.softplus(-relativistic_logits_2).mean()
            adversarial_loss = (adversarial_loss_1 + adversarial_loss_2) / 2

            gamma = 10
            R1_penalty = AdversarialTraining.ZeroCenteredGradientPenalty(real_samples, real_logits).mean()
            R2_penalty_1 = AdversarialTraining.ZeroCenteredGradientPenalty(fake_samples_1, fake_logits_1).mean()
            R2_penalty_2 = AdversarialTraining.ZeroCenteredGradientPenalty(fake_samples_2, fake_logits_2).mean()

            d_loss = adversarial_loss + (gamma / 2) * (R1_penalty + R2_penalty_1 + R2_penalty_2)
            d_loss.backward()
            optimizer_d.step()

            optimizer.zero_grad()
            real_logits = discr(x7)
            fake_logits_1 = discr(x8)
            fake_logits_2 = discr(x9)

            relativistic_logits = fake_logits_1 - real_logits
            g_loss_adv = nn.functional.softplus(relativistic_logits).mean()

            diff_loss = (diff(x7, x10) + diff(x8, x11) + diff(x9, x12)) / 3
            rec_loss = mse(x1, x2) + mse(x3, x4) + mse(x5, x6)
            loss_cox = CoxLoss2(survtime, censor, pred, device)
            loss_reg = regularize_weights(model=model)

            total_loss = (
                    loss_cox +
                    1e-5 * loss_reg +
                    0.3 * rec_loss +
                    0.1 * diff_loss +
                    0.1 * g_loss_adv
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            torch.cuda.empty_cache()

            risk_pred_all = np.concatenate((risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))
            censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))
            survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))

        scheduler.step()

        if opt.measure or epoch == (opt.niter + opt.niter_decay - 1):
            cindex_epoch = CIndex_lifeline(risk_pred_all, censor_all, survtime_all)
            pvalue_epoch = cox_log_rank(risk_pred_all, censor_all, survtime_all)
            surv_acc_epoch = accuracy_cox(risk_pred_all, censor_all)
            auc_epoch = auc_formula16(survtime_all, censor_all, risk_pred_all)

            loss_test, cindex_test, pvalue_test, surv_acc_test, auc_test, pred_test = test(opt, model, data, 'test',
                                                                                           device)

            metric_logger['train']['cindex'].append(cindex_epoch)
            metric_logger['train']['pvalue'].append(pvalue_epoch)
            metric_logger['train']['surv_acc'].append(surv_acc_epoch)
            metric_logger['train']['auc'].append(auc_epoch)

            metric_logger['test']['cindex'].append(cindex_test)
            metric_logger['test']['pvalue'].append(pvalue_test)
            metric_logger['test']['surv_acc'].append(surv_acc_test)
            metric_logger['test']['auc'].append(auc_test)

            pickle.dump(pred_test, open(os.path.join(
                opt.results, opt.exp_name, opt.model_name, f'{k}_fold',
                f'{opt.model_name}_{epoch}_pred_test.pkl'), 'wb'))

            if cindex_test > c_index_best:
                c_index_best = cindex_test

        val_loss, val_cindex, val_pvalue, val_surv_acc, val_auc, _ = test(opt, model, data, 'val', device)

        if val_cindex > best_val_cindex + min_delta:
            best_val_cindex = val_cindex
            best_model_state = model.state_dict().copy()
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(
                opt.model_save, opt.exp_name, opt.model_name, f'best_model_fold_{k}.pt'))
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        print("Warning: No best model state found, using final model")

    return model, optimizer, metric_logger


def test(opt, model, data, split, device):
    model.eval()
    custom_data_loader = GraphFusionDatasetLoader(data, split)
    test_loader = DataLoader(dataset=custom_data_loader, batch_size=len(custom_data_loader), num_workers=4,
                             shuffle=False)

    risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])
    loss_test = 0

    with torch.no_grad():
        for batch_idx, (x_gene, x_path, x_cna, censor, survtime) in enumerate(test_loader):
            censor = censor.to(device)
            x_gene = x_gene.view(x_gene.size(0), -1)
            x_path = x_path.view(x_path.size(0), -1)
            x_cna = x_cna.view(x_cna.size(0), -1)

            pred, _, _, _, _, _, _, _, _, _, _, _, _, _ = model(
                x_gene.to(device), x_path.to(device), x_cna.to(device)
            )

            loss_cox = CoxLoss2(survtime, censor, pred, device)
            loss_test += loss_cox.data.item()

            risk_pred_all = np.concatenate((risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))
            censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))
            survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))

    loss_test /= len(test_loader.dataset)
    cindex_test = CIndex_lifeline(risk_pred_all, censor_all, survtime_all)
    pvalue_test = cox_log_rank(risk_pred_all, censor_all, survtime_all)
    surv_acc_test = accuracy_cox(risk_pred_all, censor_all)
    auc_test = auc_formula16(survtime_all, censor_all, risk_pred_all)

    pred_test = [risk_pred_all, survtime_all, censor_all]
    return loss_test, cindex_test, pvalue_test, surv_acc_test, auc_test, pred_test


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
        train_idx, val_idx = train_test_split(train_val_idx, test_size=0.15, random_state=111,
                                              stratify=data['censored'][train_val_idx])
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


# main.py
import os
import pickle
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from utils import parse_args, split_data_cv, km_plot_and_save
from train import train, test


def main():
    opt = parse_args()
    device = torch.device(f'cuda:{opt.gpu_ids[0]}') if opt.gpu_ids else torch.device('cpu')
    print(f"Using device: {device}")
    print(f"GPU count: {torch.cuda.device_count()}")

    os.makedirs(os.path.join(opt.model_save, opt.exp_name, opt.model_name), exist_ok=True)

    data_cv_path = os.path.join(opt.dataroot, opt.datatype)
    print(f"Loading {data_cv_path}")
    data_cv = pickle.load(open(data_cv_path, 'rb'))['datasets']
    data_cv_splits = split_data_cv(data_cv)

    average_results = {
        'cindex': [],
        'auc': []
    }

    os_time, os_status, risk_pred = [], [], []

    for k, data in data_cv_splits.items():
        print("*******************************************")
        print(f"************** SPLIT ({k}/{len(data_cv_splits.items()) - 1}) **************")
        print("*******************************************")

        fold_dir = os.path.join(opt.results, opt.exp_name, opt.model_name, f'{k}_fold')
        os.makedirs(fold_dir, exist_ok=True)

        model, optimizer, metric_logger = train(opt, data, device, k)

        loss_train, cindex_train, pvalue_train, surv_acc_train, auc_train, pred_train = test(opt, model, data, 'train',
                                                                                             device)
        loss_test, cindex_test, pvalue_test, surv_acc_test, auc_test, pred_test = test(opt, model, data, 'test', device)

        print(f"[Final] Training set: C-Index: {cindex_train:.10f}, P-Value: {pvalue_train:.10e}, AUC: {auc_train:.4f}")
        print(f"[Final] Testing set: C-Index: {cindex_test:.10f}, P-Value: {pvalue_test:.10e}, AUC: {auc_test:.4f}")

        average_results['cindex'].append(cindex_test)
        average_results['auc'].append(auc_test)

        torch.save({
            'split': k,
            'opt': opt,
            'epoch': opt.niter + opt.niter_decay,
            'data': data,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metric_logger
        }, os.path.join(opt.model_save, opt.exp_name, opt.model_name, f'{opt.model_name}_{k}.pt'))

        print()

        pickle.dump(pred_train, open(os.path.join(fold_dir, f'{opt.model_name}_{k}pred_train.pkl'), 'wb'))
        pickle.dump(pred_test, open(os.path.join(fold_dir, f'{opt.model_name}_{k}pred_test.pkl'), 'wb'))

        df = pd.DataFrame({'os_time': pred_test[1], 'os_status': pred_test[2], 'risk_pred': pred_test[0]})
        df.to_csv(os.path.join(opt.results, f"{k}-fold_pred.csv"), index=False, header=True)

        PI = pred_test[0] > np.median(pred_test[0])
        np.savetxt(os.path.join(opt.results, f"{k}-fold_label_test.csv"), PI + 0, delimiter=",")

        risk_pred.extend(pred_test[0])
        os_time.extend(pred_test[1])
        os_status.extend(pred_test[2])

    cindex_values = average_results['cindex']
    auc_values = average_results['auc']

    cindex_mean = np.mean(cindex_values)
    cindex_std = np.std(cindex_values, ddof=1)
    auc_mean = np.mean(auc_values)
    auc_std = np.std(auc_values, ddof=1)

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print("Per-fold Results:")
    for i, (cindex, auc) in enumerate(zip(cindex_values, auc_values)):
        print(f"  Fold {i + 1}: C-index = {cindex:.4f}, AUC = {auc:.4f}")
    print("-" * 80)
    print("Summary Statistics:")
    print(f"C-index: {cindex_mean:.4f} ± {cindex_std:.4f}")
    print(f"AUC: {auc_mean:.4f} ± {auc_std:.4f}")

    df = pd.DataFrame({'os_time': os_time, 'os_status': os_status, 'risk_pred': risk_pred})
    df.to_csv(os.path.join(opt.results, "out_pred_5fold.csv"), index=False, header=True)

    results_df = pd.DataFrame({
        'fold': range(len(cindex_values)),
        'cindex': [round(x, 4) for x in cindex_values],
        'auc': [round(x, 4) for x in auc_values]
    })
    results_df.to_csv(os.path.join(opt.results, "detailed_results.csv"), index=False)

    final_stats = {
        'cindex_mean': round(cindex_mean, 4),
        'cindex_std': round(cindex_std, 4),
        'auc_mean': round(auc_mean, 4),
        'auc_std': round(auc_std, 4),
        'cindex_values': [round(x, 4) for x in cindex_values],
        'auc_values': [round(x, 4) for x in auc_values]
    }
    pickle.dump(final_stats, open(os.path.join(opt.results, opt.exp_name, opt.model_name, 'final_results.pkl'), 'wb'))

    print("\n" + "=" * 80)
    print("Generating KM Curve")
    print("=" * 80)

    merged_df = pd.DataFrame({
        'os_time': os_time,
        'os_status': os_status,
        'risk_pred': risk_pred
    })

    os.makedirs(opt.results, exist_ok=True)

    out_png = Path(opt.results) / "km_curve-lgg.png"
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 16})
    p = km_plot_and_save(merged_df, out_png, risk_threshold="median", title_prefix="MAF-Surv")
    print(f"KM curve saved: {out_png}  |  Log-rank p={p:.2e}")

    print(f"\nResults saved to: {os.path.join(opt.results, opt.exp_name, opt.model_name)}")


if __name__ == "__main__":
    main()