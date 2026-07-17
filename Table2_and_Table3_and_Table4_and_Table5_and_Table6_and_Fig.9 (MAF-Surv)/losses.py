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