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
    train_loader = DataLoader(dataset=custom_data_loader, batch_size=len(custom_data_loader), num_workers=4, shuffle=False)
    
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
            
            loss_test, cindex_test, pvalue_test, surv_acc_test, auc_test, pred_test = test(opt, model, data, 'test', device)
            
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
    test_loader = DataLoader(dataset=custom_data_loader, batch_size=len(custom_data_loader), num_workers=4, shuffle=False)
    
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