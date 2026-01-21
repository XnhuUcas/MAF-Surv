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
        print(f"************** SPLIT ({k}/{len(data_cv_splits.items())-1}) **************")
        print("*******************************************")
        
        fold_dir = os.path.join(opt.results, opt.exp_name, opt.model_name, f'{k}_fold')
        os.makedirs(fold_dir, exist_ok=True)
        
        model, optimizer, metric_logger = train(opt, data, device, k)
        
        loss_train, cindex_train, pvalue_train, surv_acc_train, auc_train, pred_train = test(opt, model, data, 'train', device)
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
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print("Per-fold Results:")
    for i, (cindex, auc) in enumerate(zip(cindex_values, auc_values)):
        print(f"  Fold {i+1}: C-index = {cindex:.4f}, AUC = {auc:.4f}")
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
    
    print("\n" + "="*80)
    print("Generating KM Curve")
    print("="*80)
    
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