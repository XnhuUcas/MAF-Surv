import copy
import os
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from train import extract_tsne_features, test, train
from utils import (
    _to_numeric_1d,
    km_plot_and_save,
    parse_args,
    save_tsne_plot,
    split_data_cv,
)


def get_dataset_display_name(dataroot):
    dataroot_path = Path(dataroot)
    current_name = dataroot_path.name.upper()
    if current_name in {"SIS200", "SIS80", "RF80", "RF200", "TOP80", "TOP200"} and dataroot_path.parent.name:
        return dataroot_path.parent.name.upper()
    return current_name


def generate_fold0_tsne(opt, full_data, device, result_dir, model):
    # 使用第一折训练好的模型直接在完整数据集上做 t-SNE 可视化
    tsne_features = extract_tsne_features(model, full_data, device)
    dataset_name = get_dataset_display_name(opt.dataroot)
    tsne_plot_specs = [
        ('original', f'Original Features ({dataset_name})', result_dir / f'tsne_original_features_{dataset_name.lower()}.png'),
        ('specific', f'Modality Specific Features ({dataset_name})', result_dir / f'tsne_modality_specific_features_{dataset_name.lower()}.png'),
        ('aligned', f'MAF-Surv Aligned Features ({dataset_name})', result_dir / f'tsne_maf_surv_aligned_features_{dataset_name.lower()}.png'),
    ]
    for stage_name, title, save_path in tsne_plot_specs:
        save_tsne_plot(tsne_features[stage_name], title, save_path)


def run_single_anchor(opt, full_data, data_cv_splits, device, anchor_modality):
    opt = copy.deepcopy(opt)
    opt.anchor_modality = anchor_modality
    total_start_time = time.time()

    # 将锚点信息写入输出目录，避免不同实验结果互相覆盖
    run_model_name = f"{opt.model_name}_anchor-{opt.anchor_modality}"
    opt.model_name = run_model_name

    print("\n" + "#" * 80)
    print(f"RUNNING ANCHOR: {anchor_modality}")
    print("#" * 80)
    print(f"Using device: {device}")
    print(f"GPU count: {torch.cuda.device_count()}")

    os.makedirs(os.path.join(opt.model_save, opt.exp_name, run_model_name), exist_ok=True)

    average_results = {
        'cindex': [],
        'auc': [],
        'fold_runtime_seconds': [],
    }

    os_time, os_status, risk_pred = [], [], []
    fold0_model = None

    for k, data in data_cv_splits.items():
        fold_start_time = time.time()
        print("*******************************************")
        print(f"************** SPLIT ({k}/{len(data_cv_splits.items()) - 1}) **************")
        print("*******************************************")

        fold_dir = os.path.join(opt.results, opt.exp_name, run_model_name, f'{k}_fold')
        os.makedirs(fold_dir, exist_ok=True)

        model, optimizer, metric_logger, _ = train(opt, data, device, k)
        if k == 0:
            # 保留第一折模型，后续直接在完整数据集上提特征画 t-SNE
            fold0_model = copy.deepcopy(model)

        _, cindex_train, pvalue_train, _, auc_train, pred_train = test(opt, model, data, 'train', device)
        _, cindex_test, pvalue_test, _, auc_test, pred_test = test(opt, model, data, 'test', device)

        print(f"[Final] Training set: C-Index: {cindex_train:.10f}, P-Value: {pvalue_train:.10e}, AUC: {auc_train:.4f}")
        print(f"[Final] Testing set: C-Index: {cindex_test:.10f}, P-Value: {pvalue_test:.10e}, AUC: {auc_test:.4f}")

        average_results['cindex'].append(cindex_test)
        average_results['auc'].append(auc_test)

        torch.save(
            {
                'split': k,
                'opt': opt,
                'epoch': opt.niter + opt.niter_decay,
                'data': data,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metric_logger,
            },
            os.path.join(opt.model_save, opt.exp_name, run_model_name, f'{run_model_name}_{k}.pt'),
        )

        print()

        pickle.dump(pred_train, open(os.path.join(fold_dir, f'{run_model_name}_{k}pred_train.pkl'), 'wb'))
        pickle.dump(pred_test, open(os.path.join(fold_dir, f'{run_model_name}_{k}pred_test.pkl'), 'wb'))

        pred_df = pd.DataFrame({'os_time': pred_test[1], 'os_status': pred_test[2], 'risk_pred': pred_test[0]})
        pred_df.to_csv(os.path.join(opt.results, opt.exp_name, run_model_name, f"{k}-fold_pred.csv"), index=False, header=True)

        pi_label = pred_test[0] > np.median(pred_test[0])
        np.savetxt(
            os.path.join(opt.results, opt.exp_name, run_model_name, f"{k}-fold_label_test.csv"),
            pi_label + 0,
            delimiter=",",
        )

        risk_pred.extend(pred_test[0])
        os_time.extend(pred_test[1])
        os_status.extend(pred_test[2])

        fold_elapsed_seconds = time.time() - fold_start_time
        average_results['fold_runtime_seconds'].append(fold_elapsed_seconds)
        pd.DataFrame([{
            'fold_runtime_seconds': round(fold_elapsed_seconds, 2),
            'fold_runtime_minutes': round(fold_elapsed_seconds / 60.0, 2),
        }]).to_csv(
            os.path.join(fold_dir, f'{run_model_name}_{k}_runtime.csv'),
            index=False,
        )

    cindex_values = average_results['cindex']
    auc_values = average_results['auc']

    cindex_mean = np.mean(cindex_values)
    cindex_std = np.std(cindex_values, ddof=1)
    auc_mean = np.mean(auc_values)
    auc_std = np.std(auc_values, ddof=1)
    fold_runtime_seconds = average_results['fold_runtime_seconds']
    fold_runtime_minutes = [x / 60.0 for x in fold_runtime_seconds]

    result_dir = Path(opt.results) / opt.exp_name / run_model_name
    result_dir.mkdir(parents=True, exist_ok=True)

    total_elapsed_seconds = time.time() - total_start_time

    if fold0_model is not None:
        generate_fold0_tsne(opt, full_data, device, result_dir, fold0_model)

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print("Per-fold Results:")
    for i, (cindex, auc, runtime_sec) in enumerate(zip(cindex_values, auc_values, fold_runtime_seconds)):
        print(
            f"  Fold {i + 1}: C-index = {cindex:.4f}, "
            f"AUC = {auc:.4f}, "
            f"runtime = {runtime_sec:.2f}s ({runtime_sec / 60.0:.2f} min)"
        )
    print("-" * 80)
    print("Summary Statistics:")
    print(f"C-index: {cindex_mean:.4f} ± {cindex_std:.4f}")
    print(f"AUC: {auc_mean:.4f} ± {auc_std:.4f}")

    merged_df = pd.DataFrame({
        'os_time': _to_numeric_1d(os_time),
        'os_status': _to_numeric_1d(os_status),
        'risk_pred': _to_numeric_1d(risk_pred),
    })
    merged_df.to_csv(result_dir / "out_pred_5fold.csv", index=False, header=True)

    results_df = pd.DataFrame({
        'fold': range(len(cindex_values)),
        'cindex': [round(x, 4) for x in cindex_values],
        'auc': [round(x, 4) for x in auc_values],
        'fold_runtime_seconds': [round(x, 2) for x in fold_runtime_seconds],
        'fold_runtime_minutes': [round(x / 60.0, 2) for x in fold_runtime_seconds],
    })
    results_df['total_runtime_seconds'] = round(total_elapsed_seconds, 2)
    results_df['total_runtime_minutes'] = round(total_elapsed_seconds / 60.0, 2)
    results_df.to_csv(result_dir / "detailed_results.csv", index=False)

    summary_metric_rows = [
        {
            'metric': 'cindex',
            'mean': round(float(cindex_mean), 6),
            'std': round(float(cindex_std), 6),
        },
        {
            'metric': 'auc',
            'mean': round(float(auc_mean), 6),
            'std': round(float(auc_std), 6),
        },
        {
            'metric': 'fold_runtime_seconds',
            'mean': round(float(np.mean(fold_runtime_seconds)), 6),
            'std': round(float(np.std(fold_runtime_seconds, ddof=1)), 6),
        },
        {
            'metric': 'fold_runtime_minutes',
            'mean': round(float(np.mean(fold_runtime_minutes)), 6),
            'std': round(float(np.std(fold_runtime_minutes, ddof=1)), 6),
        },
    ]
    pd.DataFrame(summary_metric_rows).to_csv(result_dir / "summary_metrics.csv", index=False)

    final_stats = {
        'cindex_mean': round(cindex_mean, 4),
        'cindex_std': round(cindex_std, 4),
        'auc_mean': round(auc_mean, 4),
        'auc_std': round(auc_std, 4),
        'cindex_values': [round(x, 4) for x in cindex_values],
        'auc_values': [round(x, 4) for x in auc_values],
        'total_runtime_seconds': round(total_elapsed_seconds, 2),
        'total_runtime_minutes': round(total_elapsed_seconds / 60.0, 2),
    }
    pickle.dump(final_stats, open(result_dir / 'final_results.pkl', 'wb'))

    runtime_df = pd.DataFrame([{
        'total_runtime_seconds': round(total_elapsed_seconds, 2),
        'total_runtime_minutes': round(total_elapsed_seconds / 60.0, 2),
    }])
    runtime_df.to_csv(result_dir / 'runtime_summary.csv', index=False)

    print("\n" + "=" * 80)
    print("Generating KM Curve")
    print("=" * 80)

    out_png = result_dir / f"km_curve_{opt.anchor_modality}.png"
    plt.rcParams.update({'font.size': 16})
    p_value = km_plot_and_save(merged_df, out_png, risk_threshold="median", title_prefix="MAF-Surv")
    print(f"KM curve saved: {out_png}  |  Log-rank p={p_value:.2e}")
    print(f"Total runtime: {total_elapsed_seconds:.2f} seconds ({total_elapsed_seconds / 60.0:.2f} minutes)")
    print(f"\nResults saved to: {result_dir}")


def main():
    opt = parse_args()
    device = torch.device(f'cuda:{opt.gpu_ids[0]}') if opt.gpu_ids else torch.device('cpu')

    data_cv_path = os.path.join(opt.dataroot, opt.datatype)
    print(f"Loading {data_cv_path}")
    data_cv = pickle.load(open(data_cv_path, 'rb'))['datasets']
    data_cv_splits = split_data_cv(data_cv)

    # 默认依次运行三个 anchor；若手动指定则按指定顺序执行
    valid_anchors = {'gene', 'path', 'cna'}
    anchor_list = [anchor.strip() for anchor in opt.anchors_to_run.split(',') if anchor.strip()]
    anchor_list = [anchor for anchor in anchor_list if anchor in valid_anchors]
    if not anchor_list:
        anchor_list = [opt.anchor_modality]

    for anchor_modality in anchor_list:
        run_single_anchor(opt, data_cv, data_cv_splits, device, anchor_modality)


if __name__ == "__main__":
    main()
