# =========================
#  Step 3: Train GNN
# =========================
"""
功能:
1. 加载数据（包含所有嵌入和得分）
2. 升级LLM得分矩阵
3. GNN训练
"""
import torch
import numpy as np

from core.data_utils.load import load_data
from core.GNNs.trainer import GNNTrainer
from config import get_args

import os
import json


def apply_dataset_config(args, cfg_path: str = "configs/dataset_params.json"):
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"[apply_dataset_config] config file not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg_all = json.load(f)

    default_cfg = cfg_all.get("default", {})
    ds_cfg = cfg_all.get(args.dataset, {})

    # 合并：default -> dataset 覆盖
    merged = dict(default_cfg)
    merged.update(ds_cfg)

    # 只覆盖你要求的这些参数
    keys = [
        "warmup_epochs",
        "label_update_interval",
        "total_epochs",
        "co_train_lr",
        "co_train_tau",
        "post_warmup_lr",
        "post_warmup_tau",
    ]

    for k in keys:
        if k in merged:
            setattr(args, k, merged[k])

    return args
def run(args):
    """
    运行GNN训练
    """
    args.printer.info(f"\n{'='*60}")
    args.printer.info("Step 3: GNN Training")
    args.printer.info(f"{'='*60}")
    args.printer.info(f"   Dataset: {args.dataset}")
    args.printer.info(f"   Model: {args.gnn_model}")
    args.printer.info(f"   Runs: {args.runs}")
    args.printer.info(f"{'='*60}\n")
    # 加载数据
    data, num_classes = load_data(args.dataset, args.data_root)



    # 多次运行
    all_acc = []
    all_f1 = []
    r = args.runs
    ls = [0,1,2,3,4]
    ls = [0]
    for i in ls:
        args.seed = i

        args.printer.info(f"\n{'='*60}")
        args.printer.info(f"Run {i + 1}/{r}")
        args.printer.info(f"{'='*60}")

        # 创建训练器
        trainer = GNNTrainer(args, data, num_classes)
        result = trainer.train()

        acc = result['acc'] * 100
        macro_f1 = result['macro_f1'] * 100
        all_acc.append(acc)
        all_f1.append(macro_f1)

        # 每次运行后打印详细统计
        cur_mean_acc = float(np.mean(all_acc))
        cur_mean_f1 = float(np.mean(all_f1))
        args.printer.info(f"\n[{i+1}/{r}] run={i+1} | {args.dataset} | acc={acc:.2f}% | macro_f1={macro_f1:.2f}%")
        args.printer.info(f"  acc_list={[f'{x:.2f}' for x in all_acc]}")
        args.printer.info(f"  f1_list={[f'{x:.2f}' for x in all_f1]}")
        args.printer.info(f"  mean_acc={cur_mean_acc:.2f}%, mean_f1={cur_mean_f1:.2f}%")

    if len(all_acc) == 0:
        args.printer.info("No runs executed.")
        return

    # 统计结果
    mean_acc = np.mean(all_acc)
    std_acc = np.std(all_acc, ddof=1) if len(all_acc) > 1 else 0.0
    mean_f1 = np.mean(all_f1)
    std_f1 = np.std(all_f1, ddof=1) if len(all_f1) > 1 else 0.0

    args.printer.info(f"\n{'='*60}")
    args.printer.info("Final Summary")
    args.printer.info(f"{'='*60}")
    args.printer.info(f"   Dataset: {args.dataset}")
    args.printer.info(f"   Model: {args.gnn_model}")
    args.printer.info(f"   Runs: {len(all_acc)}")
    args.printer.info(f"   Accuracy:  {mean_acc:.2f} ± {std_acc:.2f}")
    args.printer.info(f"   Macro-F1:  {mean_f1:.2f} ± {std_f1:.2f}")
    args.printer.info(f"{'='*60}\n")

    return {
        'mean_acc': mean_acc,
        'std_acc': std_acc,
        'mean_f1': mean_f1,
        'std_f1': std_f1
    }


if __name__ == '__main__':
    args = get_args()
    args = apply_dataset_config(args, cfg_path="dataset/dataset_params.json")
    run(args)

