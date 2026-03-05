"""
LLM Score Matrix Top-K 准确率分析

分析 data.llm_score_matrix 中：
1. Top-1 准确率: 真实标签等于最高分类别
2. Top-2 准确率: 真实标签在前两高分类别中
3. Top-3 准确率: 真实标签在前三高分类别中

同时对比 lp_best_distribution.pt 的准确率
"""

import torch
from pathlib import Path
import os


def load_data(dataset_name, base_path="e:/CS/research/w3/w3_code_project/w3_new_TAPE_TTT"):
    """加载数据"""
    data_path = Path(base_path) / "code" / "dataset" / dataset_name / "geometric_data_with_texts.pt"
    data = torch.load(data_path)
    
    llm_scores = data.llm_score_matrix
    y = data.y.squeeze()
    
    return llm_scores, y


def load_lp_distribution(dataset_name, base_path="e:/CS/research/w3/w3_code_project/w3_new_TAPE_TTT"):
    """加载LP最优分布"""
    lp_path = Path(base_path) / "code" / "dataset" / dataset_name / "lp_best_distribution.pt"
    if lp_path.exists():
        lp_dist = torch.load(lp_path)
        return lp_dist
    return None


def compute_topk_accuracy(scores, y):
    """计算Top-K准确率"""
    num_nodes = scores.size(0)
    
    # 获取排序后的类别索引（从高到低）
    sorted_indices = scores.argsort(dim=1, descending=True)  # [N, C]
    
    # Top-1
    top1_pred = sorted_indices[:, 0]
    top1_correct = (top1_pred == y).sum().item()
    top1_acc = top1_correct / num_nodes
    
    # Top-2
    top2_preds = sorted_indices[:, :2]
    top2_correct = (top2_preds == y.unsqueeze(1)).any(dim=1).sum().item()
    top2_acc = top2_correct / num_nodes
    
    # Top-3
    top3_preds = sorted_indices[:, :3]
    top3_correct = (top3_preds == y.unsqueeze(1)).any(dim=1).sum().item()
    top3_acc = top3_correct / num_nodes
    
    # Top-2 随机选择：从top-2中随机选一个标签
    random_idx_2 = torch.randint(0, 2, (num_nodes,))  # 每个节点随机选0或1
    top2_random_pred = top2_preds[torch.arange(num_nodes), random_idx_2]
    top2_random_correct = (top2_random_pred == y).sum().item()
    top2_random_acc = top2_random_correct / num_nodes
    
    # Top-3 随机选择：从top-3中随机选一个标签
    random_idx_3 = torch.randint(0, 3, (num_nodes,))  # 每个节点随机选0,1,2
    top3_random_pred = top3_preds[torch.arange(num_nodes), random_idx_3]
    top3_random_correct = (top3_random_pred == y).sum().item()
    top3_random_acc = top3_random_correct / num_nodes
    
    return top1_acc, top2_acc, top3_acc, top2_random_acc, top3_random_acc


def analyze_topk_accuracy(dataset_name):
    """分析LLM分数矩阵的Top-K准确率"""
    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*70}")
    
    # 加载数据
    llm_scores, y = load_data(dataset_name)
    num_nodes = llm_scores.size(0)
    num_classes = llm_scores.size(1)
    
    print(f"Nodes: {num_nodes}, Classes: {num_classes}")
    
    # 原始LLM分数矩阵的Top-K准确率
    llm_top1, llm_top2, llm_top3, llm_top2_rand, llm_top3_rand = compute_topk_accuracy(llm_scores, y)
    
    print(f"\n📊 原始LLM Score Matrix Top-K准确率:")
    print(f"  Top-1: {llm_top1:.4f} ({llm_top1*100:.2f}%)")
    print(f"  Top-2 (包含): {llm_top2:.4f} ({llm_top2*100:.2f}%)")
    print(f"  Top-3 (包含): {llm_top3:.4f} ({llm_top3*100:.2f}%)")
    print(f"  Top-2 随机选: {llm_top2_rand:.4f} ({llm_top2_rand*100:.2f}%)")
    print(f"  Top-3 随机选: {llm_top3_rand:.4f} ({llm_top3_rand*100:.2f}%)")
    
    # 加载LP分布
    lp_dist = load_lp_distribution(dataset_name)
    lp_top1, lp_top2, lp_top3, lp_top2_rand, lp_top3_rand = None, None, None, None, None
    
    if lp_dist is not None:
        lp_top1, lp_top2, lp_top3, lp_top2_rand, lp_top3_rand = compute_topk_accuracy(lp_dist, y)
        
        print(f"\n📊 LP Best Distribution Top-K准确率:")
        print(f"  Top-1: {lp_top1:.4f} ({lp_top1*100:.2f}%)")
        print(f"  Top-2 (包含): {lp_top2:.4f} ({lp_top2*100:.2f}%)")
        print(f"  Top-3 (包含): {lp_top3:.4f} ({lp_top3*100:.2f}%)")
        print(f"  Top-2 随机选: {lp_top2_rand:.4f} ({lp_top2_rand*100:.2f}%)")
        print(f"  Top-3 随机选: {lp_top3_rand:.4f} ({lp_top3_rand*100:.2f}%)")
        
        print(f"\n📊 对比 (LP - LLM):")
        print(f"  Top-1: {lp_top1 - llm_top1:+.4f}")
        print(f"  Top-2 随机选: {lp_top2_rand - llm_top2_rand:+.4f}")
        print(f"  Top-3 随机选: {lp_top3_rand - llm_top3_rand:+.4f}")
    else:
        print(f"\n⚠️ LP分布文件不存在: lp_best_distribution.pt")
    
    return {
        'dataset': dataset_name,
        'num_nodes': num_nodes,
        'num_classes': num_classes,
        'llm_top1': llm_top1,
        'llm_top2': llm_top2,
        'llm_top3': llm_top3,
        'llm_top2_rand': llm_top2_rand,
        'llm_top3_rand': llm_top3_rand,
        'lp_top1': lp_top1,
        'lp_top2': lp_top2,
        'lp_top3': lp_top3,
        'lp_top2_rand': lp_top2_rand,
        'lp_top3_rand': lp_top3_rand
    }


def main():
    """主函数"""
    print("=" * 70)
    print("LLM Score Matrix vs LP Distribution Top-K Accuracy Analysis")
    print("=" * 70)
    
    datasets = ["cora", "citeseer", "pubmed", "arxiv", "wikics", "elephoto", "bookchild", "bookhis"]
    
    all_results = []
    for dataset in datasets:
        try:
            result = analyze_topk_accuracy(dataset)
            all_results.append(result)
        except Exception as e:
            print(f"\nError processing {dataset}: {e}")
            import traceback
            traceback.print_exc()
    
    # 汇总表格 - 原始LLM
    print(f"\n\n{'='*90}")
    print("Summary: 原始LLM Score Matrix Top-K准确率")
    print(f"{'='*90}")
    print(f"{'Dataset':<12} {'Nodes':<10} {'Classes':<8} {'Top-1':<12} {'Top-2':<12} {'Top-3':<12}")
    print("-" * 90)
    
    for r in all_results:
        print(f"{r['dataset']:<12} {r['num_nodes']:<10} {r['num_classes']:<8} "
              f"{r['llm_top1']:.4f}       {r['llm_top2']:.4f}       {r['llm_top3']:.4f}")
    
    # 汇总表格 - LP分布
    print(f"\n{'='*90}")
    print("Summary: LP Best Distribution Top-K准确率")
    print(f"{'='*90}")
    print(f"{'Dataset':<12} {'Top-1':<12} {'Top-2':<12} {'Top-3':<12} {'Top1 Δ':<10} {'Top2 Δ':<10} {'Top3 Δ':<10}")
    print("-" * 90)
    
    for r in all_results:
        if r['lp_top1'] is not None:
            d1 = r['lp_top1'] - r['llm_top1']
            d2 = r['lp_top2'] - r['llm_top2']
            d3 = r['lp_top3'] - r['llm_top3']
            print(f"{r['dataset']:<12} {r['lp_top1']:.4f}       {r['lp_top2']:.4f}       {r['lp_top3']:.4f}       "
                  f"{d1:+.4f}     {d2:+.4f}     {d3:+.4f}")
        else:
            print(f"{r['dataset']:<12} N/A")
    
    print("=" * 90)


if __name__ == "__main__":
    main()
