"""
熵值分析：分析平方后目标分布的熵值与准确率的关系

对于平方后的目标分布P²，按熵值排序，分析不同百分位节点的准确率。
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path


def load_data(dataset_name, base_path="e:/CS/research/w3/w3_code_project/w3_new_TAPE_TTT"):
    """加载数据"""
    data_path = Path(base_path) / "data" / dataset_name
    
    # 加载LLM score matrix
    llm_score_path = data_path / "llm_score_matrix.pt"
    llm_scores = torch.load(llm_score_path)
    
    # 加载标签
    label_path = data_path / f"{dataset_name}_fixed_sbert.pt"
    data = torch.load(label_path)
    y = data.y.squeeze()
    
    return llm_scores, y


def compute_squared_distribution(scores, tau=0.1):
    """计算平方后的目标分布"""
    # 先softmax
    P = F.softmax(scores / tau, dim=1)
    # 平方并归一化
    P_squared = P ** 2
    P_squared = P_squared / P_squared.sum(dim=1, keepdim=True)
    return P_squared


def compute_entropy(P):
    """计算每个节点的熵"""
    eps = 1e-12
    P_clamped = P.clamp_min(eps)
    entropy = -(P_clamped * P_clamped.log()).sum(dim=1)
    return entropy


def analyze_entropy_accuracy(P, y, percentiles=[5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]):
    """
    按熵值百分位分析准确率
    
    Args:
        P: 目标分布 [N, C]
        y: 真实标签 [N]
        percentiles: 要分析的百分位列表
    
    Returns:
        results: dict, 每个百分位的准确率
    """
    N = P.size(0)
    entropy = compute_entropy(P)
    
    # 按熵值排序 (从大到小)
    sorted_indices = torch.argsort(entropy, descending=True)
    
    # 预测标签
    preds = P.argmax(dim=1)
    correct = (preds == y)
    
    results = {}
    prev_k = 0
    
    for pct in percentiles:
        k = int(N * pct / 100)
        if k == 0:
            continue
            
        # 累计: 熵最大的前pct%
        top_indices = sorted_indices[:k]
        acc_cumulative = correct[top_indices].float().mean().item()
        
        # 区间: 从prev_k到k
        if k > prev_k:
            interval_indices = sorted_indices[prev_k:k]
            acc_interval = correct[interval_indices].float().mean().item()
        else:
            acc_interval = 0
        
        results[pct] = {
            'cumulative_acc': acc_cumulative,
            'interval_acc': acc_interval,
            'num_nodes': k,
            'interval_nodes': k - prev_k
        }
        
        prev_k = k
    
    return results, entropy


def run_analysis(dataset_name):
    """运行单个数据集的分析"""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")
    
    # 加载数据
    llm_scores, y = load_data(dataset_name)
    print(f"Nodes: {llm_scores.size(0)}, Classes: {llm_scores.size(1)}")
    
    # 计算平方后的目标分布
    P_squared = compute_squared_distribution(llm_scores)
    
    # 分析
    percentiles = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    results, entropy = analyze_entropy_accuracy(P_squared, y, percentiles)
    
    # 打印结果
    print(f"\n{'Percentile':<12} {'Nodes':<8} {'Cumul Acc':<12} {'Interval':<10} {'Interval Acc':<12}")
    print("-" * 60)
    
    for pct in percentiles:
        if pct in results:
            r = results[pct]
            print(f"Top {pct:3d}%     {r['num_nodes']:<8} {r['cumulative_acc']:.4f}       "
                  f"{r['interval_nodes']:<10} {r['interval_acc']:.4f}")
    
    # 熵值统计
    print(f"\nEntropy Statistics:")
    print(f"  Min: {entropy.min().item():.4f}")
    print(f"  Max: {entropy.max().item():.4f}")
    print(f"  Mean: {entropy.mean().item():.4f}")
    print(f"  Std: {entropy.std().item():.4f}")
    
    return results


def main():
    """主函数"""
    print("=" * 60)
    print("Entropy-Accuracy Analysis for Squared Target Distribution")
    print("=" * 60)
    
    datasets = ["cora", "citeseer", "pubmed"]
    
    all_results = {}
    for dataset in datasets:
        try:
            results = run_analysis(dataset)
            all_results[dataset] = results
        except Exception as e:
            print(f"Error processing {dataset}: {e}")
    
    # 汇总表格
    print(f"\n\n{'='*80}")
    print("Summary: Cumulative Accuracy by Entropy Percentile")
    print(f"{'='*80}")
    print(f"{'Dataset':<12}", end="")
    for pct in [5, 10, 20, 50, 90, 100]:
        print(f"Top{pct}%".ljust(10), end="")
    print()
    print("-" * 80)
    
    for dataset, results in all_results.items():
        print(f"{dataset:<12}", end="")
        for pct in [5, 10, 20, 50, 90, 100]:
            if pct in results:
                print(f"{results[pct]['cumulative_acc']:.4f}".ljust(10), end="")
            else:
                print("N/A".ljust(10), end="")
        print()


if __name__ == "__main__":
    main()
