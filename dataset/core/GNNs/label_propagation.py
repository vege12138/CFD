"""
标签传播分析：使用图结构传播LLM预测分布来纠正标签

算法: Y^{(t+1)} = α * S * Y^{(t)} + (1 - α) * Y^{(0)}
其中 S = D^{-1}A 是归一化邻接矩阵
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path


def load_data(dataset_name, base_path="e:/CS/research/w3/w3_code_project/w3_new_TAPE_TTT"):
    """加载数据"""
    data_path = Path(base_path) / "code" / "dataset" / dataset_name
    
    # 加载图数据 (包含llm_score_matrix)
    graph_path = data_path / "geometric_data_with_texts.pt"
    data = torch.load(graph_path)
    
    y = data.y.squeeze()
    edge_index = data.edge_index
    
    # LLM score matrix
    if hasattr(data, 'llm_score_matrix'):
        llm_scores = data.llm_score_matrix
    else:
        raise ValueError(f"No llm_score_matrix found in {graph_path}")
    
    return llm_scores, y, edge_index


def compute_normalized_adj(edge_index, num_nodes, add_self_loop=True):
    """计算归一化邻接矩阵 S = D^{-1}A"""
    row, col = edge_index[0], edge_index[1]
    
    # 添加自环
    if add_self_loop:
        self_loop = torch.arange(num_nodes)
        row = torch.cat([row, self_loop])
        col = torch.cat([col, self_loop])
    
    # 计算度
    deg = torch.zeros(num_nodes)
    deg.scatter_add_(0, row, torch.ones(row.size(0)))
    
    # D^{-1}A (行归一化)
    deg_inv = 1.0 / deg.clamp_min(1e-12)
    
    # 构建稀疏矩阵
    edge_weight = deg_inv[row]
    S = torch.sparse_coo_tensor(
        indices=torch.stack([row, col]),
        values=edge_weight,
        size=(num_nodes, num_nodes)
    ).coalesce()
    
    return S


def label_propagation(Y0, S, alpha=0.8, num_iter=20, fix_high_conf=None, conf_threshold=0.9):
    """
    标签传播算法

    Args:
        Y0: 初始标签分布 [N, C]
        S: 归一化邻接矩阵 (稀疏)
        alpha: 传播权重 (0~1), 越大越相信邻居
        num_iter: 迭代次数
        fix_high_conf: 是否固定高置信节点
        conf_threshold: 高置信阈值

    Returns:
        Y: 传播后的标签分布 [N, C]
    """
    Y = Y0.clone()

    # 如果固定高置信节点
    if fix_high_conf:
        max_conf = Y0.max(dim=1)[0]
        fixed_mask = max_conf >= conf_threshold
        print(f"   Fixed high confidence nodes: {fixed_mask.sum().item()} ({fixed_mask.float().mean()*100:.1f}%)")

    for t in range(num_iter):
        # Y^{(t+1)} = α * S * Y^{(t)} + (1 - α) * Y^{(0)}
        Y_new = alpha * torch.sparse.mm(S, Y) + (1 - alpha) * Y0

        # 固定高置信节点
        if fix_high_conf:
            Y_new[fixed_mask] = Y0[fixed_mask]

        Y = Y_new

    # 归一化
    Y = Y / Y.sum(dim=1, keepdim=True).clamp_min(1e-12)

    return Y


def compute_adj_reconstruction(Y, prototypes, edge_index, num_nodes):
    """
    计算邻接矩阵重构相似度:
    1. H = Y @ prototypes  (N x 768)
    2. sim = H @ H.T / (||H_i|| * ||H_j||)  (N x N)
    3. 与原邻接矩阵比较
    """
    # 计算加权原型嵌入
    H = torch.mm(Y, prototypes)  # [N, 768]
    H = F.normalize(H, dim=1)  # 归一化
    
    # 计算相似度矩阵 (只在边上计算，避免N^2)
    row, col = edge_index[0], edge_index[1]
    
    # 边上的余弦相似度
    sim_edges = (H[row] * H[col]).sum(dim=1)  # [E]
    
    # 平均边相似度作为重构相似度
    recon_sim = sim_edges.mean().item()
    
    return recon_sim


def analyze_label_propagation(dataset_name, alphas=[0.6,0.7,0.8,0.9], num_iters=[3]):
    """分析标签传播效果"""
    print(f"\n{'='*60}")
    print(f"Label Propagation Analysis: {dataset_name}")
    print(f"{'='*60}")

    # 加载数据
    llm_scores, y, edge_index = load_data(dataset_name)
    num_nodes = llm_scores.size(0)

    # 初始准确率
    Y0 = llm_scores
    init_preds = Y0.argmax(dim=1)
    init_acc = (init_preds == y).float().mean().item()
    
    # 计算原始分布的行求和错误率（行求和不为1的比例）
    Y0_row_sum = Y0.sum(dim=1)
    Y0_invalid_rows = (torch.abs(Y0_row_sum - 1.0) > 1e-6).float().mean().item()
    
    print(f"\n📊 Initial LLM Accuracy: {init_acc:.4f} ({init_acc*100:.2f}%)")
    print(f"📊 Initial Row-Sum Error Rate: {Y0_invalid_rows:.4f} ({Y0_invalid_rows*100:.2f}%)")

    # 计算归一化邻接矩阵
    S = compute_normalized_adj(edge_index, num_nodes, add_self_loop=True)

    # 测试不同参数
    print(f"\n{'α':<8} {'Iters':<8} {'Accuracy':<12} {'Improvement':<12}")
    print("-" * 50)

    best_acc = init_acc
    best_params = None
    best_Y = None

    for alpha in alphas:
        for num_iter in num_iters:
            Y = label_propagation(Y0, S, alpha=alpha, num_iter=num_iter)
            preds = Y.argmax(dim=1)
            acc = (preds == y).float().mean().item()
            improvement = acc - init_acc

            print(f"{alpha:<8.2f} {num_iter:<8} {acc:.4f}       {improvement:+.4f}")

            # 选择准确率最大的配置
            if acc > best_acc:
                best_acc = acc
                best_params = (alpha, num_iter)
                best_Y = Y.clone()

    # 如果没有改进，使用原始Y0
    if best_Y is None:
        best_Y = Y0

    print(f"\n✅ Best Result:")
    if best_params:
        print(f"   α={best_params[0]}, iters={best_params[1]}")
    else:
        print(f"   No improvement over initial")
    print(f"   Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
    print(f"   Improvement: {best_acc - init_acc:+.4f} ({(best_acc - init_acc)*100:+.2f}%)")
    
    # 计算最佳Y分布的行求和错误率
    best_Y_row_sum = best_Y.sum(dim=1)
    best_Y_invalid_rows = (torch.abs(best_Y_row_sum - 1.0) > 1e-6).float().mean().item()
    print(f"   Best Y Row-Sum Error Rate: {best_Y_invalid_rows:.4f} ({best_Y_invalid_rows*100:.2f}%)")
    print(f"   (Original: {Y0_invalid_rows:.4f} -> Best: {best_Y_invalid_rows:.4f})")

    # 保存最佳Y分布矩阵
    from pathlib import Path
    save_dir = Path("e:/CS/research/w3/w3_code_project/w3_new_TAPE_TTT/code/dataset") / dataset_name
    save_path = save_dir / "lp_best_distribution.pt"
    torch.save(best_Y, save_path)
    print(f"   💾 Saved best Y to: {save_path}")

    return init_acc, best_acc, best_params, best_Y


def main():
    """主函数"""
    print("=" * 60)
    print("Label Propagation for LLM Prediction Correction")
    print("=" * 60)

    datasets = ["cora", "citeseer", "pubmed", "wikics", "elephoto", "bookchild", "bookhis"]

    results = {}
    for dataset in datasets:
        try:
            init_acc, best_acc, best_params, best_Y = analyze_label_propagation(dataset)
            results[dataset] = {
                'init_acc': init_acc,
                'lp_acc': best_acc,
                'improvement': best_acc - init_acc,
                'params': best_params
            }
        except Exception as e:
            print(f"Error processing {dataset}: {e}")

    # 汇总表格
    print(f"\n\n{'='*80}")
    print("Summary Table (Selected by Max Accuracy)")
    print(f"{'='*80}")
    print(f"{'Dataset':<12} {'Init Acc':<12} {'LP Acc':<12} {'Improvement':<14} {'Best Params'}")
    print("-" * 80)

    for dataset, r in results.items():
        if r['params']:
            params_str = f"α={r['params'][0]}, iter={r['params'][1]}"
        else:
            params_str = "No improvement"
        print(f"{dataset:<12} {r['init_acc']:.4f}       {r['lp_acc']:.4f}       {r['improvement']:+.4f}         {params_str}")

    print("-" * 80)

    # 平均值
    if results:
        avg_init = sum(r['init_acc'] for r in results.values()) / len(results)
        avg_lp = sum(r['lp_acc'] for r in results.values()) / len(results)
        avg_imp = sum(r['improvement'] for r in results.values()) / len(results)
        print(f"{'Average':<12} {avg_init:.4f}       {avg_lp:.4f}       {avg_imp:+.4f}")

    print(f"{'='*80}")


if __name__ == "__main__":
    main()
