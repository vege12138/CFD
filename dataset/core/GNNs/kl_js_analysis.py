# =========================
#  KL/JS Divergence Analysis
# =========================
"""
计算原始LLM Score与处理后P分布的节点级KL/JS散度:
- 按KL/JS值选取最相似的前 5%, 10%, ..., 90% 节点
- 比较准确率

用法: python kl_js_analysis.py
"""
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import add_self_loops

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def load_data(dataset, data_root='dataset'):
    """加载数据"""
    data_dir = os.path.join(data_root, dataset)
    pt_file = os.path.join(data_dir, "geometric_data_with_texts.pt")
    
    if not os.path.exists(pt_file):
        raise FileNotFoundError(f"Data file not found: {pt_file}")
    
    try:
        data = torch.load(pt_file, weights_only=False)
    except TypeError:
        data = torch.load(pt_file)
    
    if hasattr(data, 'num_classes') and data.num_classes:
        num_classes = int(data.num_classes)
    elif hasattr(data, 'y') and data.y is not None:
        num_classes = int(data.y.max().item()) + 1
    else:
        class_map = {'cora': 7, 'citeseer': 6, 'pubmed': 3, 'arxiv': 40}
        num_classes = class_map.get(dataset.lower(), 0)
    
    return data, num_classes


def compute_sym_norm_adj(edge_index, num_nodes, device):
    """计算对称归一化邻接矩阵"""
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    row, col = edge_index[0], edge_index[1]
    deg = torch.bincount(row, minlength=num_nodes).float().to(device)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    adj_norm = torch.sparse_coo_tensor(
        indices=edge_index,
        values=edge_weight,
        size=(num_nodes, num_nodes),
        device=device
    ).coalesce()
    return adj_norm


def preprocess_to_P(llm_scores, edge_index, label_prototypes, num_classes, num_nodes, device):
    """预处理LLM得分 -> P分布"""
    P = llm_scores
    proto = label_prototypes.to(device)
    H = torch.mm(P, proto)

    # 图传播 2层
    adj_norm = compute_sym_norm_adj(edge_index, num_nodes, device)
    for _ in range(2):
        H = torch.sparse.mm(adj_norm, H)

    # 计算伪类中心
    pseudo_y = llm_scores.argmax(dim=1)
    mu_init = torch.zeros(num_classes, H.size(1), device=device, dtype=H.dtype)

    for j in range(num_classes):
        mask = (pseudo_y == j)
        if mask.sum() > 0:
            mu_init[j] = H[mask].mean(dim=0)
        else:
            mu_init[j] = proto[j]

    # 计算Q分布
    Hn = F.normalize(H, dim=1)
    mun = F.normalize(mu_init, dim=1)
    sim = torch.mm(Hn, mun.T)
    q = F.softmax(sim, dim=1)
    return q


def compute_kl_per_node(p, q):
    """计算每个节点的KL散度 KL(p || q)"""
    p = p.clamp_min(1e-12)
    q = q.clamp_min(1e-12)
    kl = (p * (p.log() - q.log())).sum(dim=1)  # [N]
    return kl


def compute_js_per_node(p, q):
    """计算每个节点的JS散度"""
    p = p.clamp_min(1e-12)
    q = q.clamp_min(1e-12)
    m = 0.5 * (p + q)
    js = 0.5 * (p * (p.log() - m.log())).sum(dim=1) + 0.5 * (q * (q.log() - m.log())).sum(dim=1)
    return js


def analyze_by_divergence(scores_orig, scores_proc, labels, divergence_values, percentages, select_low=True):
    """
    按散度值选取节点，计算准确率
    
    Args:
        scores_orig: 原始分布 [N, C]
        scores_proc: 处理后分布 [N, C]
        labels: 真实标签 [N]
        divergence_values: 每个节点的散度值 [N]
        percentages: 要分析的百分比列表
        select_low: True选取散度最小的(最相似), False选取散度最大的
    
    Returns:
        dict: {pct: (orig_acc, proc_acc, num_nodes)}
    """
    preds_orig = scores_orig.argmax(dim=1)
    preds_proc = scores_proc.argmax(dim=1)
    
    # 按散度排序
    if select_low:
        sorted_indices = torch.argsort(divergence_values, descending=False)  # 从小到大
    else:
        sorted_indices = torch.argsort(divergence_values, descending=True)  # 从大到小
    
    results = {}
    num_nodes = len(labels)
    
    for pct in percentages:
        k = max(1, int(num_nodes * pct / 100))
        selected_indices = sorted_indices[:k]
        
        orig_correct = (preds_orig[selected_indices] == labels[selected_indices]).float().sum().item()
        proc_correct = (preds_proc[selected_indices] == labels[selected_indices]).float().sum().item()
        
        results[pct] = (orig_correct / k, proc_correct / k, k)
    
    return results


def run_analysis(dataset_name, data_root, device):
    """对单个数据集运行分析"""
    print(f"\n{'=' * 80}")
    print(f"📊 Dataset: {dataset_name}")
    print(f"{'=' * 80}")
    
    # 加载数据
    data, num_classes = load_data(dataset_name, data_root)
    
    llm_scores = data.llm_score_matrix.to(device)
    label_prototypes = data.label_prototypes.to(device)
    edge_index = data.edge_index.to(device)
    labels = data.y.squeeze().to(device)
    num_nodes = labels.shape[0]
    
    print(f"   Nodes: {num_nodes}, Classes: {num_classes}")
    
    # 预处理得到P分布
    P_distribution = preprocess_to_P(
        llm_scores, edge_index, label_prototypes,
        num_classes, num_nodes, device
    )
    
    # 计算节点级散度
    kl_values = compute_kl_per_node(llm_scores, P_distribution)  # KL(orig || proc)
    js_values = compute_js_per_node(llm_scores, P_distribution)
    
    print(f"   KL Divergence: mean={kl_values.mean().item():.4f}, std={kl_values.std().item():.4f}")
    print(f"   JS Divergence: mean={js_values.mean().item():.4f}, std={js_values.std().item():.4f}")
    
    percentages = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    
    # ============ 按KL值分析 (最相似) ============
    print(f"\n📈 Select by KL Divergence (LOWEST = most similar):")
    print(f"{'-' * 80}")
    print(f"{'Top %':^8} | {'Orig Acc':^12} | {'Proc Acc':^12} | {'Δ':^10} | {'Nodes':^8}")
    print(f"{'-' * 80}")
    
    kl_results_low = analyze_by_divergence(llm_scores, P_distribution, labels, kl_values, percentages, select_low=True)
    for pct in percentages:
        orig_acc, proc_acc, n = kl_results_low[pct]
        delta = proc_acc - orig_acc
        delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        print(f"{pct:^8}% | {orig_acc:.4f}       | {proc_acc:.4f}       | {delta_str:^10} | {n:^8}")
    
    # ============ 按JS值分析 (最相似) ============
    print(f"\n📈 Select by JS Divergence (LOWEST = most similar):")
    print(f"{'-' * 80}")
    print(f"{'Top %':^8} | {'Orig Acc':^12} | {'Proc Acc':^12} | {'Δ':^10} | {'Nodes':^8}")
    print(f"{'-' * 80}")
    
    js_results_low = analyze_by_divergence(llm_scores, P_distribution, labels, js_values, percentages, select_low=True)
    for pct in percentages:
        orig_acc, proc_acc, n = js_results_low[pct]
        delta = proc_acc - orig_acc
        delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        print(f"{pct:^8}% | {orig_acc:.4f}       | {proc_acc:.4f}       | {delta_str:^10} | {n:^8}")
    
    # ============ 按KL值分析 (最不相似) ============
    print(f"\n📉 Select by KL Divergence (HIGHEST = least similar):")
    print(f"{'-' * 80}")
    print(f"{'Top %':^8} | {'Orig Acc':^12} | {'Proc Acc':^12} | {'Δ':^10} | {'Nodes':^8}")
    print(f"{'-' * 80}")
    
    kl_results_high = analyze_by_divergence(llm_scores, P_distribution, labels, kl_values, percentages, select_low=False)
    for pct in percentages:
        orig_acc, proc_acc, n = kl_results_high[pct]
        delta = proc_acc - orig_acc
        delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        print(f"{pct:^8}% | {orig_acc:.4f}       | {proc_acc:.4f}       | {delta_str:^10} | {n:^8}")
    
    # ============ 按JS值分析 (最不相似) ============
    print(f"\n📉 Select by JS Divergence (HIGHEST = least similar):")
    print(f"{'-' * 80}")
    print(f"{'Top %':^8} | {'Orig Acc':^12} | {'Proc Acc':^12} | {'Δ':^10} | {'Nodes':^8}")
    print(f"{'-' * 80}")
    
    js_results_high = analyze_by_divergence(llm_scores, P_distribution, labels, js_values, percentages, select_low=False)
    for pct in percentages:
        orig_acc, proc_acc, n = js_results_high[pct]
        delta = proc_acc - orig_acc
        delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        print(f"{pct:^8}% | {orig_acc:.4f}       | {proc_acc:.4f}       | {delta_str:^10} | {n:^8}")
    
    print(f"{'=' * 80}")
    
    return {
        'kl_low': kl_results_low,
        'kl_high': kl_results_high,
        'js_low': js_results_low,
        'js_high': js_results_high,
        'kl_mean': kl_values.mean().item(),
        'js_mean': js_values.mean().item()
    }


if __name__ == "__main__":
    # 配置
    DATASETS = ["cora", "citeseer", "pubmed", "arxiv", "bookchild", "bookhis", "elephoto", "wikics"]
    DATA_ROOT = 'dataset'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 切换到code目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    code_dir = os.path.dirname(os.path.dirname(script_dir))
    os.chdir(code_dir)
    
    print(f"🚀 KL/JS Divergence Analysis")
    print(f"   Device: {DEVICE}")
    print(f"   Working Dir: {os.getcwd()}")
    
    all_results = {}
    for dataset in DATASETS:
        try:
            results = run_analysis(dataset, DATA_ROOT, DEVICE)
            all_results[dataset] = results
        except Exception as e:
            print(f"❌ Error processing {dataset}: {e}")
    
    # 汇总
    print(f"\n\n{'#' * 80}")
    print(f"# Summary: Mean Divergences")
    print(f"{'#' * 80}")
    for ds in all_results:
        r = all_results[ds]
        print(f"{ds:12} | KL mean: {r['kl_mean']:.4f} | JS mean: {r['js_mean']:.4f}")
