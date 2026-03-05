# =========================
#  Confidence-based Accuracy Analysis
# =========================
"""
对比原始LLM Score矩阵和处理后的P分布:
- 按置信度选取前 10%, 20%, ..., 90% 的节点
- 计算对应的准确率

用法: python confidence_analysis.py
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


def analyze_topk_accuracy(scores, labels, percentages):
    """按置信度选取top k%节点，计算准确率"""
    max_conf, preds = scores.max(dim=1)
    sorted_indices = torch.argsort(max_conf, descending=True)
    
    results = {}
    num_nodes = len(labels)
    
    for pct in percentages:
        k = max(1, int(num_nodes * pct / 100))
        top_k_indices = sorted_indices[:k]
        correct = (preds[top_k_indices] == labels[top_k_indices]).float().sum().item()
        results[pct] = (correct / k, k)
    
    return results


def analyze_topk_accuracy_per_class(scores, labels, num_classes, percentages):
    """按类别分别选取top k%节点，计算准确率"""
    max_conf, preds = scores.max(dim=1)
    
    # 每个类别的结果: {class_id: {pct: (acc, num_nodes)}}
    class_results = {}
    
    for c in range(num_classes):
        # 该类别的节点mask
        class_mask = (labels == c)
        class_indices = torch.where(class_mask)[0]
        num_class_nodes = len(class_indices)
        
        if num_class_nodes == 0:
            class_results[c] = {pct: (0.0, 0) for pct in percentages}
            continue
        
        # 该类别节点的置信度
        class_conf = max_conf[class_indices]
        class_preds = preds[class_indices]
        class_labels = labels[class_indices]
        
        # 按置信度排序
        sorted_idx = torch.argsort(class_conf, descending=True)
        
        results = {}
        for pct in percentages:
            k = max(1, int(num_class_nodes * pct / 100))
            top_k_idx = sorted_idx[:k]
            correct = (class_preds[top_k_idx] == class_labels[top_k_idx]).float().sum().item()
            results[pct] = (correct / k, k)
        
        class_results[c] = results
    
    return class_results


def run_analysis(dataset_name, data_root, device):
    """对单个数据集运行分析"""
    print(f"\n{'=' * 70}")
    print(f"📊 Dataset: {dataset_name}")
    print(f"{'=' * 70}")
    
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
    
    # 全局分析
    percentages = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    orig_results = analyze_topk_accuracy(llm_scores, labels, percentages)
    proc_results = analyze_topk_accuracy(P_distribution, labels, percentages)
    
    # 打印全局表格
    print(f"\n📈 Global Top-k% Accuracy:")
    print(f"{'-' * 70}")
    print(f"{'Top %':^8} | {'Original LLM':^20} | {'Processed P':^20} | {'Δ Acc':^10}")
    print(f"{'-' * 70}")
    
    for pct in percentages:
        orig_acc, orig_n = orig_results[pct]
        proc_acc, proc_n = proc_results[pct]
        delta = proc_acc - orig_acc
        delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        print(f"{pct:^8}% | {orig_acc:.4f} ({orig_n:5d}节点) | {proc_acc:.4f} ({proc_n:5d}节点) | {delta_str:^10}")
    
    # 全数据集
    orig_full = (llm_scores.argmax(dim=1) == labels).float().mean().item()
    proc_full = (P_distribution.argmax(dim=1) == labels).float().mean().item()
    delta_full = proc_full - orig_full
    delta_str = f"+{delta_full:.4f}" if delta_full >= 0 else f"{delta_full:.4f}"
    print(f"{'-' * 70}")
    print(f"{'100':^8}% | {orig_full:.4f} ({num_nodes:5d}节点) | {proc_full:.4f} ({num_nodes:5d}节点) | {delta_str:^10}")
    print(f"{'=' * 70}")
    
    # ============ 按类别分析 ============
    print(f"\n📊 Per-Class Top-k% Accuracy:")
    orig_class_results = analyze_topk_accuracy_per_class(llm_scores, labels, num_classes, percentages)
    proc_class_results = analyze_topk_accuracy_per_class(P_distribution, labels, num_classes, percentages)
    
    # 获取类别名称（如果有）
    label2class = getattr(data, 'label2class', None)
    
    for c in range(num_classes):
        # 获取类别名
        if label2class is not None:
            if isinstance(label2class, dict):
                class_name = label2class.get(c, [f"Class_{c}"])[0] if isinstance(label2class.get(c), list) else label2class.get(c, f"Class_{c}")
            elif isinstance(label2class, list) and c < len(label2class):
                class_name = label2class[c]
            else:
                class_name = f"Class_{c}"
        else:
            class_name = f"Class_{c}"
        
        # 类别节点数
        class_count = (labels == c).sum().item()
        if class_count == 0:
            continue
            
        print(f"\n  🏷️ Class {c}: {class_name} ({class_count} nodes)")
        print(f"  {'-' * 65}")
        print(f"  {'Top %':^6} | {'Orig Acc':^12} | {'Proc Acc':^12} | {'Δ':^8}")
        print(f"  {'-' * 65}")
        
        for pct in percentages:
            orig_acc, orig_n = orig_class_results[c][pct]
            proc_acc, proc_n = proc_class_results[c][pct]
            delta = proc_acc - orig_acc
            delta_str = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"
            print(f"  {pct:^6}% | {orig_acc:.4f} ({orig_n:3d}) | {proc_acc:.4f} ({proc_n:3d}) | {delta_str:^8}")
    
    return {
        'original': orig_results,
        'processed': proc_results,
        'original_full': orig_full,
        'processed_full': proc_full,
        'orig_class': orig_class_results,
        'proc_class': proc_class_results
    }


if __name__ == "__main__":
    # 配置
    DATASETS = ["cora", "citeseer", "pubmed", "arxiv","bookchild","bookhis","elephoto","wikics"]
    DATA_ROOT = 'dataset'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 切换到code目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    code_dir = os.path.dirname(os.path.dirname(script_dir))
    os.chdir(code_dir)
    
    print(f"🚀 Confidence-based Accuracy Analysis")
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
    print(f"\n\n{'#' * 70}")
    print(f"# Summary")
    print(f"{'#' * 70}")
    for ds in all_results:
        r = all_results[ds]
        print(f"{ds:12} | Original: {r['original_full']:.4f} → Processed: {r['processed_full']:.4f} | Δ = {r['processed_full']-r['original_full']:+.4f}")
