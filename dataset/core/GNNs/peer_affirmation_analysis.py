"""
邻居支持指标（Neighbor Support Index）分析

简化公式:
h_t = Σ_{j∈N1} p_j^(y_t) + 0.5 * Σ_{k∈N2} p_k^(y_t)

一跳邻居权重=1，二跳邻居权重=0.5

比较:
1. 单纯置信度选择的节点准确率
2. 邻居支持指标选择的节点准确率  
3. 分类别选择top k%节点的准确率
"""

import torch
import torch.nn.functional as F
from pathlib import Path


def load_data(dataset_name, base_path="e:/CS/research/w3/w3_code_project/w3_new_TAPE_TTT"):
    """加载数据"""
    data_path = Path(base_path) / "code" / "dataset" / dataset_name / "geometric_data_with_texts.pt"
    data = torch.load(data_path)
    
    llm_scores = data.llm_score_matrix
    y = data.y.squeeze()
    edge_index = data.edge_index
    
    return llm_scores, y, edge_index


def compute_2hop_neighbors(edge_index, num_nodes):
    """计算二跳邻居矩阵 A^2 (不含自环和一跳)"""
    row, col = edge_index[0], edge_index[1]
    
    # 构建邻接矩阵
    A = torch.zeros(num_nodes, num_nodes)
    A[row, col] = 1.0
    
    # A^2 = A @ A
    A2 = torch.mm(A, A)
    
    # 移除自环和一跳边
    A2 = A2 - torch.diag(torch.diag(A2))
    A2 = A2 * (1 - A)
    A2 = (A2 > 0).float()
    
    return A, A2


def compute_neighbor_support(P, edge_index, num_nodes):
    """
    计算邻居支持指标（简化版）
    
    h_t = Σ_{j∈N1} p_j^(y_t) + 0.5 * Σ_{k∈N2} p_k^(y_t)
    
    一跳权重=1, 二跳权重=0.5
    """
    # 获取每个节点的预测标签
    pred_labels = P.argmax(dim=1)  # [N]
    
    # 计算邻居矩阵
    A, A2 = compute_2hop_neighbors(edge_index, num_nodes)
    
    # 对每个节点t，取P[:, y_t]，然后与邻接矩阵相乘
    P_for_labels = P[:, pred_labels].T  # [N, N]
    
    # 一跳邻居贡献 (权重=1)
    one_hop_contrib = (A * P_for_labels).sum(dim=1)
    
    # 二跳邻居贡献 (权重=0.5)
    two_hop_contrib = (A2 * P_for_labels).sum(dim=1)
    
    # 邻居支持指标
    h = one_hop_contrib + 0.5 * two_hop_contrib
    
    return h


def compute_local_diversity(P, edge_index, num_nodes):
    """
    计算局部多样性指标 D_local (基于软标签)
    
    P_c(u) = (1/|N(u)|) * Σ_{v∈N(u)} P[v, c]   (邻居对类别c的平均软标签)
    D_local(u) = -Σ_c P_c(u) * log(P_c(u))     (熵)
    
    D_local越低 -> 邻居越一致 -> 节点越可靠
    选最小的节点
    """
    row, col = edge_index[0], edge_index[1]
    num_classes = P.size(1)
    
    # 构建邻接矩阵
    A = torch.zeros(num_nodes, num_nodes)
    A[row, col] = 1.0
    
    # 添加自环
    A = A + torch.eye(num_nodes)
    
    # 计算邻居度数
    deg = A.sum(dim=1, keepdim=True).clamp_min(1)  # [N, 1]
    
    # P_c(u) = (A @ P) / deg  -> 每个节点邻居对各类别的平均软标签
    P_neighbor = torch.mm(A, P) / deg  # [N, C]
    
    # 计算熵 D_local(u) = -Σ_c P_c(u) * log(P_c(u))
    eps = 1e-12
    P_neighbor = P_neighbor.clamp_min(eps)
    D_local = -(P_neighbor * P_neighbor.log()).sum(dim=1)  # [N]
    
    return D_local


def compute_cbc_score(P, y_pred, edge_index, num_nodes):
    """
    计算类条件介数中心性 (Class-conditional Betweenness Centrality)
    
    CBC越小表示节点越"简单"（不作为跨类别桥节点）
    选最小的节点应该更可靠
    """
    import numpy as np
    
    num_classes = int(y_pred.max().item()) + 1
    
    # 计算转移概率矩阵 (使用邻接矩阵的行归一化)
    row, col = edge_index[0], edge_index[1]
    A = torch.zeros(num_nodes, num_nodes)
    A[row, col] = 1.0
    
    # 添加自环
    A = A + torch.eye(num_nodes)
    
    # 行归一化得到转移概率
    deg = A.sum(dim=1, keepdim=True).clamp_min(1e-12)
    Pi = A / deg  # 转移概率矩阵
    
    CBC_matrix = []
    
    for c in range(num_classes):
        # 属于该类的节点索引
        iter_c_idx = torch.nonzero(y_pred == c).reshape(-1)
        # 不属于该类的节点索引
        iter_o_idx = torch.nonzero(y_pred != c).reshape(-1)
        
        if len(iter_c_idx) == 0 or len(iter_o_idx) == 0:
            CBC_matrix.append(np.zeros(num_nodes))
            continue
        
        # denominator = π(u,v): 该类到其他类的转移概率
        denominator = Pi[iter_c_idx][:, iter_o_idx]
        
        i_CBC = []
        for i in range(num_nodes):
            # numerator = π(u,i) * π(i,v)
            numerator = Pi[iter_c_idx, i].reshape(-1, 1) * Pi[i, iter_o_idx].reshape(1, -1)
            # CBC_i = sum π(u,i)π(i,v)/π(u,v)
            i_CBC.append((numerator / (denominator + 1e-16)).sum().item())
        
        CBC_matrix.append(np.array(i_CBC) / max(len(iter_c_idx), 1))
    
    # 转换为tensor
    CBC_value = torch.FloatTensor(np.array(CBC_matrix)).t()  # [N, C]
    
    # 归一化
    cbc_min = CBC_value.min(0).values
    cbc_max = CBC_value.max(0).values
    cbc_range = (cbc_max - cbc_min).clamp_min(1e-12)
    CBC_value = (CBC_value - cbc_min) / cbc_range
    
    # 求和得到最终CBC得分
    CBC_score = CBC_value.sum(1)
    
    return CBC_score


def analyze_neighbor_support(dataset_name):
    """分析邻居支持指标与单纯置信度的对比"""
    print(f"\n{'='*80}")
    print(f"Neighbor Support Index Analysis: {dataset_name}")
    print(f"{'='*80}")
    
    # 加载数据
    llm_scores, y, edge_index = load_data(dataset_name)
    num_nodes = llm_scores.size(0)
    num_classes = llm_scores.size(1)
    
    # 概率分布
    P = llm_scores
    
    # 初始准确率
    init_preds = P.argmax(dim=1)
    init_acc = (init_preds == y).float().mean().item()
    print(f"\n📊 Overall LLM Accuracy: {init_acc:.4f} ({init_acc*100:.2f}%)")
    
    # 1. 单纯置信度
    confidence = P.max(dim=1)[0]
    
    # 2. 邻居支持指标
    print(f"⏳ Computing Neighbor Support Index...")
    h = compute_neighbor_support(P, edge_index, num_nodes)
    
    # ==================== 全局选择 ====================
    print(f"\n{'='*80}")
    print(f"Part 1: Global Top K% Selection")
    print(f"{'='*80}")
    print(f"{'Top%':<8} {'#Nodes':<10} {'Conf Acc':<12} {'NSI Acc':<12} {'Diff':<10}")
    print("-" * 60)
    
    global_results = {}
    for top_percent in [0.05, 0.10, 0.15]:
        k = max(1, int(num_nodes * top_percent))
        
        # 置信度选择
        _, conf_indices = torch.topk(confidence, k)
        conf_acc = (init_preds[conf_indices] == y[conf_indices]).float().mean().item()
        
        # 邻居支持指标选择
        _, nsi_indices = torch.topk(h, k)
        nsi_acc = (init_preds[nsi_indices] == y[nsi_indices]).float().mean().item()
        
        diff = nsi_acc - conf_acc
        print(f"{int(top_percent*100):2d}%      {k:<10} {conf_acc:.4f}       {nsi_acc:.4f}       {diff:+.4f}")
        
        global_results[top_percent] = {'conf': conf_acc, 'nsi': nsi_acc, 'diff': diff}
    
    # ==================== 分类别选择 ====================
    print(f"\n{'='*80}")
    print(f"Part 2: Per-Class Top K% Selection")
    print(f"{'='*80}")
    print(f"{'Top%':<8} {'Conf Acc':<12} {'NSI Acc':<12} {'Diff':<10}")
    print("-" * 60)
    
    class_results = {}
    for top_percent in [0.05, 0.10, 0.15]:
        conf_correct = 0
        conf_total = 0
        nsi_correct = 0
        nsi_total = 0
        
        for c in range(num_classes):
            # 找到预测为类别c的节点
            class_mask = (init_preds == c)
            class_indices = class_mask.nonzero(as_tuple=True)[0]
            
            if len(class_indices) == 0:
                continue
            
            k = max(1, int(len(class_indices) * top_percent))
            
            # 置信度选择
            class_conf = confidence[class_indices]
            _, top_conf_idx = torch.topk(class_conf, k)
            selected_conf = class_indices[top_conf_idx]
            conf_correct += (init_preds[selected_conf] == y[selected_conf]).sum().item()
            conf_total += k
            
            # 邻居支持指标选择
            class_h = h[class_indices]
            _, top_nsi_idx = torch.topk(class_h, k)
            selected_nsi = class_indices[top_nsi_idx]
            nsi_correct += (init_preds[selected_nsi] == y[selected_nsi]).sum().item()
            nsi_total += k
        
        conf_acc = conf_correct / max(conf_total, 1)
        nsi_acc = nsi_correct / max(nsi_total, 1)
        diff = nsi_acc - conf_acc
        
        print(f"{int(top_percent*100):2d}%      {conf_acc:.4f}       {nsi_acc:.4f}       {diff:+.4f}")
        
        class_results[top_percent] = {'conf': conf_acc, 'nsi': nsi_acc, 'diff': diff}
    
    # ==================== CBC指标（选最小的）====================
    print(f"\n{'='*80}")
    print(f"Part 3: CBC Score - Select LOWEST K% (simple nodes)")
    print(f"{'='*80}")
    print(f"⏳ Computing CBC Score...")
    cbc_score = compute_cbc_score(P, init_preds, edge_index, num_nodes)
    
    print(f"{'Top%':<8} {'#Nodes':<10} {'Conf Acc':<12} {'CBC Low Acc':<12} {'Diff':<10}")
    print("-" * 70)
    
    cbc_results = {}
    for top_percent in [0.05, 0.10, 0.15]:
        k = max(1, int(num_nodes * top_percent))
        
        # 置信度选择（最高）
        _, conf_indices = torch.topk(confidence, k)
        conf_acc = (init_preds[conf_indices] == y[conf_indices]).float().mean().item()
        
        # CBC选择（最低 - 使用负值topk或bottomk）
        _, cbc_indices = torch.topk(cbc_score, k, largest=False)  # 选最小的
        cbc_acc = (init_preds[cbc_indices] == y[cbc_indices]).float().mean().item()
        
        diff = cbc_acc - conf_acc
        print(f"{int(top_percent*100):2d}%      {k:<10} {conf_acc:.4f}       {cbc_acc:.4f}       {diff:+.4f}")
        
        cbc_results[top_percent] = {'conf': conf_acc, 'cbc': cbc_acc, 'diff': diff}
    
    # ==================== D_local指标（选最小的）====================
    print(f"\n{'='*80}")
    print(f"Part 4: D_local (Soft Label Neighbor Entropy) - Select LOWEST K%")
    print(f"{'='*80}")
    print(f"⏳ Computing D_local...")
    d_local = compute_local_diversity(P, edge_index, num_nodes)
    
    print(f"{'Top%':<8} {'#Nodes':<10} {'Conf Acc':<12} {'D_local Low':<12} {'Diff':<10}")
    print("-" * 70)
    
    dlocal_results = {}
    for top_percent in [0.05, 0.10, 0.15]:
        k = max(1, int(num_nodes * top_percent))
        
        # 置信度选择（最高）
        _, conf_indices = torch.topk(confidence, k)
        conf_acc = (init_preds[conf_indices] == y[conf_indices]).float().mean().item()
        
        # D_local选择（最低 - 邻居熵最小的节点）
        _, dlocal_indices = torch.topk(d_local, k, largest=False)  # 选最小的
        dlocal_acc = (init_preds[dlocal_indices] == y[dlocal_indices]).float().mean().item()
        
        diff = dlocal_acc - conf_acc
        print(f"{int(top_percent*100):2d}%      {k:<10} {conf_acc:.4f}       {dlocal_acc:.4f}       {diff:+.4f}")
        
        dlocal_results[top_percent] = {'conf': conf_acc, 'dlocal': dlocal_acc, 'diff': diff}
    
    return init_acc, global_results, class_results, cbc_results, dlocal_results


def main():
    """主函数"""
    print("=" * 80)
    print("Node Selection: Multiple Metrics Comparison")
    print("=" * 80)
    
    datasets = ["cora", "citeseer", "pubmed", "arxiv", "wikics", "elephoto", "bookchild", "bookhis"]
    
    all_results = {}
    for dataset in datasets:
        try:
            init_acc, global_res, class_res, cbc_res, dlocal_res = analyze_neighbor_support(dataset)
            all_results[dataset] = {
                'init_acc': init_acc,
                'global': global_res,
                'class': class_res,
                'cbc': cbc_res,
                'dlocal': dlocal_res
            }
        except Exception as e:
            print(f"Error processing {dataset}: {e}")
            import traceback
            traceback.print_exc()
    
    # 汇总表格
    print(f"\n\n{'='*100}")
    print("Summary: Global Selection (Confidence / NSI)")
    print(f"{'='*100}")
    print(f"{'Dataset':<12} {'Init Acc':<12} {'Top 5%':<20} {'Top 10%':<20} {'Top 15%':<20}")
    print("-" * 100)
    
    for dataset, data in all_results.items():
        g = data['global']
        s5 = f"{g.get(0.05, {}).get('conf', 0):.2f}/{g.get(0.05, {}).get('nsi', 0):.2f}"
        s10 = f"{g.get(0.10, {}).get('conf', 0):.2f}/{g.get(0.10, {}).get('nsi', 0):.2f}"
        s15 = f"{g.get(0.15, {}).get('conf', 0):.2f}/{g.get(0.15, {}).get('nsi', 0):.2f}"
        print(f"{dataset:<12} {data['init_acc']:.4f}       {s5:<20} {s10:<20} {s15:<20}")
    
    print(f"\n{'='*100}")
    print("Summary: D_local (Soft Neighbor Entropy) - Select LOWEST K% (Conf / D_local)")
    print(f"{'='*100}")
    print(f"{'Dataset':<12} {'Top 5%':<20} {'Top 10%':<20} {'Top 15%':<20}")
    print("-" * 100)
    
    for dataset, data in all_results.items():
        dl = data.get('dlocal', {})
        s5 = f"{dl.get(0.05, {}).get('conf', 0):.2f}/{dl.get(0.05, {}).get('dlocal', 0):.2f}"
        s10 = f"{dl.get(0.10, {}).get('conf', 0):.2f}/{dl.get(0.10, {}).get('dlocal', 0):.2f}"
        s15 = f"{dl.get(0.15, {}).get('conf', 0):.2f}/{dl.get(0.15, {}).get('dlocal', 0):.2f}"
        print(f"{dataset:<12} {s5:<20} {s10:<20} {s15:<20}")
    
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
