"""
改动：
1) 汇总表格里每个 PRxx 单元格显示：Acc*100（Δ*100）
   例：65.03(+1.20)
2) Init 同样 *100
"""

import torch
from pathlib import Path


def load_data(dataset_name, base_path="e:/CS/research/w3/w3_code_project/w3_new_TAPE_TTT"):
    data_path = Path(base_path) / "code" / "dataset" / dataset_name
    graph_path = data_path / "geometric_data_with_texts.pt"
    data = torch.load(graph_path)

    y = data.y.squeeze()
    edge_index = data.edge_index

    if hasattr(data, "llm_score_matrix"):
        llm_scores = data.llm_score_matrix
    else:
        raise ValueError(f"No llm_score_matrix found in {graph_path}")

    return llm_scores, y, edge_index


def compute_normalized_adj(edge_index, num_nodes, add_self_loop=True, device="cpu"):
    row, col = edge_index[0].to(device), edge_index[1].to(device)

    if add_self_loop:
        self_loop = torch.arange(num_nodes, device=device)
        row = torch.cat([row, self_loop])
        col = torch.cat([col, self_loop])

    deg = torch.zeros(num_nodes, device=device)
    deg.scatter_add_(0, row, torch.ones(row.size(0), device=device))

    deg_inv = 1.0 / deg.clamp_min(1e-12)
    edge_weight = deg_inv[row]

    S = torch.sparse_coo_tensor(
        indices=torch.stack([row, col]),
        values=edge_weight,
        size=(num_nodes, num_nodes),
        device=device
    ).coalesce()
    return S


def label_propagation(Y0, S, alpha=0.8, num_iter=3):
    Y = Y0.clone()
    for _ in range(num_iter):
        Y = alpha * torch.sparse.mm(S, Y) + (1 - alpha) * Y0
    Y = Y / Y.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return Y


def compute_pagerank(edge_index, num_nodes, damping=0.85, num_iter=50, tol=1e-8, device="cpu"):
    src = edge_index[0].to(device)
    dst = edge_index[1].to(device)

    out_deg = torch.zeros(num_nodes, device=device)
    out_deg.scatter_add_(0, src, torch.ones(src.size(0), device=device))
    dangling = (out_deg == 0)

    inv_out = 1.0 / out_deg.clamp_min(1e-12)

    PT = torch.sparse_coo_tensor(
        indices=torch.stack([dst, src]),
        values=inv_out[src],
        size=(num_nodes, num_nodes),
        device=device
    ).coalesce()

    r = torch.full((num_nodes, 1), 1.0 / num_nodes, device=device)
    teleport = (1.0 - damping) / num_nodes

    for _ in range(num_iter):
        r_old = r
        r = damping * torch.sparse.mm(PT, r_old)

        dangling_mass = r_old[dangling].sum()
        r = r + damping * (dangling_mass / num_nodes)
        r = r + teleport

        if torch.norm(r - r_old, p=1).item() < tol:
            break

    return r.squeeze(1)


def topk_truncate_rows(rows, k=3):
    M, C = rows.size(0), rows.size(1)
    k = min(k, C)

    topv, topi = torch.topk(rows, k=k, dim=1)
    new_rows = torch.zeros_like(rows)
    new_rows.scatter_(1, topi, topv)

    new_rows = new_rows / new_rows.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return new_rows


def analyze_one_dataset(
    dataset_name,
    ratios=(0.3, 0.5, 0.7, 0.9),
    alpha=0.8,
    num_iter=3,
    damping=0.85,
    pr_iter=50,
    base_path="e:/CS/research/w3/w3_code_project/w3_new_TAPE_TTT",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    llm_scores, y, edge_index = load_data(dataset_name, base_path=base_path)
    y = y.to(device)
    edge_index = edge_index.to(device)

    Y0_full = llm_scores.to(device)
    Y0_full = Y0_full / Y0_full.sum(dim=1, keepdim=True).clamp_min(1e-12)

    N = Y0_full.size(0)

    init_acc = (Y0_full.argmax(dim=1) == y).float().mean().item()

    pr = compute_pagerank(edge_index, num_nodes=N, damping=damping, num_iter=pr_iter, device=device)
    pr_rank = torch.argsort(pr, descending=True)

    S = compute_normalized_adj(edge_index, N, add_self_loop=True, device=device)

    out = {"init_acc": init_acc, "by_ratio": {}}

    for r in ratios:
        k = max(1, int(N * r))
        seeds = pr_rank[:k]

        # 非 seeds 全置0，当作没预测
        Y0_mod = torch.zeros_like(Y0_full)

        # seeds 做 top-3 截断
        seed_rows_top3 = topk_truncate_rows(Y0_full[seeds], k=3)
        Y0_mod[seeds] = seed_rows_top3

        Y_lp = label_propagation(Y0_mod, S, alpha=alpha, num_iter=num_iter)
        acc = (Y_lp.argmax(dim=1) == y).float().mean().item()

        out["by_ratio"][r] = (acc, acc - init_acc)

    return out


def _fmt_acc_delta(acc, delta):
    """acc/delta 都是 0~1，输出为 65.03(+1.20) 这种"""
    acc100 = acc * 100.0
    delta100 = delta * 100.0
    sign = "+" if delta100 >= 0 else "-"
    return f"{acc100:.2f}({sign}{abs(delta100):.2f})"


def print_summary_table(results, ratios):
    # 表头：每个 PR% 只占一列（包含 acc 与 delta）
    header = ["Dataset", "Init(%)"] + [f"PR{int(r*100)}" for r in ratios]

    # 宽度（PR 列更宽以容纳 65.03(+1.20)）
    colw = [12, 10] + [14] * len(ratios)

    def fmt_row(items):
        s = ""
        for i, it in enumerate(items):
            s += str(it).ljust(colw[i])
        return s

    total_w = sum(colw)
    print("\n" + "=" * total_w)
    print("Summary: PageRank-Seed Top3 + Non-seed=0 + LP (Acc% (Δ%))")
    print("=" * total_w)
    print(fmt_row(header))
    print("-" * total_w)

    for ds, info in results.items():
        row = [ds, f"{info['init_acc']*100:.2f}"]
        for r in ratios:
            acc, delta = info["by_ratio"][r]
            row.append(_fmt_acc_delta(acc, delta))
        print(fmt_row(row))

    print("-" * total_w)

    # 平均（按数据集等权）
    if results:
        avg_init = sum(info["init_acc"] for info in results.values()) / len(results)
        row = ["Average", f"{avg_init*100:.2f}"]
        for r in ratios:
            avg_acc = sum(info["by_ratio"][r][0] for info in results.values()) / len(results)
            avg_delta = sum(info["by_ratio"][r][1] for info in results.values()) / len(results)
            row.append(_fmt_acc_delta(avg_acc, avg_delta))
        print(fmt_row(row))

    print("=" * total_w)


def main():
    datasets = ["cora", "citeseer", "pubmed", "wikics", "elephoto", "bookchild", "bookhis", "arxiv"]
    ratios = (0.3, 0.5, 0.7, 0.9)

    alpha = 0.8
    num_iter = 3

    results = {}
    for ds in datasets:
        try:
            results[ds] = analyze_one_dataset(
                dataset_name=ds,
                ratios=ratios,
                alpha=alpha,
                num_iter=num_iter,
                damping=0.85,
                pr_iter=50,
            )
        except Exception as e:
            print(f"[{ds}] ERROR: {e}")

    if results:
        print_summary_table(results, ratios)


if __name__ == "__main__":
    main()
