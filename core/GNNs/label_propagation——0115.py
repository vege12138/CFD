"""
修改点：不同数据集用不同 alpha（iter 固定为 3），并保持 Top-1/2/3 对比与综合统计。

你给的参数：
cora       alpha=0.7, iter=3
citeseer   alpha=0.9, iter=3
pubmed     alpha=0.6, iter=3
wikics     alpha=0.8, iter=3
elephoto   alpha=0.8, iter=3
bookchild  alpha=0.8, iter=3
bookhis    alpha=0.7, iter=3
arxiv      alpha=0.7, iter=3

注意：
- 下面的 summarize_best_topk 仍是“按数据集等权宏平均”+“按节点数加权平均”
- 每个数据集的 iter 都是 3；如果你未来想每个数据集 iter 也不同，可仿照 alpha_map 再加一个 iter_map
"""

import torch
import torch.nn.functional as F
from pathlib import Path


def load_data(dataset_name, base_path="e:/CS/research/w3/w3_code_project/w3_new_TAPE_TTT"):
    data_path = Path(base_path) / "code" / "dataset" / dataset_name
    graph_path = data_path / "geometric_data_with_texts.pt"
    data = torch.load(graph_path)

    y = data.y.squeeze()
    edge_index = data.edge_index
    if dataset_name == 'elephoto':
        a = 1
    if hasattr(data, "llm_score_matrix"):
        llm_scores = data.llm_score_matrix
    else:
        raise ValueError(f"No llm_score_matrix found in {graph_path}")

    return llm_scores, y, edge_index


def to_probabilities(llm_scores, eps=1e-12):
    llm_scores = llm_scores.float()

    row_sum = llm_scores.sum(dim=1)
    has_negative = (llm_scores.min() < 0).item()
    row_sum_bad = (torch.abs(row_sum - 1.0) > 1e-3).float().mean().item() > 0.01

    if has_negative or row_sum_bad:
        probs = F.softmax(llm_scores, dim=1)
    else:
        probs = llm_scores / llm_scores.sum(dim=1, keepdim=True).clamp_min(eps)
    return probs


def keep_topk_and_renorm(probs, k=1, eps=1e-12):
    assert k >= 1
    N, C = probs.shape

    topk_vals, topk_idx = torch.topk(probs, k=min(k, C), dim=1, largest=True, sorted=False)

    mask = torch.zeros_like(probs, dtype=torch.bool)
    mask.scatter_(1, topk_idx, True)

    probs_topk = torch.where(mask, probs, torch.zeros_like(probs))
    probs_topk = probs_topk / probs_topk.sum(dim=1, keepdim=True).clamp_min(eps)
    return probs_topk


def compute_normalized_adj(edge_index, num_nodes, add_self_loop=True, device=None):
    if device is None:
        device = edge_index.device

    row, col = edge_index[0].to(device), edge_index[1].to(device)

    if add_self_loop:
        self_loop = torch.arange(num_nodes, device=device)
        row = torch.cat([row, self_loop], dim=0)
        col = torch.cat([col, self_loop], dim=0)

    deg = torch.zeros(num_nodes, device=device, dtype=torch.float32)
    deg.scatter_add_(0, row, torch.ones(row.size(0), device=device, dtype=torch.float32))

    deg_inv = 1.0 / deg.clamp_min(1e-12)
    edge_weight = deg_inv[row]

    S = torch.sparse_coo_tensor(
        indices=torch.stack([row, col], dim=0),
        values=edge_weight,
        size=(num_nodes, num_nodes),
        device=device,
        dtype=torch.float32
    ).coalesce()

    return S


def label_propagation(Y0, S, alpha=0.8, num_iter=3, eps=1e-12):
    Y = Y0.clone()
    for _ in range(num_iter):
        Y = alpha * torch.sparse.mm(S, Y) + (1.0 - alpha) * Y0
    Y = Y / Y.sum(dim=1, keepdim=True).clamp_min(eps)
    return Y


@torch.no_grad()
def accuracy_from_distribution(Y, y_true):
    preds = Y.argmax(dim=1)
    return (preds == y_true).float().mean().item()


def row_sum_error_rate(Y, tol=1e-6):
    rs = Y.sum(dim=1)
    return (torch.abs(rs - 1.0) > tol).float().mean().item()


def analyze_one_dataset(dataset_name, alpha=0.8, num_iter=3, topks=(1, 2, 3),
                        base_path="e:/CS/research/w3/w3_code_project/w3_new_TAPE_TTT",
                        save_lp=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    llm_scores, y, edge_index = load_data(dataset_name, base_path=base_path)
    llm_scores = llm_scores.to(device)
    y = y.to(device)
    edge_index = edge_index.to(device)

    N = llm_scores.size(0)

    Y_full = to_probabilities(llm_scores)
    S = compute_normalized_adj(edge_index, N, add_self_loop=True, device=device)

    records = []
    for k in topks:
        Y0 = keep_topk_and_renorm(Y_full, k=k)

        init_acc = accuracy_from_distribution(Y0, y)
        init_row_err = row_sum_error_rate(Y0)

        Y_lp = label_propagation(Y0, S, alpha=alpha, num_iter=num_iter)
        lp_acc = accuracy_from_distribution(Y_lp, y)
        lp_row_err = row_sum_error_rate(Y_lp)

        rec = {
            "dataset": dataset_name,
            "alpha": float(alpha),
            "iters": int(num_iter),
            "topk": int(k),
            "num_nodes": int(N),
            "init_acc": float(init_acc),
            "lp_acc": float(lp_acc),
            "delta": float(lp_acc - init_acc),
            "init_row_err": float(init_row_err),
            "lp_row_err": float(lp_row_err),
        }
        records.append(rec)

        if save_lp:
            save_dir = Path(base_path) / "code" / "dataset" / dataset_name
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"lp_topk{k}_a{alpha}_iter{num_iter}.pt"
            torch.save(Y_lp.detach().cpu(), save_path)

    return records


def summarize_best_topk(all_records, topks=(1, 2, 3)):
    macro = {}
    weighted = {}

    for k in topks:
        rows = [r for r in all_records if r["topk"] == k]
        if not rows:
            continue

        macro[k] = sum(r["delta"] for r in rows) / len(rows)

        total_nodes = sum(r["num_nodes"] for r in rows)
        weighted[k] = sum(r["delta"] * r["num_nodes"] for r in rows) / max(1, total_nodes)

    best_topk_macro = max(macro.items(), key=lambda x: x[1])[0] if macro else None
    best_topk_weighted = max(weighted.items(), key=lambda x: x[1])[0] if weighted else None

    return macro, weighted, best_topk_macro, best_topk_weighted


def main():
    TOPKS = (1, 2, 3)

    BASE_PATH = "e:/CS/research/w3/w3_code_project/w3_new_TAPE_TTT"
    datasets = ["cora", "citeseer", "pubmed", "wikics", "elephoto", "bookchild", "bookhis"]

    # 迭代次数固定
    NUM_ITER = 3

    # 不同数据集的 alpha
    alpha_map = {
        "cora": 0.7,
        "citeseer": 0.9,
        "pubmed": 0.6,
        "wikics": 0.8,
        "elephoto": 0.8,
        "bookchild": 0.8,
        "bookhis": 0.7,
        "arxiv": 0.7,
    }

    all_records = []

    print("=" * 140)
    print(f"LP Comparison (dataset-specific alpha, fixed iters={NUM_ITER}) | Top-k={list(TOPKS)}")
    print("alpha_map =", alpha_map)
    print("=" * 140)

    for ds in datasets:
        try:
            alpha = alpha_map[ds]
            recs = analyze_one_dataset(
                ds,
                alpha=alpha,
                num_iter=NUM_ITER,
                topks=TOPKS,
                base_path=BASE_PATH,
                save_lp=False,
            )
            all_records.extend(recs)
        except Exception as e:
            print(f"[Error] dataset={ds}: {e}")

    # -------- 逐数据集输出 --------
    print("\n" + "=" * 140)
    print("Per-Dataset Results")
    print("=" * 140)
    print(f"{'Dataset':<12} {'alpha':<7} {'iter':<6} {'N':<9} {'Topk':<6} {'InitAcc':<10} {'LPAcc':<10} {'Delta':<10} {'RowErr(init->lp)'}")
    print("-" * 140)
    for r in all_records:
        row_err_str = f"{r['init_row_err']:.4f}->{r['lp_row_err']:.4f}"
        print(f"{r['dataset']:<12} {r['alpha']:<7.2f} {r['iters']:<6d} {r['num_nodes']:<9d} {r['topk']:<6d} "
              f"{r['init_acc']:<10.4f} {r['lp_acc']:<10.4f} {r['delta']:<+10.4f} {row_err_str}")

    # -------- 按 topk 的宏平均准确率（每个数据集等权）--------
    print("\n" + "=" * 140)
    print("Average by Top-k (Macro over datasets)")
    print("=" * 140)
    print(f"{'Topk':<6} {'AvgInitAcc':<12} {'AvgLPAcc':<12} {'AvgDelta':<12}")
    print("-" * 140)

    for k in TOPKS:
        rows = [r for r in all_records if r["topk"] == k]
        if not rows:
            continue
        avg_init = sum(r["init_acc"] for r in rows) / len(rows)
        avg_lp = sum(r["lp_acc"] for r in rows) / len(rows)
        avg_delta = sum(r["delta"] for r in rows) / len(rows)
        print(f"{k:<6d} {avg_init:<12.4f} {avg_lp:<12.4f} {avg_delta:<+12.4f}")

    # -------- 综合判断 topk=1/2/3 谁平均提升最大 --------
    macro, weighted, best_k_macro, best_k_weighted = summarize_best_topk(all_records, topks=TOPKS)

    print("\n" + "=" * 140)
    print("Which Top-k gives the BEST average improvement?")
    print("=" * 140)
    print("Macro Avg Improvement (each dataset equally weighted):")
    for k in TOPKS:
        if k in macro:
            print(f"  Top-{k}: {macro[k]:+.6f}  ({macro[k]*100:+.3f}%)")
    if best_k_macro is not None:
        print(f"==> BEST (Macro): Top-{best_k_macro}")

    print("\nWeighted Avg Improvement (weighted by num_nodes):")
    for k in TOPKS:
        if k in weighted:
            print(f"  Top-{k}: {weighted[k]:+.6f}  ({weighted[k]*100:+.3f}%)")
    if best_k_weighted is not None:
        print(f"==> BEST (Weighted): Top-{best_k_weighted}")

    print("=" * 140)


if __name__ == "__main__":
    main()
