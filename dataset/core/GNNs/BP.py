"""
Belief Propagation (BP) for correcting LLM pseudo-label distributions on graphs
- log-space + damping（数值更稳）
- 可选：统计/估计 兼容矩阵 M（Potts 或 Full matrix）
- 对每个数据集：分别对 Top-1 / Top-2 / Top-3 截断归一化后的初始分布运行 BP
- 输出：各自 init acc / bp acc / improvement，并汇总哪个 top-k 平均提升最大

BP 视角：
- 节点势函数（unary potential）：phi_i(y)  <- 来自 LLM 分布（Top-k 截断后）
- 边势函数（pairwise potential）：psi_ij(y_i,y_j) <- 由 M 给定（同类更相容/异类更相容等）

Loopy BP（环图近似）：
    m_{i->j}(y_j) ∝ Σ_{y_i}  psi(y_i,y_j) * phi_i(y_i) * Π_{k∈N(i)\{j}} m_{k->i}(y_i)

在 log 空间：
    log m_{i->j}(y_j) = logsumexp_{y_i}( log psi(y_i,y_j) + log phi_i(y_i) + Σ_{k≠j} log m_{k->i}(y_i) )  + const

实现关键：
- 用“反向边索引 rev[e]”实现 “排除 j->i 的消息”
- 用 chunk 处理边，避免一次性临时张量过大
"""

import math
import torch
import torch.nn.functional as F
from pathlib import Path


# =========================
#  1) 数据加载
# =========================
def load_data(dataset_name, base_path="e:/CS/research/w3/w3_code_project/w3_new_TAPE_TTT"):
    """加载：llm_score_matrix, y, edge_index"""
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


# =========================
#  2) 概率分布与 Top-k 截断
# =========================
def to_probabilities(llm_scores, eps=1e-12):
    """
    把 llm_scores 转成每行和为 1 的概率分布：
    - 若存在负数 / 行和明显不为 1 -> 当作 logits/未归一化，softmax
    - 否则做一次归一化兜底
    """
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
    """
    每行保留 top-k 概率，其余置 0，然后重新归一化
    """
    assert k >= 1
    N, C = probs.shape
    k = min(k, C)

    topk_vals, topk_idx = torch.topk(probs, k=k, dim=1, largest=True, sorted=False)
    mask = torch.zeros_like(probs, dtype=torch.bool)
    mask.scatter_(1, topk_idx, True)

    probs_topk = torch.where(mask, probs, torch.zeros_like(probs))
    probs_topk = probs_topk / probs_topk.sum(dim=1, keepdim=True).clamp_min(eps)
    return probs_topk


@torch.no_grad()
def accuracy_from_distribution(Y, y_true):
    """argmax 分类准确率"""
    preds = Y.argmax(dim=1)
    return (preds == y_true).float().mean().item()


# =========================
#  3) 构建“有向边 + 反向边 rev”索引（BP 必备）
# =========================
def build_directed_edges_with_reverse(edge_index, num_nodes, make_undirected=True, remove_self_loops=False):
    """
    返回：
      src, dst: [E_dir]
      rev: [E_dir]  rev[e] 是与 e 方向相反的边索引（dst->src）
    说明：
      - BP 更新 m_{i->j} 时，需要排除来自 j->i 的那条消息，所以必须有 rev 映射
      - 为保证 rev 总存在，建议 make_undirected=True（补齐反向边）
    """
    row = edge_index[0].to(torch.long).cpu()
    col = edge_index[1].to(torch.long).cpu()

    if remove_self_loops:
        mask = row != col
        row, col = row[mask], col[mask]

    if make_undirected:
        # 拼接反向边，保证每条边都有 reverse
        row2 = torch.cat([row, col], dim=0)
        col2 = torch.cat([col, row], dim=0)
        row, col = row2, col2

    # 用 key = src*N + dst 代表一条有向边
    N = int(num_nodes)
    keys = row * N + col  # int64

    # sort + unique（去重），得到稳定的“按 key 排序”的边列表
    perm = torch.argsort(keys)
    keys_sorted = keys[perm]

    keep = torch.ones_like(keys_sorted, dtype=torch.bool)
    keep[1:] = keys_sorted[1:] != keys_sorted[:-1]  # 去掉重复 key

    perm_u = perm[keep]
    src = row[perm_u]
    dst = col[perm_u]
    keys_u = (src * N + dst)  # 已排序且唯一

    # rev：对每条边 (i->j) 找到 (j->i) 的位置
    keys_rev = dst * N + src  # 反向 key
    pos = torch.searchsorted(keys_u, keys_rev)

    # 检查是否都能找到
    ok = (pos >= 0) & (pos < keys_u.numel()) & (keys_u[pos] == keys_rev)
    if not bool(ok.all().item()):
        bad = (~ok).sum().item()
        raise RuntimeError(f"Reverse edge missing for {bad} directed edges. Try make_undirected=True.")

    rev = pos.to(torch.long)
    return src.to(torch.long), dst.to(torch.long), rev.to(torch.long)


# =========================
#  4) 估计/设置兼容矩阵 M（pairwise potential）
# =========================
def estimate_homophily_from_edges(yhat, src, dst):
    """h = P(y_i == y_j) over directed edges（近似同质性）"""
    yhat = yhat.to(torch.long)
    return (yhat[src] == yhat[dst]).float().mean().item()


def estimate_potts_beta_from_homophily(h, num_classes, eps=1e-6):
    """
    Potts 模型：diag 权重 w=exp(beta)，offdiag 权重=1
    若只看 pairwise（无 unary），两点同类概率约为：P(same)= w / (w + (C-1))
    反解：w = (C-1)*h/(1-h)
    """
    C = int(num_classes)
    h = max(eps, min(1.0 - eps, float(h)))
    w = (C - 1) * h / (1.0 - h)
    w = max(eps, w)
    beta = math.log(w)
    return beta


def estimate_full_M_from_edges(yhat, src, dst, num_classes, smoothing=1.0, symmetric=True, eps=1e-12):
    """
    统计 full 兼容矩阵 M（C x C）：
      counts[a,b] = 边上 (y_i=a, y_j=b) 的出现次数
    然后做平滑 +（可选）对称化 + 行归一化，得到 M 作为 P(y_j | y_i)
    """
    C = int(num_classes)
    yhat = yhat.to(torch.long).cpu()
    src = src.cpu()
    dst = dst.cpu()

    a = yhat[src]
    b = yhat[dst]
    idx = a * C + b  # [E]
    cnt = torch.bincount(idx, minlength=C * C).float().reshape(C, C)

    if symmetric:
        cnt = 0.5 * (cnt + cnt.t())

    cnt = cnt + float(smoothing)  # Laplace smoothing
    M = cnt / cnt.sum(dim=1, keepdim=True).clamp_min(eps)  # 行归一化
    logM = torch.log(M.clamp_min(eps))
    return logM  # [C,C] (CPU tensor)


# =========================
#  5) BP 核心（log-space + damping + chunk）
# =========================
def belief_propagation_logspace(
    Y0_probs,          # [N,C] 初始概率（Top-k 后）
    src, dst, rev,     # [E] 有向边与反向边索引
    num_iter=10,
    damping=0.5,       # 0~1：越大越相信新消息（常用 0.3~0.7）
    unary_temp=1.0,    # unary 温度：log_phi = log(Y0)/temp（temp小更信LLM）
    M_mode="potts",    # "potts" 或 "full"
    potts_beta=1.0,    # Potts 的 beta（diag=exp(beta), offdiag=1）
    full_logM=None,    # [C,C]（CPU 或 GPU 都行）
    pair_strength=1.0, # 对 logM 或 beta 的缩放
    chunk_size=200_000,
    msg_store_dtype=torch.float16,  # 存消息用 half 减内存；计算用 float32
    device=None,
    eps=1e-12,
    verbose=False
):
    """
    返回：
      beliefs_probs: [N,C] BP 推断后的概率分布
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Y0 = Y0_probs.to(device=device, dtype=torch.float32)
    N, C = Y0.shape
    E = src.numel()

    src = src.to(device)
    dst = dst.to(device)
    rev = rev.to(device)

    # unary：用 log_phi 表示（log 空间更稳）
    # 说明：unary_temp 越小 => log_phi 越尖锐 => 更相信 LLM
    log_phi = torch.log(Y0.clamp_min(eps)) / float(unary_temp)

    # 初始化 log-message：均匀分布（log(1/C)）
    log_msg = torch.full((E, C), fill_value=-math.log(C), device=device, dtype=msg_store_dtype)

    # 准备 pairwise 参数
    if M_mode == "potts":
        # Potts: psi(y_i,y_j)=exp(beta) if same else 1
        beta = float(potts_beta) * float(pair_strength)
        # w-1 用 expm1(beta) 更稳定
        w_minus1 = torch.expm1(torch.tensor(beta, device=device, dtype=torch.float32))
    elif M_mode == "full":
        if full_logM is None:
            raise ValueError("M_mode='full' requires full_logM.")
        logM = full_logM.to(device=device, dtype=torch.float32) * float(pair_strength)  # [C,C]
    else:
        raise ValueError(f"Unknown M_mode={M_mode}")

    # 迭代 BP
    for it in range(num_iter):
        # sum_in[i] = Σ_{k->i} log m_{k->i}  （product of messages => sum of logs）
        sum_in = torch.zeros((N, C), device=device, dtype=torch.float32)
        sum_in.index_add_(0, dst, log_msg.to(torch.float32))  # 按 dst 聚合 incoming 消息（log-space）

        # node "belief score"（未归一化）：log_phi + incoming_sum
        node_log_score = log_phi + sum_in  # [N,C]

        # 分 chunk 更新所有边的消息，避免一次性产生超大临时张量
        for st in range(0, E, chunk_size):
            ed = min(E, st + chunk_size)
            sl = slice(st, ed)

            # cavity：对边 i->j，排除来自 j->i 的那条消息（rev）
            # cavity[e] = log_phi[i] + Σ_{k->i} log m_{k->i} - log m_{j->i}
            cavity = node_log_score[src[sl]] - log_msg[rev[sl]].to(torch.float32)  # [B,C]

            if M_mode == "potts":
                # Potts 的高效更新（O(C)）：
                # msg_un[y] = S + (w-1)*exp(cavity[y]), 其中 S = Σ exp(cavity)
                logS = torch.logsumexp(cavity, dim=1, keepdim=True)  # [B,1]
                t = cavity - logS                                  # [B,C] = log softmax numerator
                # arg = (w-1) * exp(t)  (注意：w-1 可能为负，但 > -1，log1p 仍安全；数值边界做 clamp)
                arg = w_minus1 * torch.exp(t)
                arg = arg.clamp_min(-0.999999)  # 防止极端情况下 log1p(<-1)
                log_new = logS + torch.log1p(arg)  # [B,C]
            else:
                # Full M（O(C^2)）：log_new[yj] = logsumexp_{yi}( cavity[yi] + logM[yi,yj] )
                # 这里会构造 [B,C,C]，C 不大时可行；B 用 chunk 控制
                log_new = torch.logsumexp(cavity.unsqueeze(2) + logM.unsqueeze(0), dim=1)  # [B,C]

            # 归一化消息（每行和为 1）
            log_new = log_new - torch.logsumexp(log_new, dim=1, keepdim=True)

            # damping：m <- (1-d)*m_old + d*m_new
            d = float(damping)
            if d < 1.0:
                log_old = log_msg[sl].to(torch.float32)
                # log( (1-d)*exp(old) + d*exp(new) ) = logaddexp(old+log(1-d), new+log(d))
                log_mix = torch.logaddexp(
                    log_old + math.log(max(1e-12, 1.0 - d)),
                    log_new + math.log(max(1e-12, d))
                )
                # 再归一化一次，避免数值漂移
                log_mix = log_mix - torch.logsumexp(log_mix, dim=1, keepdim=True)
                log_msg[sl] = log_mix.to(msg_store_dtype)
            else:
                log_msg[sl] = log_new.to(msg_store_dtype)

        if verbose:
            # 打印一个粗略的“平均最大置信度”作为收敛感知
            with torch.no_grad():
                sum_in2 = torch.zeros((N, C), device=device, dtype=torch.float32)
                sum_in2.index_add_(0, dst, log_msg.to(torch.float32))
                log_b = log_phi + sum_in2
                log_b = log_b - torch.logsumexp(log_b, dim=1, keepdim=True)
                b = torch.exp(log_b)
                mean_conf = b.max(dim=1).values.mean().item()
            print(f"   [it={it+1:02d}] mean max-belief = {mean_conf:.4f}")

    # 最终 beliefs：b_i(y) ∝ phi_i(y) * Π_{k->i} m_{k->i}(y)
    sum_in = torch.zeros((N, C), device=device, dtype=torch.float32)
    sum_in.index_add_(0, dst, log_msg.to(torch.float32))
    log_belief = log_phi + sum_in
    log_belief = log_belief - torch.logsumexp(log_belief, dim=1, keepdim=True)
    beliefs = torch.exp(log_belief).clamp_min(eps)
    beliefs = beliefs / beliefs.sum(dim=1, keepdim=True).clamp_min(eps)

    return beliefs


# =========================
#  6) 单数据集：Top-1/2/3 跑 BP 并比较准确率
# =========================
def analyze_dataset_bp(
    dataset_name,
    topks=(1, 2, 3),
    base_path="e:/CS/research/w3/w3_code_project/w3_new_TAPE_TTT",
    num_iter=10,
    damping=0.5,
    unary_temp=1.0,
    M_mode="potts",           # "potts" or "full"
    estimate_M=True,          # 是否用当前 top-k 的硬预测统计 M / 或统计 homophily->beta
    potts_beta=1.0,           # 若 estimate_M=False 则使用此 beta
    pair_strength=1.0,
    chunk_size=200_000,
    device=None,
    verbose=False,
    save_bp=False
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    llm_scores, y, edge_index = load_data(dataset_name, base_path=base_path)
    llm_scores = llm_scores.to(torch.float32)
    y = y.to(torch.long)

    # 统一转概率
    Y_full = to_probabilities(llm_scores)

    N, C = Y_full.shape

    # 构建有向边与 rev（放 CPU 构建更省显存）
    src, dst, rev = build_directed_edges_with_reverse(edge_index, num_nodes=N, make_undirected=True)

    results = []
    for k in topks:
        # 1) Top-k 截断后的初始分布
        Y0 = keep_topk_and_renorm(Y_full, k=k)  # [N,C]

        # 2) 初始准确率
        init_acc = accuracy_from_distribution(Y0, y)

        # 3) 准备 M（按 top-k 的硬预测来统计更一致）
        yhat = Y0.argmax(dim=1)  # [N]

        full_logM = None
        beta = float(potts_beta)

        # 估计 M / beta
        if estimate_M:
            h = estimate_homophily_from_edges(yhat, src, dst)
            if M_mode == "potts":
                beta = estimate_potts_beta_from_homophily(h, num_classes=C)
            else:
                full_logM = estimate_full_M_from_edges(
                    yhat=yhat, src=src, dst=dst, num_classes=C,
                    smoothing=1.0, symmetric=True
                )
        else:
            h = estimate_homophily_from_edges(yhat, src, dst)

        # 4) 跑 BP
        beliefs = belief_propagation_logspace(
            Y0_probs=Y0,
            src=src, dst=dst, rev=rev,
            num_iter=num_iter,
            damping=damping,
            unary_temp=unary_temp,
            M_mode=M_mode,
            potts_beta=beta,
            full_logM=full_logM,
            pair_strength=pair_strength,
            chunk_size=chunk_size,
            msg_store_dtype=torch.float16,
            device=device,
            verbose=verbose
        )

        bp_acc = accuracy_from_distribution(beliefs.detach().cpu(), y)
        delta = bp_acc - init_acc

        rec = {
            "dataset": dataset_name,
            "N": int(N),
            "C": int(C),
            "E_dir": int(src.numel()),
            "topk": int(k),
            "init_acc": float(init_acc),
            "bp_acc": float(bp_acc),
            "delta": float(delta),
            "homophily_hat": float(h),
            "potts_beta_used": float(beta) if M_mode == "potts" else None,
            "M_mode": M_mode
        }
        results.append(rec)

        if save_bp:
            save_dir = Path(base_path) / "code" / "dataset" / dataset_name
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"bp_topk{k}_mode{M_mode}_iter{num_iter}_damp{damping}_temp{unary_temp}.pt"
            torch.save(beliefs.detach().cpu(), save_path)

    return results


# =========================
#  7) 汇总：哪个 top-k 平均提升最大
# =========================
def summarize_best_topk(all_records, topks=(1, 2, 3)):
    """
    计算两种“平均提升”：
      - macro：每个数据集等权平均（推荐先看）
      - weighted：按节点数 N 加权（大图影响更大）
    """
    macro = {}
    weighted = {}

    for k in topks:
        rows = [r for r in all_records if r["topk"] == k]
        if not rows:
            continue

        macro[k] = sum(r["delta"] for r in rows) / len(rows)

        total_n = sum(r["N"] for r in rows)
        weighted[k] = sum(r["delta"] * r["N"] for r in rows) / max(1, total_n)

    best_macro = max(macro.items(), key=lambda x: x[1])[0] if macro else None
    best_weighted = max(weighted.items(), key=lambda x: x[1])[0] if weighted else None
    return macro, weighted, best_macro, best_weighted


# =========================
#  8) main：一键运行所有数据集 + Top-1/2/3
# =========================
def main():
    BASE_PATH = "e:/CS/research/w3/w3_code_project/w3_new_TAPE_TTT"
    datasets = ["cora", "citeseer", "pubmed", "wikics", "elephoto", "bookchild", "bookhis", "arxiv"]

    # ====== BP 超参（你要“一组参数跑所有 top1/2/3”，就固定这些）======
    NUM_ITER = 10
    DAMPING = 0.5
    UNARY_TEMP = 1.0        # temp<1 更信 LLM；temp>1 更平滑
    PAIR_STRENGTH = 1.0

    # ====== M 模式 ======
    # "potts"：最省、最容易跑（推荐先用）
    # "full" ：统计一个 CxC 的 M（更灵活，但每次消息更新 O(C^2) 更慢）
    M_MODE = "potts"
    ESTIMATE_M = True       # 用 top-k 的硬预测统计 homophily -> beta（或统计 full M）

    TOPKS = (1, 2, 3)

    # 大图（arxiv）如果显存/速度吃紧，可把 chunk_size 调小一点（更省峰值内存但更慢）
    CHUNK_SIZE = 200_000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_records = []

    print("=" * 120)
    print(f"BP (log-space+damping) | M_mode={M_MODE} | estimate_M={ESTIMATE_M} | "
          f"iter={NUM_ITER} damp={DAMPING} unary_temp={UNARY_TEMP} pair_strength={PAIR_STRENGTH}")
    print("=" * 120)

    for ds in datasets:
        try:
            recs = analyze_dataset_bp(
                dataset_name=ds,
                topks=TOPKS,
                base_path=BASE_PATH,
                num_iter=NUM_ITER,
                damping=DAMPING,
                unary_temp=UNARY_TEMP,
                M_mode=M_MODE,
                estimate_M=ESTIMATE_M,
                potts_beta=1.0,          # 若 ESTIMATE_M=False 才会用到
                pair_strength=PAIR_STRENGTH,
                chunk_size=CHUNK_SIZE,
                device=device,
                verbose=False,
                save_bp=False
            )
            all_records.extend(recs)
        except Exception as e:
            print(f"[Error] dataset={ds}: {e}")

    # ====== 打印逐数据集结果 ======
    print("\n" + "=" * 120)
    print("Per-Dataset Results")
    print("=" * 120)
    header = f"{'Dataset':<12} {'N':<9} {'C':<4} {'E_dir':<10} {'Topk':<6} {'InitAcc':<10} {'BPAcc':<10} {'Delta':<10} {'homophily':<10} {'beta(Potts)':<12}"
    print(header)
    print("-" * 120)

    for r in all_records:
        beta_str = f"{r['potts_beta_used']:.4f}" if r["potts_beta_used"] is not None else "-"
        print(f"{r['dataset']:<12} {r['N']:<9d} {r['C']:<4d} {r['E_dir']:<10d} {r['topk']:<6d} "
              f"{r['init_acc']:<10.4f} {r['bp_acc']:<10.4f} {r['delta']:<+10.4f} {r['homophily_hat']:<10.4f} {beta_str:<12}")

    # ====== 综合：哪个 top-k 平均提升最大 ======
    macro, weighted, best_macro, best_weighted = summarize_best_topk(all_records, topks=TOPKS)

    print("\n" + "=" * 120)
    print("Which Top-k gives the BEST average improvement?")
    print("=" * 120)

    print("Macro Avg Improvement (each dataset equally weighted):")
    for k in TOPKS:
        if k in macro:
            print(f"  Top-{k}: {macro[k]:+.6f}  ({macro[k]*100:+.3f}%)")
    if best_macro is not None:
        print(f"==> BEST (Macro): Top-{best_macro}")

    print("\nWeighted Avg Improvement (weighted by num_nodes):")
    for k in TOPKS:
        if k in weighted:
            print(f"  Top-{k}: {weighted[k]:+.6f}  ({weighted[k]*100:+.3f}%)")
    if best_weighted is not None:
        print(f"==> BEST (Weighted): Top-{best_weighted}")

    print("=" * 120)


if __name__ == "__main__":
    main()
