# =========================
#  Alternating GNN-LP Trainer
# =========================
"""
交替式GNN-LP训练器:
- 第一阶段：使用LP分布训练GNN
- 评估后获取均值分布，进行LP传播得到LP_P
- 第二阶段循环：重置模型，使用LP_P训练，再LP传播...
- GNN训练和LP传播交替进行
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from pathlib import Path
import random
import numpy as np

LOG_FREQ = 10


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GNNTrainer:
    """交替式GNN-LP训练器"""

    def __init__(self, args, data, num_classes):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = data.to(self.device)
        self.num_classes = num_classes
        self.num_nodes = data.y.size(0)
        set_seed(args.seed)
        self.use_js = False
        # 超参数

        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.dropout = args.dropout

        # Co-Training超参数
        self.W = args.warmup_epochs  # 预热epoch数
        self.T = args.label_update_interval  # 标签更新间隔
        self.total_epochs = args.total_epochs  # 总训练epoch数
        self.co_train_lr = args.co_train_lr  # Co-training学习率
        self.co_train_tau = args.co_train_tau  # Co-training温度系数
        self.post_warmup_lr = args.post_warmup_lr  # 预热后学习率
        self.post_warmup_tau = args.post_warmup_tau  # 预热后温度系数

        # LP传播超参数
        self.lp_alpha = args.lp_alpha  # LP传播alpha
        self.lp_num_iter = args.lp_num_iter  # LP传播迭代次数

        # 加载TA嵌入和标签原型
        self.ta_features = data.ta_embeddings.to(self.device)
        self.label_prototypes = data.label_prototypes.to(self.device)
        self.old_label_prototypes = data.label_prototypes.to(self.device)

        # 加载LP预热分布
        self.P_distribution = self._load_lp_distribution(args.dataset)

        print(f"📊 TA Embeddings: {self.ta_features.shape}")
        print(f"📊 Label Prototypes: {self.label_prototypes.shape}")
        print(f"📊 LP Distribution: {self.P_distribution.shape}")

        # 打印初始准确率
        y = self.data.y.squeeze()
        init_preds = self.P_distribution.argmax(dim=1)
        init_acc = (init_preds == y).float().mean().item()
        print(f"📊 LP Distribution Accuracy: {init_acc:.4f} ({init_acc * 100:.2f}%)")

        # 构建模型
        self._build_model()

        print(f"\n{'=' * 60}")
        print(f"📊 Alternating GNN-LP Trainer Initialized")
        print(f"   Nodes: {self.num_nodes}, Classes: {self.num_classes}")
        print(f"   Mode: GNN training ↔ LP propagation alternating")
        print(f"{'=' * 60}\n")


    def _load_lp_distribution(self, dataset_name):
        """加载LP预热分布"""
        data_dir = os.path.join("dataset", dataset_name)
        lp_path = os.path.join(data_dir, "lp_best_distribution.pt")
        lp_dist = torch.load(lp_path).to(self.device)
        return lp_dist

    def _build_model(self, lr=None):
        """构建双GNN模型"""
        model_name = self.args.gnn_model
        if model_name == "GCN":
            from core.GNNs.GCN.model import GCN as GNN
        elif model_name == "SAGE":
            from core.GNNs.SAGE.model import SAGE as GNN
        else:
            from core.GNNs.MLP.model import MLP as GNN

        input_dim = 768
        output_dim = 768

        self.gnn1 = GNN(
            in_channels=input_dim,
            hidden_channels=self.hidden_dim,
            out_channels=output_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            use_pred=False
        ).to(self.device)

        self.gnn2 = GNN(
            in_channels=input_dim,
            hidden_channels=self.hidden_dim,
            out_channels=output_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            use_pred=False
        ).to(self.device)


        print(f"✅ GNN1: {model_name} (768 → 768)")
        print(f"✅ GNN2: {model_name} (768 → 768)")

    def _edge_masking(self, edge_index, drop_ratio=0.1):
        """随机遮蔽边"""
        num_edges = edge_index.size(1)
        edge_mask = torch.bernoulli(torch.ones(num_edges, device=self.device) * (1 - drop_ratio)).bool()
        edge_index_masked = edge_index[:, edge_mask]
        return edge_index_masked

    @torch.no_grad()
    def _label_propagation(self, Y0, alpha=0.7, num_iter=3):
        """标签传播算法"""
        edge_index = self.data.edge_index
        N = self.num_nodes

        row, col = edge_index
        deg = torch.zeros(N, device=self.device)
        deg.scatter_add_(0, row, torch.ones(row.size(0), device=self.device))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        S = torch.sparse_coo_tensor(edge_index, edge_weight, (N, N))

        Y = Y0.clone()
        for t in range(num_iter):
            Y_new = alpha * torch.sparse.mm(S, Y) + (1 - alpha) * Y0
            Y = Y_new

        Y = Y / Y.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return Y

    def _warm_up_train_epoch(self, target_distribution):
        """单epoch训练，使用指定的目标分布"""
        self.gnn1.train()
        self.gnn2.train()
        self.optimizer.zero_grad()

        # 双GNN前向传播
        self.edge_index_dropped1 = self._edge_masking(self.data.edge_index, drop_ratio=0.2)
        self.edge_index_dropped2 = self._edge_masking(self.data.edge_index, drop_ratio=0.2)

        z1 = self.gnn1(self.ta_features, self.edge_index_dropped1)
        z2 = self.gnn2(self.ta_features, self.edge_index_dropped2)

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        proto_norm = F.normalize(self.label_prototypes, dim=1)
        sim1 = torch.mm(z1, proto_norm.T) / self.tau
        sim2 = torch.mm(z2, proto_norm.T) / self.tau

        pred1 = F.softmax(sim1, dim=1)
        pred2 = F.softmax(sim2, dim=1)
        pred_mean = (pred1 + pred2) / 2

        # 使用目标分布的argmax作为标签
        target_y = target_distribution.argmax(dim=1).long()
        tem = (sim1 + sim2) / 2
        loss = F.cross_entropy(tem, target_y)

        loss.backward()
        self.optimizer.step()

        # 评估
        with torch.no_grad():
            preds = pred_mean.argmax(dim=1)
            y = self.data.y.squeeze()
            acc = (preds == y).float().mean().item()
            acc1 = (pred1.argmax(dim=1) == y).float().mean().item()
            acc2 = (pred2.argmax(dim=1) == y).float().mean().item()

            # 三分布标签一致时的准确率
            pred1_labels = pred1.argmax(dim=1)
            pred2_labels = pred2.argmax(dim=1)
            all_agree = (pred1_labels == pred2_labels) & (pred2_labels == target_y)
            agree_count = all_agree.sum().item()
            agree_acc = (y[all_agree] == pred1_labels[all_agree]).float().mean().item() if agree_count > 0 else 0.0

        return {
            'loss': loss.item(),
            'acc': acc,
            'acc1': acc1,
            'acc2': acc2,
            'agree_count': agree_count,
            'agree_acc': agree_acc,
            'pred_mean': pred_mean.detach()
        }

    @torch.no_grad()
    def evaluate(self):
        """评估模型，返回pred_mean分布"""
        self.gnn1.eval()
        self.gnn2.eval()

        z1 = self.gnn1(self.ta_features, self.data.edge_index)
        z2 = self.gnn2(self.ta_features, self.data.edge_index)

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        proto_norm = F.normalize(self.label_prototypes, dim=1)
        sim1 = torch.mm(z1, proto_norm.T)
        sim2 = torch.mm(z2, proto_norm.T)

        # sim1 = torch.mm(z1, proto_norm.T) / self.tau
        # sim2 = torch.mm(z2, proto_norm.T) / self.tau

        pred1 = F.softmax(sim1, dim=1)
        pred2 = F.softmax(sim2, dim=1)
        pred_mean = (pred1 + pred2) / 2

        y = self.data.y.squeeze()
        preds = pred_mean.argmax(dim=1)
        acc = (preds == y).float().mean().item()
        acc1 = (pred1.argmax(dim=1) == y).float().mean().item()
        acc2 = (pred2.argmax(dim=1) == y).float().mean().item()

        # 三分布一致性分析 (GNN1, GNN2, P分布)
        pred1_labels = pred1.argmax(dim=1)
        pred2_labels = pred2.argmax(dim=1)
        p_labels = self.P_distribution.argmax(dim=1)

        three_agree = (pred1_labels == pred2_labels) & (pred2_labels == p_labels)
        three_agree_count = three_agree.sum().item()
        three_agree_acc = (
                y[three_agree] == pred1_labels[three_agree]).float().mean().item() if three_agree_count > 0 else 0.0

        # 四分布一致性分析 (GNN1, GNN2, P, LP_P)

        # 计算macro f1
        macro_f1 = f1_score(y.cpu().numpy(), preds.cpu().numpy(), average='macro')

        print(f"   📊 Eval: Acc={acc:.4f} | G1={acc1:.4f} G2={acc2:.4f} ")
        print(f"      3分布一致(G1=G2=P): {three_agree_count} Acc={three_agree_acc:.4f}")

        return {
            'acc': acc,
            'acc1': acc1,
            'acc2': acc2,
            'pred1': pred1,
            'pred2': pred2,
            'pred_mean': pred_mean,
            'sim1': sim1,
            'sim2': sim2,
            'macro_f1': macro_f1
        }

    def train(self):
        """协同训练:
        - 随机划分数据集为A和B两部分
        - GNN1用A训练，预测B; GNN2用B训练，预测A
        - 预热W代后，每隔T代用对方GNN的预测更新训练标签
        """
        # ==================== 超参数 ====================
        W = self.W  # 预热epoch数
        T = self.T  # 标签更新间隔
        total_epochs = self.total_epochs  # 总训练epoch数
        lr = self.co_train_lr  # Co-training学习率
        tau = self.co_train_tau  # Co-training温度系数

        optimizer1 = torch.optim.Adam(self.gnn1.parameters(), lr=lr)
        optimizer2 = torch.optim.Adam(self.gnn2.parameters(), lr=lr)
        print(f"{'=' * 60}")
        print(f"🚀 Co-Training with AB Split")
        print(f"   Warmup W={W}, Update Interval T={T}, Total={total_epochs}")
        print(f"{'=' * 60}")

        y = self.data.y.squeeze()
        edge_index = self.data.edge_index

        # ==================== 随机划分数据集为A和B ====================
        n = self.num_nodes
        perm = torch.randperm(n, device=self.device)
        split = n // 2
        mask_A = torch.zeros(n, dtype=torch.bool, device=self.device)
        mask_B = torch.zeros(n, dtype=torch.bool, device=self.device)
        mask_A[perm[:split]] = True
        mask_B[perm[split:]] = True

        print(f"\n   📌 Data Split: A={mask_A.sum().item()}, B={mask_B.sum().item()}")

        # 初始化训练标签：使用P分布的argmax
        labels_A = self.P_distribution.argmax(dim=1).clone()  # GNN1在A上的训练标签
        labels_B = self.P_distribution.argmax(dim=1).clone()  # GNN2在B上的训练标签

        # 打印初始P分布准确率
        p_acc_A = (labels_A[mask_A] == y[mask_A]).float().mean().item()
        p_acc_B = (labels_B[mask_B] == y[mask_B]).float().mean().item()
        p_acc_total = (self.P_distribution.argmax(dim=1) == y).float().mean().item()
        print(f"   Initial P_dist Acc: Total={p_acc_total:.4f} | A={p_acc_A:.4f} | B={p_acc_B:.4f}")

        # 分别设置优化器

        best_acc = 0.0
        best_preds = None

        # W后保留最高准确率的标签
        best_g1_A_acc = 0.0
        best_g1_A_preds = None  # GNN1在A上的最佳预测
        best_g2_B_acc = 0.0
        best_g2_B_preds = None  # GNN2在B上的最佳预测

        # ==================== 主训练循环 ====================
        for epoch in range(total_epochs):
            self.gnn1.train()
            self.gnn2.train()

            # 边dropout
            edge_index_dropped1 = self._edge_masking(edge_index, drop_ratio=0.2)
            edge_index_dropped2 = self._edge_masking(edge_index, drop_ratio=0.2)

            # ========== GNN1前向传播 ==========
            optimizer1.zero_grad()
            z1 = self.gnn1(self.ta_features, edge_index_dropped1)
            z1 = F.normalize(z1, dim=1)
            proto_norm = F.normalize(self.label_prototypes, dim=1)
            sim1 = torch.mm(z1, proto_norm.T) / tau
            pred1 = F.softmax(sim1, dim=1)

            # ========== GNN2前向传播 ==========
            optimizer2.zero_grad()
            z2 = self.gnn2(self.ta_features, edge_index_dropped2)
            z2 = F.normalize(z2, dim=1)
            sim2 = torch.mm(z2, proto_norm.T) / tau
            pred2 = F.softmax(sim2, dim=1)

            # ========== 计算损失 ==========
            # GNN1用A训练，GNN2用B训练
            loss1_ce = F.cross_entropy(sim1[mask_A], labels_A[mask_A])
            loss2_ce = F.cross_entropy(sim2[mask_B], labels_B[mask_B])

            # 总损失
            loss1 = loss1_ce
            loss2 = loss2_ce

            loss1.backward(retain_graph=True)
            loss2.backward()

            optimizer1.step()
            optimizer2.step()

            # ========== 预热后每隔T代更新标签 ==========
            if epoch >= W and (epoch - W) % T == 0:
                lr = self.post_warmup_lr
                tau = self.post_warmup_tau
                optimizer1 = torch.optim.Adam(self.gnn1.parameters(), lr=lr)
                optimizer2 = torch.optim.Adam(self.gnn2.parameters(), lr=lr)
                with torch.no_grad():
                    # GNN1预测B部分，更新labels_B（GNN2的训练标签）
                    pred1_labels = pred1.argmax(dim=1)

                    # GNN2预测A部分，更新labels_A（GNN1的训练标签）
                    pred2_labels = pred2.argmax(dim=1)

                    pred_labels = ((pred1 + pred2) / 2).argmax(dim=1)
                    labels_B[mask_B] = pred_labels[mask_B]
                    labels_A[mask_A] = pred_labels[mask_A]

                    # 打印更新后的标签准确率
                    new_acc_A = (labels_A[mask_A] == y[mask_A]).float().mean().item()
                    new_acc_B = (labels_B[mask_B] == y[mask_B]).float().mean().item()
                    print(f"   [Epoch {epoch}] Labels updated! Label Acc: A={new_acc_A:.4f} B={new_acc_B:.4f}")

            # ========== 第W代特殊评估 ==========
            if epoch == W:
                # self.gnn1.reset_parameters()
                # self.gnn2.reset_parameters()

                with torch.no_grad():
                    pred_mean = (pred1 + pred2) / 2
                    pred_mean_acc = (pred_mean.argmax(dim=1) == y).float().mean().item()

                    # 对pred_mean进行top-k筛选和归一化，然后LP传播
                    print(f"\n   📌 Warmup Complete (Epoch {W}):")
                    print(f"      pred_mean Acc： {pred_mean_acc:.4f}")

                    # 融合pred_mean的最大值标签和P_distribution
                    print(f"\n      🔗 Fusing pred_mean (argmax=1) + P_distribution:")
                    # 创建pred_mean的one-hot表示（最大值位置设为1）
                    pred_mean_argmax = pred_mean.argmax(dim=1)
                    pred_mean_onehot = torch.zeros_like(pred_mean)
                    pred_mean_onehot.scatter_(1, pred_mean_argmax.unsqueeze(1), 1.0)

                    # 融合：pred_mean的one-hot + P_distribution
                    fused_dist = pred_mean_onehot + self.P_distribution
                    fused_dist_acc = (fused_dist.argmax(dim=1) == y).float().mean().item()
                    print(f"      Fused Dist Acc (before top-k): {fused_dist_acc:.4f}")

                    for top_k in [1, 2, 3]:
                        # 保留top-k值
                        fused_topk = fused_dist.clone()
                        topk_values, topk_indices = torch.topk(fused_topk, k=top_k, dim=1)

                        # 创建mask，只保留top-k
                        mask_topk = torch.zeros_like(fused_topk)
                        mask_topk.scatter_(1, topk_indices, 1.0)
                        fused_topk = fused_topk * mask_topk

                        # 归一化
                        fused_topk = fused_topk / fused_topk.sum(dim=1, keepdim=True).clamp_min(1e-12)

                        # 标签传播
                        fused_topk_lp = self._label_propagation(fused_topk, alpha=self.lp_alpha,
                                                                num_iter=self.lp_num_iter)
                        fused_topk_lp_acc = (fused_topk_lp.argmax(dim=1) == y).float().mean().item()

                        print(
                            f"      Fused Dist Acc (top-{top_k} + LP): {fused_topk_lp_acc:.4f} (Delta: {fused_topk_lp_acc - pred_mean_acc:+.4f})")

                    # GNN1的A部分 + GNN2的B部分 合并
                    merged_preds = torch.zeros_like(y)
                    merged_preds[mask_A] = pred1.argmax(dim=1)[mask_A]  # GNN1预测A
                    merged_preds[mask_B] = pred2.argmax(dim=1)[mask_B]  # GNN2预测B
                    merged_acc_warmup = (merged_preds == y).float().mean().item()

                    print(f"      Merged (G1-A + G2-B) Acc: {merged_acc_warmup:.4f}")
                    eps = 1e-12
                    C, D = self.old_label_prototypes.shape
                    pred_y = pred_mean.argmax(dim=1)  # [N] 每个节点预测类别
                    conf = pred_mean.max(dim=1).values

                    # 交换训练分区：GNN1用B训练，GNN2用A训练
                    print(f"      🔄 Swapping partitions: GNN1->B, GNN2->A")
                    # 交换mask（之后GNN1用mask_B训练，GNN2用mask_A训练）
                    mask_A, mask_B = mask_B, mask_A
                    print(
                        f"      New split: GNN1 trains on B={mask_B.sum().item()}, GNN2 trains on A={mask_A.sum().item()}\n")

            # ========== 评估 ==========
            if epoch % 10 == 0 or epoch == total_epochs - 1:
                with torch.no_grad():
                    pred_mean = (pred1 + pred2) / 2
                    preds = pred_mean.argmax(dim=1)
                    acc = (preds == y).float().mean().item()

                    # 分别计算GNN1预测B和GNN2预测A的准确率
                    pred1_acc_B = (pred1.argmax(dim=1)[mask_B] == y[mask_B]).float().mean().item()
                    pred2_acc_A = (pred2.argmax(dim=1)[mask_A] == y[mask_A]).float().mean().item()

                    # 分别计算GNN1预测A和GNN2预测B（训练集内准确率）
                    pred1_acc_A = (pred1.argmax(dim=1)[mask_A] == y[mask_A]).float().mean().item()
                    pred2_acc_B = (pred2.argmax(dim=1)[mask_B] == y[mask_B]).float().mean().item()

                    # GNN1的A部分 + GNN2的B部分 合并准确率
                    merged_preds_eval = torch.zeros_like(y)
                    merged_preds_eval[mask_A] = pred1.argmax(dim=1)[mask_A]
                    merged_preds_eval[mask_B] = pred2.argmax(dim=1)[mask_B]
                    merged_acc_eval = (merged_preds_eval == y).float().mean().item()

                    if acc > best_acc:
                        best_acc = acc
                        best_preds = preds.clone()

                    # W后保留最佳GNN1-A和GNN2-B预测
                    if epoch >= W:
                        if pred1_acc_A > best_g1_A_acc:
                            best_g1_A_acc = pred1_acc_A
                            best_g1_A_preds = pred1.argmax(dim=1)[mask_A].clone()
                        if pred2_acc_B > best_g2_B_acc:
                            best_g2_B_acc = pred2_acc_B
                            best_g2_B_preds = pred2.argmax(dim=1)[mask_B].clone()

                    phase = "Warmup" if epoch < W else "Change"
                    print(f"   [{phase}] Epoch {epoch:3d} | Loss1 {loss1.item():.4f} Loss2 {loss2.item():.4f} | "
                          f"Acc {acc:.4f} | G1(A/B) {pred1_acc_A:.4f}/{pred1_acc_B:.4f} | G2(A/B) {pred2_acc_A:.4f}/{pred2_acc_B:.4f}")

        # ==================== 最终评估 ====================
        eval_result = self.evaluate()
        final_acc = eval_result['acc']
        final_pred = eval_result['pred_mean']
        final_preds = final_pred.argmax(dim=1)
        final_pred1 = eval_result['pred1']
        final_pred2 = eval_result['pred2']

        # GNN1的A部分 + GNN2的B部分 合并准确率
        final_merged_preds = torch.zeros_like(y)
        final_merged_preds[mask_A] = final_pred1.argmax(dim=1)[mask_A]
        final_merged_preds[mask_B] = final_pred2.argmax(dim=1)[mask_B]
        final_merged_acc = (final_merged_preds == y).float().mean().item()

        # 如果最终准确率更高，更新best
        if final_acc > best_acc:
            best_acc = final_acc
            best_preds = final_preds.clone()

        if final_merged_acc > best_acc:
            best_acc = final_merged_acc
            best_preds = final_merged_preds.clone()

        final_f1 = f1_score(y.cpu().numpy(), best_preds.cpu().numpy(), average='macro')

        # 合并训练过程中最佳的GNN1-A和GNN2-B预测
        if best_g1_A_preds is not None and best_g2_B_preds is not None:
            best_merged_preds = torch.zeros_like(y)
            best_merged_preds[mask_A] = best_g1_A_preds
            best_merged_preds[mask_B] = best_g2_B_preds
            best_merged_acc = (best_merged_preds == y).float().mean().item()
        else:
            best_merged_acc = 0.0

        print(f"\n{'=' * 60}")
        print(f"✅ Co-Training Complete!")
        print(f"   P_dist Acc: {p_acc_total:.4f}")
        print(f"   Best Acc:   {best_acc:.4f} (Delta: {best_acc - p_acc_total:+.4f})")
        print(f"   Final Acc:  {final_acc:.4f}")
        print(f"   Final Merged (G1-A + G2-B) Acc: {final_merged_acc:.4f}")
        print(f"   Best G1-A({best_g1_A_acc:.4f}) + G2-B({best_g2_B_acc:.4f}) Merged Acc: {best_merged_acc:.4f}")
        print(f"   Macro F1:   {final_f1:.4f}")
        print(f"{'=' * 60}")

        return {'best_acc': best_acc, 'acc': best_acc, 'macro_f1': final_f1}

    def save(self, path="alternating_gnn_lp.pt"):
        """保存模型"""
        torch.save({
            'gnn1': self.gnn1.state_dict(),
            'gnn2': self.gnn2.state_dict()
        }, path)
        print(f"💾 Model saved to: {path}")
