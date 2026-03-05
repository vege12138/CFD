# =========================
#  Label Encoder
# =========================
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelEncoder(nn.Module):
    """
    MLP + Prototype Self-Attention Label Encoder (兼容旧接口)

    作用：
    - 先用 MLP 编码 label_embeddings
    - 再在“原型之间”做一次 self-attention（每个原型与所有原型交互）
    - 输出仍为 [num_classes, hidden_dim]，并 L2 normalize

    输入：
    - label_embeddings: [C, label_input_dim]
    - cluster_centers:  保留参数以兼容旧接口（本版本不使用）
    输出：
    - proto_encoded: [C, hidden_dim]
    """

    def __init__(
        self,
        label_input_dim=768,
        hidden_dim=768,
        num_layers=2,
        dropout=0.1,
        num_heads=8,          # 注意力头数
        attn_dropout=0.3,     # 注意力 dropout
        use_residual=True,    # 是否残差

    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.use_residual = use_residual

        # -------- 1) MLP 编码器 --------
        if num_layers == 0:
            self.mlp = nn.Identity()
            # 如果 num_layers==0 且维度不一致，需要一个投影
            self.in_proj = nn.Identity() if label_input_dim == hidden_dim else nn.Linear(label_input_dim, hidden_dim)
        else:
            layers = []
            for i in range(num_layers):
                in_d = label_input_dim if i == 0 else hidden_dim
                layers.append(nn.Linear(in_d, hidden_dim))
                if i < num_layers - 1:
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout))
            self.mlp = nn.Sequential(*layers)
            self.in_proj = nn.Identity()

        # -------- 2) 原型间 self-attention --------
        # MultiheadAttention 需要 [B, L, D]，这里 B=1，L=C（类别数）
        assert hidden_dim % num_heads == 0, "hidden_dim 必须能整除 num_heads"
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.ln_attn = nn.LayerNorm(hidden_dim)
        self.drop_attn = nn.Dropout(attn_dropout)

        # -------- 3) 可选：FFN（增强表达，仍很轻量）--------

        print(
            f"✅ LabelEncoder initialized: num_layers={num_layers}, "
            f"self-attn heads={num_heads}"
        )

    def forward(self, label_embeddings, cluster_centers=None):
        # 1) MLP 编码
        x = self.in_proj(label_embeddings)
        x = self.mlp(x)  # [C, H]
        #return x

        # 2) 原型间 self-attention：把原型当作 token 序列
        #    B=1, L=C, D=H
        x_seq = x.unsqueeze(0)  # [1, C, H]
        C = x_seq.size(1)
        attn_mask = torch.eye(C, device=x_seq.device, dtype=torch.bool)  # [C,C], True=禁止注意

        attn_out, attn_weights = self.self_attn(
            x_seq, x_seq, x_seq,
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=False,  # 想看每个 head 的权重就设 False
        )

        if self.use_residual:
            x_seq = self.ln_attn(x_seq - self.drop_attn(attn_out))
        else:
            x_seq = self.ln_attn(self.drop_attn(attn_out))

        out = x_seq.squeeze(0)  # [C, H]
        return F.normalize(out, p=2, dim=1)
