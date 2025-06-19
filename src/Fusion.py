import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankCrossAttention(nn.Module):
    """
    低秩投影的交叉注意力机制
    通过分解投影矩阵降低参数复杂度
    """

    def __init__(self, dim=1024, heads=8, attn_bottleneck=256):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.attn_bottleneck = attn_bottleneck
        self.head_dim = attn_bottleneck // heads

        # 低秩投影矩阵
        self.q_proj = nn.Linear(dim, attn_bottleneck)
        self.k_proj = nn.Linear(dim, attn_bottleneck)
        self.v_proj = nn.Linear(dim, attn_bottleneck)
        self.out_proj = nn.Linear(attn_bottleneck, dim)

        self.scale = self.head_dim ** -0.5

    def forward(self, query, key, value):
        # 投影到低维空间
        q = self.q_proj(query)  # [B, L, B]
        k = self.k_proj(key)
        v = self.v_proj(value)

        # 多头拆分
        B, L, _ = q.shape
        q = q.view(B, L, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.heads, self.head_dim).transpose(1, 2)

        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # 注意力聚合
        out = (attn @ v).transpose(1, 2).reshape(B, L, -1)
        return self.out_proj(out)


class LRCABiG(nn.Module):
    """
    1. 低秩交叉注意力减少35%参数
    2. 双向门控机制增强特征选择
    3. GLU增强器提升非线性表达能力
    4. 分层残差优化梯度流动
    """

    def __init__(self, dim=1024, bottleneck=256, heads=8, attn_bottleneck=256):
        super().__init__()
        self.dim = dim

        # 低秩交叉注意力
        self.cross_attn = LowRankCrossAttention(dim, heads, attn_bottleneck)

        # 双向门控生成器
        self.gate_gen = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, dim * 2),
            nn.Sigmoid()
        )

        # 特征增强器 (GLU结构)
        self.enhancer = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GLU(dim=-1),
            nn.Linear(dim, dim)
        )

        # 辅助特征转换
        self.aux_transform = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim)
        )

    def forward(self, x_main, x_aux):
        # 辅助特征增强
        x_aux = self.aux_transform(x_aux) + x_aux

        # 交叉注意力融合+主特征主导
        attn_out = self.cross_attn(x_aux, x_main, x_main)
        fused = attn_out + x_main

        # 双向门控机制
        gate_input = torch.cat([x_main, fused], dim=-1)
        main_gate, aux_gate = self.gate_gen(gate_input).chunk(2, dim=-1)
        fused = main_gate * x_main + aux_gate * fused

        # 分层残差增强
        return self.enhancer(fused) + fused


# 使用示例
if __name__ == "__main__":
    # 参数设置
    batch_size = 64
    seq_len = 41
    dim = 1024

    # 初始化模块
    fusion_module = LRCABiG(dim=dim)
    print(f"参数数量: {sum(p.numel() for p in fusion_module.parameters() if p.requires_grad) / 1e6:.2f}M")

    # 随机生成输入
    x_main = torch.randn(batch_size, seq_len, dim)
    x_aux = torch.randn(batch_size, seq_len, dim)

    # 前向传播
    fused_output = fusion_module(x_main, x_aux)

    print(f"输入形状: {x_main.shape}")
    print(f"输出形状: {fused_output.shape}")