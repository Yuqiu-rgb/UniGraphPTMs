import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

# --- 辅助模块 ---

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ConvBlock(nn.Module):
    """
    轻量级卷积块，用于捕捉局部特征.
    使用深度可分离卷积 (Depthwise Separable Convolution) 提升效率.
    结构: Pointwise Conv -> Depthwise Conv -> Swish Activation -> Pointwise Conv
    """
    def __init__(self, dim, kernel_size=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim), # Depthwise
            nn.Conv1d(dim, dim, 1), # Pointwise
            Swish(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        # x shape: [batch, seq_len, dim]
        # Conv1d expects [batch, dim, seq_len]
        res = x
        x = x.transpose(1, 2)
        x = self.net(x)
        x = x.transpose(1, 2)
        return x + res # 残差连接

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# --- 主模块 ---

class SpatiallyAwareGatedCrossAttention(nn.Module):
    """
    空间感知门控交叉注意力模块 (SaGCA-Fusion)
    结合了卷积的局部性、相对位置编码的空间性和门控交叉注意力的可控性。

    Args:
        dim (int): 特征维度.
        seq_len (int): 序列长度，用于初始化相对位置偏置.
        num_heads (int): 多头注意力的头数.
        conv_kernel_size (int): 卷积核大小.
        ff_mult (int): 前馈网络隐藏层倍数.
        dropout (float): Dropout 比率.
    """
    def __init__(self, dim, seq_len, num_heads=8, conv_kernel_size=7, ff_mult=4, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        self.dim = dim
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 1. 创新点: 卷积预处理模块
        self.conv_main = ConvBlock(dim, conv_kernel_size)
        self.conv_side = ConvBlock(dim, conv_kernel_size)

        # 2. 核心: 交叉注意力相关层
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim * 2, bias=False)
        self.to_out = nn.Linear(dim, dim)
        
        # 3. 创新点: 相对位置偏置
        self.rel_pos_bias = nn.Embedding(2 * seq_len - 1, num_heads)
        # 预先计算相对位置索引，避免重复计算
        relative_indices = torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)
        relative_indices += seq_len - 1 # 偏移，使其从0开始
        self.register_buffer('relative_indices', relative_indices)

        # 4. 主分支门控
        self.to_gate = nn.Linear(dim, dim)

        # 5. FFN 和 LayerNorm
        self.ffn = FeedForward(dim, dim * ff_mult, dropout=dropout)
        self.norm_main = nn.LayerNorm(dim)
        self.norm_side = nn.LayerNorm(dim)
        self.norm_post_attn = nn.LayerNorm(dim)
        self.norm_post_ffn = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, main_feat, side_feat, mask=None):
        # 0. 卷积预处理，捕获局部特征
        main_conv = self.conv_main(main_feat)
        side_conv = self.conv_side(side_feat)

        # 归一化卷积后的特征
        main_norm = self.norm_main(main_conv)
        side_norm = self.norm_side(side_conv)

        # A. 主分支生成门控信号
        gate = torch.sigmoid(self.to_gate(main_norm))

        # B. 交叉注意力
        q = self.to_q(main_norm)
        k, v = self.to_kv(side_norm).chunk(2, dim=-1)

        q, k, v = map(lambda t: t.view(t.shape[0], t.shape[1], self.num_heads, self.head_dim).transpose(1, 2), (q, k, v))

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        # 注入相对位置偏置
        rel_bias = self.rel_pos_bias(self.relative_indices)
        rel_bias = rel_bias.permute(2, 0, 1).unsqueeze(0) # h i j -> 1 h i j
        dots += rel_bias
        
        if mask is not None:
            mask_value = -torch.finfo(dots.dtype).max
            dots.masked_fill_(~mask, mask_value)

        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = out.transpose(1, 2).contiguous().view(out.shape[0], self.seq_len, self.dim)
        cross_attn_out = self.to_out(out)

        # C. 门控融合 & 主干残差连接
        # 残差连接使用最原始的 main_feat，确保身份信息不丢失
        gated_info = cross_attn_out * gate
        main_with_side = main_feat + self.dropout(gated_info)
        main_with_side = self.norm_post_attn(main_with_side)

        # D. FFN
        ffn_out = self.ffn(main_with_side)
        final_output = main_with_side + self.dropout(ffn_out)
        final_output = self.norm_post_ffn(final_output)

        return final_output
