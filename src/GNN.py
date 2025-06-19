import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv


class HierarchicalGNN(nn.Module):
    def __init__(self, input_dim=1280, output_dim=1024, seq_len=41):
        super().__init__()
        self.seq_len = seq_len
        self._init_graph_constructors()

        # 浅层 GCN + 局部感知
        self.gcn_conv = GCNConv(input_dim, output_dim)
        self.local_attention = nn.Linear(input_dim, 1)
        self.shallow_norm = nn.LayerNorm(output_dim)

        # 中层 GAT + 动态边权重
        self.gat_conv = GATConv(input_dim, output_dim // 4, heads=4)
        self.medium_norm = nn.LayerNorm(output_dim)

        # 深层 GraphSAGE + 特征传播
        self.sage_conv = SAGEConv(input_dim, output_dim)
        self.feature_propagator = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, seq_len)
        )
        self.deep_norm = nn.LayerNorm(output_dim)

        # 门控融合模块
        self.gate_controller = nn.Linear(3 * output_dim, 3)

    def _init_graph_constructors(self):
        """初始化三种图结构"""
        self.register_buffer('local_edges', self._create_local_edges(window=1))
        self.register_buffer('region_edges', self._create_region_edges())
        self.register_buffer('global_edges', self._create_global_edges())

    def _create_local_edges(self, window=1):
        """创建局部滑动窗口边"""
        edges = []
        for i in range(self.seq_len):
            for j in range(max(0, i - window), min(self.seq_len, i + window + 1)):
                if i != j:
                    edges.extend([(i, j), (j, i)])
        return torch.tensor(edges).t().contiguous()

    def _create_region_edges(self):
        """创建区域块连接边"""
        blocks = torch.split(torch.arange(self.seq_len), self.seq_len // 3)
        edges = []
        for block in blocks:
            for i in block:
                for j in block:
                    if i != j:
                        edges.extend([(i, j), (j, i)])
        return torch.tensor(edges).t().contiguous()

    def _create_global_edges(self):
        """创建全局连接边"""
        return torch.combinations(torch.arange(self.seq_len), 2).t().contiguous()

    def _build_batch_edges(self, edge_index, batch_size):
        """构建批次化的边索引"""
        edge_indices = []
        for b in range(batch_size):
            offset = b * self.seq_len
            edges = edge_index + offset
            edge_indices.append(edges)
        return torch.cat(edge_indices, dim=1)

    def _local_enhance(self, x):
        """局部特征增强"""
        weights = torch.sigmoid(self.local_attention(x))
        return x * weights

    def forward(self, x):
        batch_size = x.size(0)
        x_flat = x.view(-1, x.size(-1))

        # 浅层处理流程
        local_edges = self._build_batch_edges(self.local_edges, batch_size)
        enhanced_x = self._local_enhance(x_flat)
        shallow_out = self.gcn_conv(enhanced_x, local_edges)
        shallow_out = self.shallow_norm(shallow_out.view(batch_size, self.seq_len, -1))

        # 中层处理流程
        region_edges = self._build_batch_edges(self.region_edges, batch_size)
        gat_out = self.gat_conv(x_flat, region_edges)
        gat_out = self.medium_norm(gat_out.view(batch_size, self.seq_len, -1))

        # 深层处理流程
        global_edges = self._build_batch_edges(self.global_edges, batch_size)
        propagated_weights = F.softmax(self.feature_propagator(x), dim=-1)
        sage_input = torch.bmm(propagated_weights, x)
        sage_out = self.sage_conv(sage_input.view(-1, x.size(-1)), global_edges)
        sage_out = self.deep_norm(sage_out.view(batch_size, self.seq_len, -1))

        # 动态门控融合
        combined = torch.cat([shallow_out, gat_out, sage_out], dim=-1)
        gates = F.softmax(self.gate_controller(combined), dim=-1)

        # 加权合成最终输出->深层输出
        total_output = (gates[..., 0].unsqueeze(-1) * shallow_out +
                        gates[..., 1].unsqueeze(-1) * gat_out +
                        gates[..., 2].unsqueeze(-1) * sage_out)

        return [shallow_out, gat_out, total_output]


# 使用案例
if __name__ == "__main__":
    model = HierarchicalGNN()
    inputs = torch.randn(64, 41, 1280)
    print(f"总参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")


    # 获取三层特征
    shallow, medium, deep = model(inputs)

    print("浅层特征形状:", shallow.shape)  # [64, 41, 1024]
    print("中层特征形状:", medium.shape)  # [64, 41, 1024]
    print("深层特征形状:", deep.shape)  # [64, 41, 1024]
