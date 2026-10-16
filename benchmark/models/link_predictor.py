"""
链接预测解码器
"""
import torch
import torch.nn as nn


class DotProductDecoder(nn.Module):
    """内积解码器"""

    def forward(self, z, edge_index):
        """计算边的得分"""
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=-1)


class MLPDecoder(nn.Module):
    """MLP解码器"""

    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z, edge_index):
        src, dst = edge_index
        h = torch.cat([z[src], z[dst]], dim=-1)
        return self.mlp(h).squeeze(-1)
