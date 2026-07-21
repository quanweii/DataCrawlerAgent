"""
GNN模型统一封装 - GCN/GAT/GraphSAGE/GIN/MLP
参考MAGB和OpenMAG的模型设置
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MLP(nn.Module):
    """MLP baseline（无图结构）"""

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(in_dim, out_dim))
        else:
            layers.append(nn.Linear(in_dim, hidden_dim))
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Linear(hidden_dim, out_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index=None):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x)
        return x


class GCN(nn.Module):
    """Graph Convolutional Network (Kipf & Welling, 2017)"""

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(nn.Linear(in_dim, hidden_dim))  # placeholder
        import torch_geometric
        from torch_geometric.nn import GCNConv
        if num_layers == 1:
            self.convs = nn.ModuleList([GCNConv(in_dim, out_dim)])
        else:
            self.convs = nn.ModuleList([GCNConv(in_dim, hidden_dim)])
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.convs.append(GCNConv(hidden_dim, out_dim))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GAT(nn.Module):
    """Graph Attention Network (Veličković et al., 2018)"""

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5, heads=4):
        super().__init__()
        self.dropout = dropout
        from torch_geometric.nn import GATConv
        if num_layers == 1:
            self.convs = nn.ModuleList([GATConv(in_dim, out_dim, heads=1)])
        else:
            self.convs = nn.ModuleList([GATConv(in_dim, hidden_dim, heads=heads)])
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads))
            self.convs.append(GATConv(hidden_dim * heads, out_dim, heads=1))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GraphSAGEModel(nn.Module):
    """GraphSAGE (Hamilton et al., 2017)"""

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        from torch_geometric.nn import SAGEConv
        if num_layers == 1:
            self.convs = nn.ModuleList([SAGEConv(in_dim, out_dim)])
        else:
            self.convs = nn.ModuleList([SAGEConv(in_dim, hidden_dim)])
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.convs.append(SAGEConv(hidden_dim, out_dim))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GIN(nn.Module):
    """Graph Isomorphism Network (Xu et al., 2019)"""

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        from torch_geometric.nn import GINConv
        if num_layers == 1:
            nn_layer = nn.Sequential(nn.Linear(in_dim, out_dim))
            self.convs = nn.ModuleList([GINConv(nn_layer)])
        else:
            nn1 = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(),
                                nn.Linear(hidden_dim, hidden_dim))
            self.convs = nn.ModuleList([GINConv(nn1)])
            for _ in range(num_layers - 2):
                nn_i = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim))
                self.convs.append(GINConv(nn_i))
            nn_last = nn.Sequential(nn.Linear(hidden_dim, out_dim))
            self.convs.append(GINConv(nn_last))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GCNII(nn.Module):
    """Graph Convolutional Network with Initial residual and Identity mapping (Chen et al., 2020)"""

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5, alpha=0.1, theta=0.5):
        super().__init__()
        self.dropout = dropout
        self.alpha = alpha
        self.theta = theta
        from torch_geometric.nn import GCNConv
        self.lin0 = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.lin_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        h = F.relu(self.lin0(x))
        h0 = h
        for conv in self.convs:
            h = conv(h, edge_index)
            h = h * (1 - self.alpha) + h0 * self.alpha
            h = h * (1 - self.theta) + h0 * self.theta
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.lin_out(h)


class GATv2(nn.Module):
    """Graph Attention Network v2 with dynamic attention (Brody et al., 2022)"""

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5, heads=4):
        super().__init__()
        self.dropout = dropout
        from torch_geometric.nn import GATv2Conv
        if num_layers == 1:
            self.convs = nn.ModuleList([GATv2Conv(in_dim, out_dim, heads=1)])
        else:
            self.convs = nn.ModuleList([GATv2Conv(in_dim, hidden_dim, heads=heads)])
            for _ in range(num_layers - 2):
                self.convs.append(GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads))
            self.convs.append(GATv2Conv(hidden_dim * heads, out_dim, heads=1))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class TransformerConvModel(nn.Module):
    """Graph Transformer using TransformerConv (Shi et al., 2021)"""

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5, heads=4):
        super().__init__()
        self.dropout = dropout
        from torch_geometric.nn import TransformerConv
        if num_layers == 1:
            self.convs = nn.ModuleList([TransformerConv(in_dim, out_dim, heads=1)])
        else:
            self.convs = nn.ModuleList([TransformerConv(in_dim, hidden_dim, heads=heads)])
            for _ in range(num_layers - 2):
                self.convs.append(TransformerConv(hidden_dim * heads, hidden_dim, heads=heads))
            self.convs.append(TransformerConv(hidden_dim * heads, out_dim, heads=1))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class MultimodalGNN(nn.Module):
    """
    注意力融合 + GNN：先学模态权重融合，再过GNN
    对比简单拼接的基线，验证更先进的融合方法是否能提升性能
    """

    def __init__(self, modality_dims, hidden_dim, out_dim,
                 num_layers=2, dropout=0.5, gnn_type="gcn"):
        super().__init__()
        from benchmark.encoders.multimodal_encoder import ModalityAttentionFusion
        self.fusion = ModalityAttentionFusion(modality_dims, output_dim=hidden_dim)
        self.dropout = dropout

        # GNN backbone
        gnn_cls = {
            "gcn": lambda: self._make_gcn(hidden_dim, hidden_dim, out_dim, num_layers),
            "gat": lambda: self._make_gat(hidden_dim, hidden_dim, out_dim, num_layers),
            "graphsage": lambda: self._make_sage(hidden_dim, hidden_dim, out_dim, num_layers),
        }.get(gnn_type, lambda: self._make_gcn(hidden_dim, hidden_dim, out_dim, num_layers))()
        self.gnn_layers = gnn_cls

    def _make_gcn(self, in_dim, hidden_dim, out_dim, num_layers):
        from torch_geometric.nn import GCNConv
        if num_layers == 1:
            return nn.ModuleList([GCNConv(in_dim, out_dim)])
        return nn.ModuleList(
            [GCNConv(in_dim, hidden_dim)] +
            [GCNConv(hidden_dim, hidden_dim)] * (num_layers - 2) +
            [GCNConv(hidden_dim, out_dim)]
        )

    def _make_gat(self, in_dim, hidden_dim, out_dim, num_layers):
        from torch_geometric.nn import GATConv
        if num_layers == 1:
            return nn.ModuleList([GATConv(in_dim, out_dim, heads=1)])
        return nn.ModuleList(
            [GATConv(in_dim, hidden_dim, heads=4)] +
            [GATConv(hidden_dim * 4, hidden_dim, heads=4)] * (num_layers - 2) +
            [GATConv(hidden_dim * 4, out_dim, heads=1)]
        )

    def _make_sage(self, in_dim, hidden_dim, out_dim, num_layers):
        from torch_geometric.nn import SAGEConv
        if num_layers == 1:
            return nn.ModuleList([SAGEConv(in_dim, out_dim)])
        return nn.ModuleList(
            [SAGEConv(in_dim, hidden_dim)] +
            [SAGEConv(hidden_dim, hidden_dim)] * (num_layers - 2) +
            [SAGEConv(hidden_dim, out_dim)]
        )

    def forward(self, x, edge_index, modality_embeddings=None):
        if modality_embeddings is not None:
            x = self.fusion(modality_embeddings)
        for i, conv in enumerate(self.gnn_layers[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gnn_layers[-1](x, edge_index)
        return x

    def get_modality_weights(self):
        return self.fusion.get_weights()


class RevGAT(nn.Module):
    """Reversible GAT (Li et al., 2021) - used in MAGB (KDD 2025).
    Uses reversed edges for bidirectional message passing + residual connections.
    """

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5, heads=4):
        super().__init__()
        self.dropout = dropout
        from torch_geometric.nn import GATConv
        if num_layers == 1:
            self.convs = nn.ModuleList([GATConv(in_dim, out_dim, heads=1, add_self_loops=True)])
        else:
            self.convs = nn.ModuleList([GATConv(in_dim, hidden_dim, heads=heads, add_self_loops=True)])
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, add_self_loops=True))
            self.convs.append(GATConv(hidden_dim * heads, out_dim, heads=1, add_self_loops=True))
        # Residual projections
        self.res_projections = nn.ModuleList()
        if num_layers > 1:
            if in_dim != hidden_dim * heads:
                self.res_projections.append(nn.Linear(in_dim, hidden_dim * heads))
            else:
                self.res_projections.append(nn.Identity())
            for _ in range(num_layers - 2):
                self.res_projections.append(nn.Identity())
            self.res_projections.append(nn.Identity())

    def forward(self, x, edge_index):
        # Add reversed edges for bidirectional message passing
        rev_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
        bidirectional_edges = torch.cat([edge_index, rev_edge_index], dim=1)
        # Remove duplicates
        bidirectional_edges = torch.unique(bidirectional_edges, dim=1)

        h = x
        for i, conv in enumerate(self.convs[:-1]):
            h_prev = h
            h = conv(h, bidirectional_edges)
            # Residual connection
            if i < len(self.res_projections):
                h = h + self.res_projections[i](h_prev)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.convs[-1](h, bidirectional_edges)
        return h


class ProjectedModel(nn.Module):
    """Projects multi-modal features to unified 768d before base GNN model.
    Ensures fair comparison: GNN always sees 768d regardless of modality combination."""

    def __init__(self, base_model, raw_dim, proj_dim=768):
        super().__init__()
        self.projector = nn.Linear(raw_dim, proj_dim)
        self.model = base_model
        self.raw_dim = raw_dim
        self.proj_dim = proj_dim

    def forward(self, x, edge_index):
        return self.model(self.projector(x), edge_index)

    @property
    def convs(self):
        return getattr(self.model, 'convs', None)

    @property
    def layers(self):
        return getattr(self.model, 'layers', None)

    @property
    def dropout(self):
        return getattr(self.model, 'dropout', 0.5)


MODEL_REGISTRY = {
    "mlp": MLP,
    "gcn": GCN,
    "gat": GAT,
    "graphsage": GraphSAGEModel,
    "gin": GIN,
    "gcnii": GCNII,
    "gatv2": GATv2,
    "gt": TransformerConvModel,
    "revgat": RevGAT,
    "multimodal_gnn": MultimodalGNN,
}

# 添加OpenMAG高级MAG模型
from benchmark.models.advanced_models import ADVANCED_MODEL_REGISTRY
MODEL_REGISTRY.update(ADVANCED_MODEL_REGISTRY)


def get_model(model_name: str, in_dim: int, hidden_dim: int, out_dim: int,
              num_layers: int = 2, dropout: float = 0.5, proj_dim=None, **kwargs):
    """根据名称创建模型，可选投影到 proj_dim（默认不投影）"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    if proj_dim is not None and in_dim != proj_dim:
        base = MODEL_REGISTRY[model_name](proj_dim, hidden_dim, out_dim, num_layers, dropout, **kwargs)
        return ProjectedModel(base, in_dim, proj_dim)
    return MODEL_REGISTRY[model_name](in_dim, hidden_dim, out_dim, num_layers, dropout, **kwargs)


# 逐模型学习率（OpenMAG 对每个模型单独调参；GIN 对默认 lr=0.01 敏感、易崩溃，降到 0.005 稳定）。
# 与梯度裁剪一起，消除 GIN 在高维多模态输入上的训练崩溃。
MODEL_LR = {
    "gin": 0.005,
}


def model_lr(model_name: str, default: float = 0.01) -> float:
    """返回某模型的训练学习率（GIN 0.005，其余 default）。"""
    return MODEL_LR.get(model_name, default)
