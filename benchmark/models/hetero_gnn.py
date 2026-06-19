"""
异构图神经网络模型 — RGCN, HAN, HGT

支持 torch_geometric HeteroData 输入，用于 Reddit/Amazon 异构图 benchmark。

模型：
- RGCNModel: Relational Graph Convolutional Network (Schlichtkrull et al., 2018)
- HANModel: Heterogeneous Graph Attention Network (Wang et al., 2019)
- HGTModel: Heterogeneous Graph Transformer (Hu et al., 2020)

配套：
- HeteroFeatureEncoder: 将各节点类型的原始特征投影到统一 hidden_dim
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


# ==================== 特征编码器 ====================

class HeteroFeatureEncoder(nn.Module):
    """
    将各节点类型的原始特征投影到统一 hidden_dim。
    对于无特征的节点类型（如 subreddit, category, brand），
    使用 learnable Embedding 投影到 hidden_dim。
    """

    def __init__(self, node_type_dims: dict, hidden_dim: int):
        """
        Args:
            node_type_dims: {node_type: input_feature_dim} 或 {node_type: num_nodes} (用 int)
                           如果 value 是 int，则使用 nn.Embedding(value, hidden_dim)
            hidden_dim: 统一的隐藏维度
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projections = nn.ModuleDict()
        self.embedding_types = set()

        for ntype, dim in node_type_dims.items():
            if isinstance(dim, int) and dim > 0:
                # 如果 dim 较大，视为 num_nodes → 用 Embedding
                # 阈值：如果 dim > 1000 或显式标记
                self.projections[ntype] = nn.Embedding(dim, hidden_dim)
                self.embedding_types.add(ntype)
            else:
                # 线性投影
                in_dim = dim if isinstance(dim, int) else len(dim)
                self.projections[ntype] = nn.Linear(in_dim, hidden_dim)

    def forward(self, x_dict: dict) -> dict:
        """
        Args:
            x_dict: {node_type: tensor}
                     有特征的: tensor shape [N, feat_dim]
                     无特征的: tensor shape [N] (long, node indices)
        Returns:
            {node_type: tensor shape [N, hidden_dim]}
        """
        out = {}
        for ntype, proj in self.projections.items():
            if ntype in x_dict:
                x = x_dict[ntype]
                if ntype in self.embedding_types:
                    # x 是 long tensor [N]
                    out[ntype] = proj(x.long())
                else:
                    out[ntype] = proj(x.float())
        return out


# ==================== RGCN ====================

class RGCNModel(nn.Module):
    """
    Relational Graph Convolutional Network (Schlichtkrull et al., 2018)

    将异构图扁平化为统一 edge_index + edge_type 整数张量，
    使用关系特定权重矩阵处理不同边类型。

    输入格式：(x, edge_index, edge_type) — 扁平化异构图
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 num_relations: int, num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        try:
            from torch_geometric.nn import RGCNConv
        except ImportError:
            raise ImportError("请安装 torch_geometric: pip install torch_geometric")

        self.convs = nn.ModuleList()
        self.convs.append(RGCNConv(in_dim, hidden_dim, num_relations=num_relations))
        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations))
        self.convs.append(RGCNConv(hidden_dim, out_dim, num_relations=num_relations))

        self._reset_parameters()

    def _reset_parameters(self):
        for conv in self.convs:
            nn.init.xavier_uniform_(conv.weight)
            if hasattr(conv, 'root') and conv.root is not None:
                nn.init.xavier_uniform_(conv.root)
            if hasattr(conv, 'bias') and conv.bias is not None:
                nn.init.zeros_(conv.bias)

    def forward(self, x, edge_index, edge_type):
        """
        Args:
            x: [N, in_dim] 节点特征（所有类型已拼接）
            edge_index: [2, E] 边索引
            edge_type: [E] 每条边的类型整数
        Returns:
            [N, out_dim] logits
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_type)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_type)
        return x


# ==================== HAN ====================

class HANModel(nn.Module):
    """
    Heterogeneous Graph Attention Network (Wang et al., 2019)

    使用 HANConv 直接操作 HeteroData 的 x_dict 和 edge_index_dict，
    在节点级和语义级双层注意力机制聚合信息。

    输入格式：(x_dict, edge_index_dict) — HeteroData 格式
    """

    def __init__(self, hidden_dim: int, out_dim: int,
                 metadata: tuple, num_layers: int = 2,
                 heads: int = 4, dropout: float = 0.5):
        """
        Args:
            hidden_dim: 隐藏维度（所有节点类型统一）
            out_dim: 输出类别数
            metadata: (node_types, edge_types) 从 HeteroData.metadata() 获取
            num_layers: GNN 层数
            heads: 注意力头数
            dropout: dropout 率
        """
        super().__init__()
        self.dropout_rate = dropout

        try:
            from torch_geometric.nn import HANConv
        except ImportError:
            raise ImportError("请安装 torch_geometric: pip install torch_geometric")

        node_types, edge_types = metadata

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(HANConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                metadata=metadata,
                heads=heads,
                dropout=dropout,
            ))

        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x_dict, edge_index_dict, target_node_type='post'):
        """
        Args:
            x_dict: {node_type: [N, hidden_dim]}
            edge_index_dict: {edge_type_tuple: [2, E]}
            target_node_type: 目标节点类型（用于分类）
        Returns:
            [N_target, out_dim] logits
        """
        for conv in self.convs:
            x_dict_new = conv(x_dict, edge_index_dict)
            # HANConv 可能对某些孤立节点类型返回 None
            x_dict = {
                key: F.dropout(F.relu(x), p=self.dropout_rate, training=self.training)
                if x is not None else x_dict[key]
                for key, x in x_dict_new.items()
            }

        return self.classifier(x_dict[target_node_type])


# ==================== HGT ====================

class HGTModel(nn.Module):
    """
    Heterogeneous Graph Transformer (Hu et al., 2020)

    使用 HGTConv 实现类型特定的 Transformer 注意力，
    能根据节点类型和边类型动态调整注意力权重。

    输入格式：(x_dict, edge_index_dict) — HeteroData 格式
    """

    def __init__(self, hidden_dim: int, out_dim: int,
                 metadata: tuple, num_layers: int = 2,
                 heads: int = 4, dropout: float = 0.5):
        """
        Args:
            hidden_dim: 隐藏维度
            out_dim: 输出类别数
            metadata: (node_types, edge_types) 从 HeteroData.metadata() 获取
            num_layers: Transformer 层数
            heads: 注意力头数
            dropout: dropout 率
        """
        super().__init__()
        self.dropout_rate = dropout

        try:
            from torch_geometric.nn import HGTConv
        except ImportError:
            raise ImportError("请安装 torch_geometric: pip install torch_geometric")

        node_types, edge_types = metadata

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(HGTConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                metadata=metadata,
                heads=heads,
            ))

        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x_dict, edge_index_dict, target_node_type='post'):
        """
        Args:
            x_dict: {node_type: [N, hidden_dim]}
            edge_index_dict: {edge_type_tuple: [2, E]}
            target_node_type: 目标节点类型
        Returns:
            [N_target, out_dim] logits
        """
        for conv in self.convs:
            x_dict_new = conv(x_dict, edge_index_dict)
            # HGTConv 只更新参与消息传递的节点类型，其余保持不变
            new_x_dict = {}
            for key in x_dict:
                if key in x_dict_new and x_dict_new[key] is not None:
                    new_x_dict[key] = F.dropout(
                        F.relu(x_dict_new[key]),
                        p=self.dropout_rate, training=self.training
                    )
                else:
                    new_x_dict[key] = x_dict[key]
            x_dict = new_x_dict

        return self.classifier(x_dict[target_node_type])


# ==================== 辅助函数 ====================

def flatten_hetero_data(hetero_data, target_node_type='post'):
    """
    将 HeteroData 扁平化为 RGCN 兼容格式：
    (x, edge_index, edge_type, node_type_ids)

    Args:
        hetero_data: torch_geometric HeteroData 对象
        target_node_type: 目标节点类型

    Returns:
        x: [N_total, feat_dim] 所有节点特征拼接
        edge_index: [2, E_total] 所有边拼接
        edge_type: [E_total] 每条边的类型 ID
        target_mask: [N_total] bool，标记目标节点
        node_type_ids: [N_total] 每个节点的类型 ID
    """
    import torch

    node_types = hetero_data.node_types
    edge_type_tuples = list(hetero_data.edge_types)

    # 建立映射
    node_type_to_id = {nt: i for i, nt in enumerate(node_types)}
    edge_type_to_id = {et: i for i, et in enumerate(edge_type_tuples)}

    # 拼接节点特征
    all_x = []
    node_type_ids = []
    target_mask = []
    node_offsets = {}

    offset = 0
    for nt in node_types:
        x = hetero_data[nt].x
        if x is None:
            # 无特征节点：用 one-hot 或随机初始化
            num_nodes = hetero_data[nt].num_nodes
            x = torch.randn(num_nodes, hetero_data[target_node_type].x.size(1))
        num_nodes = x.size(0)
        all_x.append(x)
        node_type_ids.append(torch.full((num_nodes,), node_type_to_id[nt], dtype=torch.long))
        target_mask.append(torch.full((num_nodes,), nt == target_node_type, dtype=torch.bool))
        node_offsets[nt] = offset
        offset += num_nodes

    x = torch.cat(all_x, dim=0)
    node_type_ids = torch.cat(node_type_ids, dim=0)
    target_mask = torch.cat(target_mask, dim=0)

    # 拼接边 + 偏移
    all_edges = []
    all_edge_types = []

    for et in edge_type_tuples:
        src_type, _, dst_type = et
        edge_index = hetero_data[et].edge_index
        src_offset = node_offsets[src_type]
        dst_offset = node_offsets[dst_type]

        shifted = edge_index.clone()
        shifted[0] += src_offset
        shifted[1] += dst_offset

        all_edges.append(shifted)
        all_edge_types.append(
            torch.full((edge_index.size(1),), edge_type_to_id[et], dtype=torch.long)
        )

    if all_edges:
        edge_index = torch.cat(all_edges, dim=1)
        edge_type = torch.cat(all_edge_types, dim=0)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_type = torch.zeros(0, dtype=torch.long)

    return x, edge_index, edge_type, target_mask, node_type_ids


# ==================== Registry ====================

HETERO_MODEL_REGISTRY = {
    "rgcn": RGCNModel,
    "han": HANModel,
    "hgt": HGTModel,
}


def get_hetero_model(model_name: str, **kwargs):
    """获取异构 GNN 模型实例"""
    if model_name not in HETERO_MODEL_REGISTRY:
        raise ValueError(
            f"未知异构模型: {model_name}. "
            f"可用: {list(HETERO_MODEL_REGISTRY.keys())}"
        )
    return HETERO_MODEL_REGISTRY[model_name](**kwargs)
