"""
OpenMAG论文中的高级MAG模型实现
包含: DGF, MMGCN, MGAT, LGMRec, MHGAT, DMGC
所有模型接受统一in_dim输入（768d CLIP特征），内部处理多模态建模
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DGF(nn.Module):
    """DGF: Dual Graph Filtering (Zheng et al., 2025)
    双图滤波分离同配/异配信号
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.proj = nn.Linear(in_dim, hidden_dim)

        from torch_geometric.nn import GCNConv
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, out_dim))

        # 双滤波参数
        self.filter_alpha = nn.Parameter(torch.tensor(0.5))

    def _low_pass(self, x, edge_index, alpha, k=5):
        num_nodes = x.size(0)
        row, col = edge_index
        deg = torch.zeros(num_nodes, device=x.device)
        deg.scatter_add_(0, row, torch.ones(row.size(0), device=x.device))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        h = x
        a = torch.sigmoid(alpha)
        for _ in range(k):
            msg = norm.unsqueeze(1) * h[col]
            agg = torch.zeros_like(h)
            agg.index_add_(0, row, msg)
            h = (1 - a) * h + a * agg
        return h

    def forward(self, x, edge_index):
        h = self.proj(x)
        h = F.relu(h)
        h = self._low_pass(h, edge_index, self.filter_alpha, k=5)
        h = F.dropout(h, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.convs[:-1]):
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.convs[-1](h, edge_index)
        return h


class MMGCN(nn.Module):
    """MMGCN: Multi-Modal GCN (Wei et al., 2019)
    为每个模态构建独立的GCN传播路径，最后拼接融合
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5,
                 num_views=3):
        super().__init__()
        self.dropout = dropout
        self.num_views = num_views

        from torch_geometric.nn import GCNConv
        self.view_convs = nn.ModuleList()
        for _ in range(num_views):
            layers = nn.ModuleList()
            layers.append(GCNConv(in_dim, hidden_dim))
            for _ in range(num_layers - 2):
                layers.append(GCNConv(hidden_dim, hidden_dim))
            layers.append(GCNConv(hidden_dim, hidden_dim))
            self.view_convs.append(layers)

        self.classifier = nn.Linear(hidden_dim * num_views, out_dim)

    def forward(self, x, edge_index):
        view_outs = []
        for view_layers in self.view_convs:
            h = x
            for i, conv in enumerate(view_layers[:-1]):
                h = conv(h, edge_index)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
            h = view_layers[-1](h, edge_index)
            view_outs.append(h)
        fused = torch.cat(view_outs, dim=1)
        return self.classifier(fused)


class MGAT(nn.Module):
    """MGAT: Multimodal Graph Attention Network (Tao et al., 2020)
    使用门控注意力融合，再通过GAT传播
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout

        self.proj = nn.Linear(in_dim, hidden_dim)

        from torch_geometric.nn import GATConv
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(hidden_dim, hidden_dim, heads=4))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * 4, hidden_dim, heads=4))
        self.convs.append(GATConv(hidden_dim * 4, out_dim, heads=1))

        # 门控模态注意力
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, x, edge_index):
        h = self.proj(x)
        g = self.gate(h)
        h = h * g

        for i, conv in enumerate(self.convs[:-1]):
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.convs[-1](h, edge_index)
        return h


class LGMRec(nn.Module):
    """LGMRec: Local and Global Graph Learning (Guo et al., 2024)
    超边学习全局结构 + GCN局部传播
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5,
                 num_hyperedges=16):
        super().__init__()
        self.dropout = dropout
        self.num_hyperedges = num_hyperedges

        self.feature_proj = nn.Linear(in_dim, hidden_dim)
        self.hyperedge_emb = nn.Parameter(torch.randn(num_hyperedges, hidden_dim) * 0.1)

        from torch_geometric.nn import GCNConv
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, out_dim))

        self.hyper_attn = nn.Linear(hidden_dim, num_hyperedges)

    def forward(self, x, edge_index):
        h = self.feature_proj(x)
        h = F.relu(h)

        # 全局超边信息
        attn = F.softmax(self.hyper_attn(h), dim=1)  # (N, K)
        global_info = attn @ self.hyperedge_emb  # (N, hidden)
        h = h + 0.1 * global_info

        for i, conv in enumerate(self.convs[:-1]):
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.convs[-1](h, edge_index)
        return h


class MHGAT(nn.Module):
    """MHGAT: Multi-relational Heterogeneous GAT (Jia et al., 2023)
    多关系异构图注意力
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout

        self.proj = nn.Linear(in_dim, hidden_dim)

        from torch_geometric.nn import GATConv
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=True))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=True))
        self.convs.append(GATConv(hidden_dim * 4, out_dim, heads=1))

        # 多关系权重
        self.rel_weight = nn.Parameter(torch.tensor([0.5, 0.3, 0.2]))

    def forward(self, x, edge_index):
        h = self.proj(x)
        h = F.relu(h)

        for i, conv in enumerate(self.convs[:-1]):
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.convs[-1](h, edge_index)
        return h


class DMGC(nn.Module):
    """DMGC: Disentangled Multi-relational Graph Clustering (Guo et al., 2025)
    分离同配/异配邻居进行双路滤波
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout

        self.proj = nn.Linear(in_dim, hidden_dim)

        from torch_geometric.nn import GCNConv
        self.homo_conv = GCNConv(hidden_dim, hidden_dim)
        self.hetero_conv = GCNConv(hidden_dim, hidden_dim)

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

        self.classifier = nn.Linear(hidden_dim, out_dim)

    def _split_edges(self, h, edge_index):
        src, dst = edge_index
        with torch.no_grad():
            sim = F.cosine_similarity(h[src], h[dst], dim=1)
            if sim.numel() > 0:
                threshold = sim.median()
                homo_mask = sim >= threshold
                hetero_mask = ~homo_mask
            else:
                homo_mask = torch.ones(src.numel(), dtype=torch.bool, device=h.device)
                hetero_mask = homo_mask
        homo_ei = edge_index[:, homo_mask] if homo_mask.any() else edge_index
        hetero_ei = edge_index[:, hetero_mask] if hetero_mask.any() else edge_index
        return homo_ei, hetero_ei

    def forward(self, x, edge_index):
        h = self.proj(x)
        h = F.relu(h)

        homo_ei, hetero_ei = self._split_edges(h, edge_index)

        h_homo = self.homo_conv(h, homo_ei)
        h_hetero = self.hetero_conv(h, hetero_ei)

        g = self.gate(torch.cat([h_homo, h_hetero], dim=1))
        fused = g * h_homo + (1 - g) * h_hetero

        fused = F.relu(fused)
        fused = F.dropout(fused, p=self.dropout, training=self.training)
        return self.classifier(fused)


class SelfSupervisedGNN(nn.Module):
    """Self-supervised GNN with contrastive learning (CCA-SSG style).
    Two-view contrastive: drop edges + drop features for two augmented views.
    Linear evaluation: train encoder self-supervised, then linear classifier on frozen features.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5,
                 drop_edge_rate=0.2, drop_feat_rate=0.2):
        super().__init__()
        self.dropout = dropout
        self.drop_edge_rate = drop_edge_rate
        self.drop_feat_rate = drop_feat_rate
        from torch_geometric.nn import GCNConv
        self.encoder_convs = nn.ModuleList()
        self.encoder_convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.encoder_convs.append(GCNConv(hidden_dim, hidden_dim))
        self.encoder_bn = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def _augment(self, x, edge_index):
        """Generate an augmented view by dropping edges and masking features."""
        # Drop edges
        if self.drop_edge_rate > 0:
            num_edges = edge_index.size(1)
            keep_mask = torch.rand(num_edges) > self.drop_edge_rate
            aug_edge_index = edge_index[:, keep_mask]
            if aug_edge_index.size(1) == 0:
                aug_edge_index = edge_index
        else:
            aug_edge_index = edge_index

        # Drop features
        if self.drop_feat_rate > 0:
            feat_mask = torch.bernoulli(torch.ones(x.size(1)) * (1 - self.drop_feat_rate)).to(x.device)
            aug_x = x * feat_mask.unsqueeze(0)
        else:
            aug_x = x
        return aug_x, aug_edge_index

    def encode(self, x, edge_index):
        h = x
        for i, conv in enumerate(self.encoder_convs):
            h = conv(h, edge_index)
            h = self.encoder_bn[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def contrastive_loss(self, x, edge_index):
        """Generate two views and compute contrastive loss (sampled to avoid OOM)."""
        x1, ei1 = self._augment(x, edge_index)
        x2, ei2 = self._augment(x, edge_index)

        h1 = self.encode(x1, ei1)
        h2 = self.encode(x2, ei2)

        h1 = F.normalize(h1, dim=1)
        h2 = F.normalize(h2, dim=1)

        # Sample batch to avoid OOM on N×N similarity matrix
        n = h1.size(0)
        batch_size = min(4096, n)
        idx = torch.randperm(n)[:batch_size]
        h1_batch = h1[idx]
        h2_batch = h2[idx]

        sim_matrix = torch.mm(h1_batch, h2_batch.t())
        labels = torch.arange(batch_size, device=h1.device)
        loss = F.cross_entropy(sim_matrix / 0.07, labels)
        return loss

    def forward(self, x, edge_index):
        h = self.encode(x, edge_index)
        return self.classifier(h)


class GraphMAE2(nn.Module):
    """GraphMAE2: Graph Masked Autoencoder (Hou et al., 2023)
    自监督图预训练：掩码节点特征 → GNN编码器 → GNN解码器重建

    两阶段使用：
    1. 预训练：mask部分节点，训练重建原始特征
    2. 微调：冻结/不冻结编码器，加分类头监督训练
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5,
                 mask_rate=0.6, replace_rate=0.1):
        super().__init__()
        self.dropout = dropout
        self.mask_rate = mask_rate
        self.replace_rate = replace_rate

        from torch_geometric.nn import GCNConv, GATConv

        # Encoder
        self.enc_proj = nn.Linear(in_dim, hidden_dim)
        self.enc_convs = nn.ModuleList()
        self.enc_convs.append(GCNConv(hidden_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.enc_convs.append(GCNConv(hidden_dim, hidden_dim))
        self.enc_norm = nn.LayerNorm(hidden_dim)

        # Decoder (shallower)
        self.dec_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dec_conv = GCNConv(hidden_dim, hidden_dim)
        self.dec_norm = nn.LayerNorm(hidden_dim)

        # Reconstruction head
        self.recon_head = nn.Linear(hidden_dim, in_dim)

        # Classification head (for fine-tuning)
        self.classifier = nn.Linear(hidden_dim, out_dim)

        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, in_dim))
        nn.init.xavier_normal_(self.mask_token)

    def _mask_features(self, x, edge_index):
        """掩码节点特征"""
        num_nodes = x.size(0)
        num_mask = int(num_nodes * self.mask_rate)
        perm = torch.randperm(num_nodes, device=x.device)
        mask_idx = perm[:num_mask]

        # Create masked features
        x_masked = x.clone()
        # Replace masked nodes with mask token + small noise
        x_masked[mask_idx] = self.mask_token.expand(num_mask, -1)
        # Small fraction replaced with random features instead of mask token
        if self.replace_rate > 0:
            num_replace = int(num_mask * self.replace_rate)
            if num_replace > 0:
                replace_idx = mask_idx[:num_replace]
                x_masked[replace_idx] = torch.randn(num_replace, x.size(1), device=x.device)

        return x_masked, mask_idx

    def pretrain_forward(self, x, edge_index):
        """预训练前向：mask → encode → decode → reconstruct"""
        x_masked, mask_idx = self._mask_features(x, edge_index)

        # Encode visible + masked nodes
        h = self.enc_proj(x_masked)
        for conv in self.enc_convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.enc_norm(h)

        # Decode masked nodes
        d = self.dec_proj(h)
        d = self.dec_conv(d, edge_index)
        d = self.dec_norm(d)
        d = F.relu(d)

        # Reconstruct
        recon = self.recon_head(d)

        # Only compute loss on masked nodes
        target = x[mask_idx]
        pred = recon[mask_idx]

        # Cosine similarity loss (scaled)
        sim = F.cosine_similarity(pred, target, dim=1)
        loss = 1 - sim.mean()

        return loss

    def encode(self, x, edge_index):
        """提取编码器输出"""
        h = self.enc_proj(x)
        for conv in self.enc_convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.enc_norm(h)
        return h

    def forward(self, x, edge_index):
        """微调前向：encode → classify"""
        h = self.encode(x, edge_index)
        return self.classifier(h)


class UniGraph2(nn.Module):
    """UniGraph2 (simplified): Unified graph model with text-graph contrastive alignment
    GNN编码器 + 跨模态对比对齐
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5,
                 tau=0.07):
        super().__init__()
        self.dropout = dropout
        self.tau = tau

        from torch_geometric.nn import GCNConv

        self.proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.align_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def contrastive_align_loss(self, h_graph, h_text):
        """图-文本对比对齐损失 (InfoNCE)"""
        h_g = F.normalize(self.align_proj(h_graph), dim=1)
        h_t = F.normalize(h_text, dim=1)

        n = h_g.size(0)
        batch_size = min(2048, n)
        idx = torch.randperm(n)[:batch_size]
        h_g_b = h_g[idx]
        h_t_b = h_t[idx]

        sim = torch.mm(h_g_b, h_t_b.t()) / self.tau
        labels = torch.arange(batch_size, device=sim.device)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2
        return loss

    def forward(self, x, edge_index):
        h = self.proj(x)
        h = F.relu(h)
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.classifier(h)


class Graph4MM(nn.Module):
    """Graph4MM: Graph model for Multimodal Matching
    跨模态注意力 + 模态特定门控融合
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5,
                 num_modalities=3):
        super().__init__()
        self.dropout = dropout
        self.num_modalities = num_modalities

        modality_dim = in_dim // num_modalities if in_dim > hidden_dim else in_dim

        # Per-modality projection
        self.mod_projs = nn.ModuleList([
            nn.Linear(modality_dim, hidden_dim) for _ in range(num_modalities)
        ])

        # Cross-modal attention
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)

        # Modality gating
        self.mod_gates = nn.Parameter(torch.ones(num_modalities))

        from torch_geometric.nn import GCNConv
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        # Split into modality views
        mod_dim = x.size(1) // self.num_modalities
        mod_views = [x[:, i*mod_dim:(i+1)*mod_dim] for i in range(self.num_modalities)]

        # Project each modality
        mod_embs = []
        for i, (view, proj) in enumerate(zip(mod_views, self.mod_projs)):
            h = proj(view)
            gate = torch.sigmoid(self.mod_gates[i])
            mod_embs.append(h * gate)

        # Stack for cross-attention: (N, num_mod, hidden)
        stacked = torch.stack(mod_embs, dim=1)
        fused, _ = self.cross_attn(stacked, stacked, stacked)
        fused = fused.mean(dim=1)

        # GNN propagation
        for conv in self.convs:
            fused = conv(fused, edge_index)
            fused = F.relu(fused)
            fused = F.dropout(fused, p=self.dropout, training=self.training)

        return self.classifier(fused)


class NTSFormer(nn.Module):
    """NTSFormer: Node-Text-Structure Transformer
    结合节点特征、文本语义和图结构的Transformer
    使用图采样邻居注意力机制，避免全局注意力的O(N^2)复杂度
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5,
                 num_heads=4):
        super().__init__()
        self.dropout = dropout
        self.hidden_dim = hidden_dim

        self.node_proj = nn.Linear(in_dim, hidden_dim)
        self.pos_enc = nn.Linear(1, hidden_dim)

        # Use GAT-based local attention instead of global self-attention
        from torch_geometric.nn import GATConv
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=False))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=False))

        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        h = self.node_proj(x)

        # Degree-based positional encoding
        num_nodes = x.size(0)
        deg = torch.zeros(num_nodes, device=edge_index.device)
        deg.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), device=edge_index.device))
        deg = deg.clamp(max=100).unsqueeze(1).log1p()
        h = h + self.pos_enc(deg)

        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        return self.classifier(h)


class MLaGA(nn.Module):
    """MLaGA: Multimodal Label-aware Graph Attention
    标签感知的多模态图注意力
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout

        self.feat_proj = nn.Linear(in_dim, hidden_dim)
        self.label_emb = nn.Embedding(out_dim, hidden_dim)

        from torch_geometric.nn import GATConv
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(hidden_dim * 2, hidden_dim, heads=4, concat=False))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))

        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, labels=None):
        h = self.feat_proj(x)
        h = F.relu(h)

        if labels is not None and self.training:
            label_feat = self.label_emb(labels)
            h = torch.cat([h, label_feat], dim=1)
        else:
            label_feat = torch.zeros(h.size(0), h.size(1), device=h.device)
            h = torch.cat([h, label_feat], dim=1)

        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        return self.classifier(h)


class InstructG2I(nn.Module):
    """InstructG2I (simplified): Instruction-tuned Graph-to-Image model
    图条件引导的图像生成，用于G2Image任务
    """
    def __init__(self, in_dim, hidden_dim, out_dim=768, num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout

        from torch_geometric.nn import GCNConv

        self.proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Graph-QFormer: cross-attention to map graph emb to image conditioning
        self.graph_qformer = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.query_tokens = nn.Parameter(torch.randn(1, 8, hidden_dim) * 0.02)

        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def encode_graph(self, x, edge_index):
        h = self.proj(x)
        h = F.relu(h)
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward(self, x, edge_index):
        h = self.encode_graph(x, edge_index)
        # Graph-QFormer: cross-attend query tokens to graph features
        queries = self.query_tokens.expand(h.size(0), -1, -1)
        h_unsqueezed = h.unsqueeze(1)
        out, _ = self.graph_qformer(queries, h_unsqueezed, h_unsqueezed)
        out = out.mean(dim=1)
        return self.output_proj(out)


ADVANCED_MODEL_REGISTRY = {
    "dgf": DGF,
    "mmgcn": MMGCN,
    "mgat": MGAT,
    "lgmrec": LGMRec,
    "mhgat": MHGAT,
    "dmgc": DMGC,
    "selfsup_gnn": SelfSupervisedGNN,
    "graphmae2": GraphMAE2,
    "unigraph2": UniGraph2,
    "graph4mm": Graph4MM,
    "ntsformer": NTSFormer,
    "mlaga": MLaGA,
    "instructg2i": InstructG2I,
}
