"""
Neural network model for algorithm/family classification.

GNN + global feature backbone with a single classification head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool

from .features import NUM_GATE_TYPES


class QCircuitAlgorithmModel(nn.Module):
    """
    Algorithm classification model with learned node embeddings.

    Output: logits [B, num_classes]
    """

    def __init__(
        self,
        global_feature_dim=28,
        hidden_dim=64,
        num_gnn_layers=2,
        use_graph_features=True,
        num_classes=2,
        gate_embedding_dim=16,
        qp_embedding_dim=16,
        max_gate_arity=3,
        max_qubits=150,
    ):
        super().__init__()

        self.use_graph_features = use_graph_features
        self.num_gnn_layers = num_gnn_layers
        self.max_gate_arity = max_gate_arity
        self.qp_embedding_dim = qp_embedding_dim

        if use_graph_features:
            self.gate_embedding = nn.Embedding(NUM_GATE_TYPES + 1, gate_embedding_dim)
            self.qubit_embedding = nn.Embedding(max_qubits, qp_embedding_dim)
            self.position_embedding = nn.Embedding(max_gate_arity, qp_embedding_dim)

            node_feature_dim = gate_embedding_dim + qp_embedding_dim + 1

            self.gnn_layers = nn.ModuleList()
            self.gnn_layers.append(TransformerConv(node_feature_dim, hidden_dim))
            for _ in range(1, num_gnn_layers):
                self.gnn_layers.append(TransformerConv(hidden_dim, hidden_dim))

            gnn_output_dim = hidden_dim
        else:
            gnn_output_dim = 0

        self.global_mlp = nn.Sequential(
            nn.Linear(global_feature_dim, hidden_dim),
            nn.SiLU(),
        )

        total_feature_dim = gnn_output_dim + hidden_dim

        self.backbone = nn.Sequential(
            nn.Linear(total_feature_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
        )

        self.class_head = nn.Linear(hidden_dim, num_classes)

    def _build_node_features(self, data):
        device = data.gate_type_idx.device
        num_nodes = data.gate_type_idx.size(0)

        gate_emb = self.gate_embedding(data.gate_type_idx)
        qubit_embs = self.qubit_embedding(data.qubit_indices)

        slot_indices = torch.arange(self.max_gate_arity, device=device)
        pos_embs = self.position_embedding(slot_indices)
        pos_embs_expanded = pos_embs.unsqueeze(0).expand(num_nodes, -1, -1)

        const_pos = torch.full_like(pos_embs_expanded, 1.0 / self.qp_embedding_dim)
        dir_mask = data.is_directional.view(num_nodes, 1, 1)
        pos_encoding = torch.where(dir_mask, pos_embs_expanded, const_pos)

        qp_features = qubit_embs * pos_encoding

        arity_range = torch.arange(self.max_gate_arity, device=device).unsqueeze(0)
        arity_mask = arity_range < data.gate_arity.unsqueeze(1)
        qp_features = qp_features * arity_mask.unsqueeze(-1).float()

        qp_sum = qp_features.sum(dim=1)

        node_features = torch.cat(
            [gate_emb, qp_sum, data.gate_index_norm.unsqueeze(-1)], dim=1
        )

        return node_features

    def forward(self, data):
        if self.use_graph_features:
            x = self._build_node_features(data)
            edge_index, batch = data.edge_index, data.batch

            for gnn_layer in self.gnn_layers:
                x = gnn_layer(x, edge_index)
                x = F.silu(x)

            gnn_features = global_mean_pool(x, batch)
        else:
            gnn_features = None

        global_features = self.global_mlp(data.global_features)

        features = [global_features]
        if gnn_features is not None:
            features.insert(0, gnn_features)

        combined_features = torch.cat(features, dim=1)
        backbone_output = self.backbone(combined_features)
        logits = self.class_head(backbone_output)

        return logits


def create_algorithm_model(config):
    """Factory function to create algorithm classifier from config."""
    model = QCircuitAlgorithmModel(
        global_feature_dim=config.get("global_feature_dim", 28),
        hidden_dim=config.get("hidden_dim", 64),
        num_gnn_layers=config.get("num_layers", 2),
        use_graph_features=config.get("use_graph_features", True),
        num_classes=config.get("num_classes", 2),
        gate_embedding_dim=config.get("gate_embedding_dim", 16),
        qp_embedding_dim=config.get("qp_embedding_dim", 16),
        max_gate_arity=config.get("max_gate_arity", 3),
        max_qubits=config.get("max_qubits", 150),
    )
    return model
