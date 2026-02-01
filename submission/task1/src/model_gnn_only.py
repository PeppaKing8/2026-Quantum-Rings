"""
GNN-only model for iQuHACK 2026 challenge.

Uses only pretrained GNN features as backbone input.
Global features are predicted as an auxiliary task (not used as input),
forcing the GNN to learn circuit-level statistics.

Architecture:
1. Pretrained GNN encoder → mean+max+std pooling → (B, hidden_dim*3)
2. Backbone: [gnn_features, backend, precision, family] → hidden
3. Main heads: threshold classification + runtime regression
4. Auxiliary head: gnn_features → predict global features (MSE)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool, global_max_pool

from .features import NUM_GATE_TYPES


class GNNOnlyModel(nn.Module):
    """
    GNN-only model with auxiliary global feature prediction.

    Input: only circuit DAG (no global features).
    Output: (threshold_logits, log_runtime, predicted_global_features)

    The auxiliary head predicts global features from the GNN pooled output,
    encouraging the GNN to encode circuit statistics.
    """

    def __init__(
        self,
        global_feature_dim=32,
        hidden_dim=64,
        num_gnn_layers=2,
        num_threshold_classes=10,
        gate_embedding_dim=16,
        dropout=0.2,
        use_circuit_family=False,
        num_circuit_families=21,
        circuit_family_embedding_dim=8,
        threshold_embedding_dim=8,
        use_threshold_input=False,
        use_layernorm=True,
        use_residual=True,
    ):
        super().__init__()

        self.num_gnn_layers = num_gnn_layers
        self.use_circuit_family = use_circuit_family
        self.use_threshold_input = use_threshold_input
        self.use_layernorm = use_layernorm
        self.use_residual = use_residual

        # --- GNN encoder (same architecture as QCircuitThresholdModel) ---

        self.gate_embedding = nn.Embedding(NUM_GATE_TYPES + 1, gate_embedding_dim)

        node_feature_dim = gate_embedding_dim + 3

        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(TransformerConv(node_feature_dim, hidden_dim))
        for _ in range(1, num_gnn_layers):
            self.gnn_layers.append(TransformerConv(hidden_dim, hidden_dim))

        if use_layernorm:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim) for _ in range(num_gnn_layers)
            ])

        if use_residual:
            self.input_proj = nn.Linear(node_feature_dim, hidden_dim)

        # mean + max + std pooling
        gnn_output_dim = hidden_dim * 3

        # --- Context embeddings ---

        if use_circuit_family:
            self.circuit_family_embedding = nn.Embedding(num_circuit_families, circuit_family_embedding_dim)
            family_dim = circuit_family_embedding_dim
        else:
            family_dim = 0

        if use_threshold_input:
            self.threshold_embedding = nn.Embedding(9, threshold_embedding_dim)
            thr_emb_dim = threshold_embedding_dim
        else:
            thr_emb_dim = 0

        # --- Backbone (GNN features only, no global features) ---

        total_feature_dim = (
            gnn_output_dim +  # GNN pooled features
            1 +               # Backend bit
            1 +               # Precision bit
            family_dim        # Circuit family embedding
        )

        self.backbone = nn.Sequential(
            nn.Linear(total_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # --- Main task heads ---

        self.threshold_head = nn.Sequential(
            nn.Linear(hidden_dim, num_threshold_classes),
        )

        self.runtime_head = nn.Sequential(
            nn.Linear(hidden_dim + thr_emb_dim, 1),
        )

        # --- Auxiliary head: predict global features from GNN output ---

        self.aux_global_head = nn.Sequential(
            nn.Linear(gnn_output_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, global_feature_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=1.0 / math.sqrt(module.embedding_dim))
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Zero-init final output layers
        nn.init.zeros_(self.threshold_head[-1].weight)
        nn.init.zeros_(self.threshold_head[-1].bias)
        nn.init.zeros_(self.runtime_head[-1].weight)
        nn.init.zeros_(self.runtime_head[-1].bias)

    def _build_node_features(self, data):
        gate_emb = self.gate_embedding(data.gate_type_idx)
        node_features = torch.cat([
            gate_emb,
            data.gate_arity.float().unsqueeze(-1),
            data.is_directional.float().unsqueeze(-1),
            data.gate_index_norm.unsqueeze(-1),
        ], dim=1)
        return node_features

    def _gnn_forward(self, data):
        """Run GNN and return pooled graph-level features."""
        x = self._build_node_features(data)
        edge_index, batch = data.edge_index, data.batch

        if self.use_residual:
            x_proj = self.input_proj(x)

        for i, gnn_layer in enumerate(self.gnn_layers):
            if self.use_residual:
                residual = x_proj if i == 0 else x
            x = gnn_layer(x, edge_index)
            if self.use_layernorm:
                x = self.layer_norms[i](x)
            x = F.silu(x)
            if self.use_residual:
                x = x + residual

        # Mean + max + std pooling
        gnn_mean = global_mean_pool(x, batch)
        gnn_max = global_max_pool(x, batch)
        mean_sq = global_mean_pool(x * x, batch)
        gnn_std = (mean_sq - gnn_mean * gnn_mean).clamp(min=1e-6).sqrt()
        gnn_features = torch.cat([gnn_mean, gnn_max, gnn_std], dim=1)

        return gnn_features

    def forward(self, data):
        """
        Forward pass.

        Args:
            data: PyG Data batch with graph features + global_features (as target)

        Returns:
            (threshold_logits, log_runtime, predicted_global_features)
        """
        disable_family = bool(getattr(self, "disable_family", False))

        gnn_features = self._gnn_forward(data)

        # Backbone input: GNN + context (no global features)
        features = [gnn_features, data.backend_bit.unsqueeze(-1), data.precision_bit.unsqueeze(-1)]
        if self.use_circuit_family:
            family_emb = self.circuit_family_embedding(data.circuit_family_idx)
            features.append(torch.zeros_like(family_emb) if disable_family else family_emb)

        combined_features = torch.cat(features, dim=1)
        final_features = self.backbone(combined_features)

        # Main heads
        threshold_logits = self.threshold_head(final_features)

        if self.use_threshold_input:
            threshold_emb = self.threshold_embedding(data.input_threshold_idx)
            runtime_input = torch.cat([final_features, threshold_emb], dim=1)
        else:
            runtime_input = final_features
        log_runtime = self.runtime_head(runtime_input).squeeze(-1)

        # Auxiliary head: predict global features from GNN output
        predicted_global = self.aux_global_head(gnn_features)

        return threshold_logits, log_runtime, predicted_global


def create_gnn_only_model(config):
    """Factory function to create GNNOnlyModel from config."""
    model = GNNOnlyModel(
        global_feature_dim=config.get('global_feature_dim', 32),
        hidden_dim=config.get('hidden_dim', 64),
        num_gnn_layers=config.get('num_layers', 2),
        num_threshold_classes=config.get('num_threshold_classes', 10),
        gate_embedding_dim=config.get('gate_embedding_dim', 16),
        dropout=config.get('dropout', 0.2),
        use_circuit_family=config.get('use_circuit_family', False),
        num_circuit_families=config.get('num_circuit_families', 21),
        circuit_family_embedding_dim=config.get('circuit_family_embedding_dim', 8),
        threshold_embedding_dim=config.get('threshold_embedding_dim', 8),
        use_threshold_input=config.get('use_threshold_input', False),
        use_layernorm=config.get('use_layernorm', True),
        use_residual=config.get('use_residual', True),
    )
    return model
