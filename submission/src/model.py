"""
Neural network model for iQuHACK 2026 challenge.

Threshold classification with permutation-invariant GNN:
1. Learned gate embeddings + scalar gate properties for node features (no qubit indices)
2. GNN (TransformerConv) with mean+max+std pooling for circuit DAG
3. Global feature MLP for circuit statistics
4. Shared backbone
5. Two heads: threshold classification (10 classes) and runtime regression
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GPSConv, global_mean_pool

from .features import NUM_GATE_TYPES


class QCircuitThresholdModel(nn.Module):
    """
    Threshold classification model with learned node embeddings.

    Node features = concat(gate_emb, gate_arity, is_directional, gate_index_norm)
    - gate_emb: learned embedding per gate type
    - gate_arity: number of qubits (scalar)
    - is_directional: whether gate is directional (scalar)
    - gate_index_norm: normalized position in circuit (scalar)

    Pooling: mean + max + std for richer graph-level representation.

    Output: (threshold_logits [B, 10], log_runtime [B])
    """

    def __init__(
        self,
        global_feature_dim=22,
        hidden_dim=64,
        num_gnn_layers=2,
        use_graph_features=True,
        num_threshold_classes=10,
        gate_embedding_dim=16,
        dropout=0.2,
        use_circuit_family=False,
        num_circuit_families=21,
        circuit_family_embedding_dim=8,
        family_hidden_dim=None,
        use_film=False,
        use_family_residual_heads=False,
        threshold_embedding_dim=8,
        use_threshold_input=False,
        use_layernorm=False,
        use_residual=False,
        gnn_type='transformer_conv',
        gps_heads=4,
        attn_type='multihead',
    ):
        super().__init__()

        self.use_graph_features = use_graph_features
        self.num_gnn_layers = num_gnn_layers
        self.use_circuit_family = use_circuit_family
        self.use_film = use_film
        self.use_family_residual_heads = use_family_residual_heads
        self.use_threshold_input = use_threshold_input
        self.use_layernorm = use_layernorm
        self.use_residual = use_residual
        self.gnn_type = gnn_type

        # Default family_hidden_dim to match circuit_family_embedding_dim
        if family_hidden_dim is None:
            family_hidden_dim = circuit_family_embedding_dim

        if use_graph_features:
            # Learned gate type embedding
            self.gate_embedding = nn.Embedding(NUM_GATE_TYPES + 1, gate_embedding_dim)  # +1 for unknown

            # Node feature dim: gate_emb + gate_arity + is_directional + gate_index_norm
            node_feature_dim = gate_embedding_dim + 3

            # GNN layers
            self.gnn_layers = nn.ModuleList()
            if gnn_type == 'gps':
                # GPS: input_proj required, GPSConv handles norms/residuals internally
                self.input_proj = nn.Linear(node_feature_dim, hidden_dim)
                for _ in range(num_gnn_layers):
                    local_conv = TransformerConv(hidden_dim, hidden_dim)
                    gps_layer = GPSConv(
                        channels=hidden_dim,
                        conv=local_conv,
                        heads=gps_heads,
                        dropout=dropout,
                        act='silu',
                        norm='layer_norm',
                        attn_type=attn_type,
                    )
                    self.gnn_layers.append(gps_layer)
            else:
                # Plain TransformerConv
                self.gnn_layers.append(TransformerConv(node_feature_dim, hidden_dim))
                for _ in range(1, num_gnn_layers):
                    self.gnn_layers.append(TransformerConv(hidden_dim, hidden_dim))

                # Optional LayerNorm after each GNN layer (for pretrained weight compatibility)
                if use_layernorm:
                    self.layer_norms = nn.ModuleList([
                        nn.LayerNorm(hidden_dim) for _ in range(num_gnn_layers)
                    ])

                # Optional input projection for residual connections
                if use_residual:
                    self.input_proj = nn.Linear(node_feature_dim, hidden_dim)

            # mean + std pooling (max pooling requires torch-scatter)
            gnn_output_dim = hidden_dim * 2
        else:
            gnn_output_dim = 0

        # Global feature processor
        self.global_mlp = nn.Sequential(
            nn.Linear(global_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Circuit family embedding with processing MLP (IMPORTANT FEATURE)
        if use_circuit_family:
            self.circuit_family_embedding = nn.Embedding(num_circuit_families, circuit_family_embedding_dim)
            
            # Family processing MLP: transform embedding to rich representation
            self.family_mlp = nn.Sequential(
                nn.Linear(circuit_family_embedding_dim, family_hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(family_hidden_dim, family_hidden_dim),
                nn.SiLU(),
            )
            family_dim = family_hidden_dim
            
            # FiLM: family produces (gamma, beta) to modulate backbone output (scale + shift)
            if use_film:
                self.film_gamma = nn.Linear(family_hidden_dim, hidden_dim)
                self.film_beta = nn.Linear(family_hidden_dim, hidden_dim)
            
            # Family residual heads: predict family-specific baseline added to main output
            if use_family_residual_heads:
                self.family_threshold_residual = nn.Linear(family_hidden_dim, num_threshold_classes)
                self.family_runtime_residual = nn.Linear(family_hidden_dim, 1)
        else:
            family_dim = 0

        # Threshold embedding for runtime head (optional)
        if use_threshold_input:
            self.threshold_embedding = nn.Embedding(9, threshold_embedding_dim)  # 9 threshold levels
            thr_emb_dim = threshold_embedding_dim
        else:
            thr_emb_dim = 0

        # Total feature dimension: GNN + global + backend + precision + family
        total_feature_dim = (
            gnn_output_dim +  # GNN features
            hidden_dim +      # Global features (processed)
            1 +               # Backend bit
            1 +               # Precision bit
            family_dim        # Circuit family embedding
        )

        self.threshold_shortcut = nn.Linear(
            global_feature_dim, num_threshold_classes, bias=False
        )

        # Backbone MLP: features -> final features
        self.backbone = nn.Sequential(
            nn.Linear(total_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Threshold classification head (simple MLP)
        self.threshold_head = nn.Sequential(
            nn.Linear(hidden_dim, num_threshold_classes),
        )

        # Runtime regression head (optionally takes threshold embedding)
        self.runtime_head = nn.Sequential(
            nn.Linear(hidden_dim + thr_emb_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """
        Custom weight initialization:
        - Embeddings: N(0, 1/sqrt(dim)) to keep magnitudes reasonable
        - Linear layers: Xavier uniform (appropriate for SiLU near origin)
        - Head output layers: zero-init so initial predictions are
          uniform logits (threshold) and near-zero (runtime in log-space)
        """
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=1.0 / math.sqrt(module.embedding_dim))
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Zero-init final output layers so initial predictions are neutral
        nn.init.zeros_(self.threshold_head[-1].weight)
        nn.init.zeros_(self.threshold_head[-1].bias)
        nn.init.zeros_(self.runtime_head[-1].weight)
        nn.init.zeros_(self.runtime_head[-1].bias)

        # Zero-init family residual heads so they start as no-op
        if getattr(self, "family_threshold_residual", None) is not None:
            nn.init.zeros_(self.family_threshold_residual.weight)
            nn.init.zeros_(self.family_threshold_residual.bias)
        if getattr(self, "family_runtime_residual", None) is not None:
            nn.init.zeros_(self.family_runtime_residual.weight)
            nn.init.zeros_(self.family_runtime_residual.bias)

    def _build_node_features(self, data):
        """
        Build node features from gate properties (permutation-invariant).

        node_feature = concat(gate_emb, gate_arity, is_directional, gate_index_norm)

        No qubit index embeddings — fully invariant to qubit relabeling.
        """
        # Gate embedding: (N, gate_emb_dim)
        gate_emb = self.gate_embedding(data.gate_type_idx)

        # Scalar features: arity, directionality, normalized position
        node_features = torch.cat([
            gate_emb,
            data.gate_arity.float().unsqueeze(-1),
            data.is_directional.float().unsqueeze(-1),
            data.gate_index_norm.unsqueeze(-1),
        ], dim=1)

        return node_features

    def forward(self, data):
        """
        Forward pass.

        Args:
            data: PyG Data batch with:
                - gate_type_idx, qubit_indices, is_directional, gate_arity,
                  gate_index_norm, edge_index (if use_graph_features)
                - global_features
                - backend_bit, precision_bit
                - input_threshold_idx (for runtime head)
                - circuit_family_idx (if use_circuit_family)

        Returns:
            tuple: (threshold_logits [B, 10], log_runtime [B])
        """
        disable_family = bool(getattr(self, "disable_family", False))

        # GNN features
        if self.use_graph_features:
            x = self._build_node_features(data)
            edge_index, batch = data.edge_index, data.batch

            if self.gnn_type == 'gps':
                # GPS: project to hidden_dim, then GPSConv layers
                x = self.input_proj(x)
                for gps_layer in self.gnn_layers:
                    x = gps_layer(x, edge_index, batch)
            else:
                # Plain TransformerConv with optional LayerNorm + residual
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

            # Mean + std pooling (no max pooling — requires torch-scatter)
            gnn_mean = global_mean_pool(x, batch)   # (batch_size, hidden_dim)
            # Std pooling: sqrt(E[x^2] - E[x]^2)
            mean_sq = global_mean_pool(x * x, batch)
            gnn_std = (mean_sq - gnn_mean * gnn_mean).clamp(min=1e-6).sqrt()
            gnn_features = torch.cat([gnn_mean, gnn_std], dim=1)  # (batch_size, hidden_dim * 2)
        else:
            gnn_features = None

        # Global features
        global_features = self.global_mlp(data.global_features)

        # Family features (processed through MLP for richer representation)
        if self.use_circuit_family:
            family_emb = self.circuit_family_embedding(data.circuit_family_idx)
            family_features = self.family_mlp(family_emb)  # Process family embedding
        else:
            family_features = None
        family_features_for_concat = None
        if family_features is not None:
            family_features_for_concat = torch.zeros_like(family_features) if disable_family else family_features

        # Feature layer: concat(graph, global, backend, precision, family)
        features = [global_features, data.backend_bit.unsqueeze(-1), data.precision_bit.unsqueeze(-1)]
        if gnn_features is not None:
            features.insert(0, gnn_features)
        if family_features_for_concat is not None:
            features.append(family_features_for_concat)

        combined_features = torch.cat(features, dim=1)

        # Backbone: features -> final features
        final_features = self.backbone(combined_features)

        # FiLM: family modulates backbone output (gamma * x + beta)
        if (not disable_family) and self.use_circuit_family and self.use_film and family_features is not None:
            gamma = self.film_gamma(family_features)
            beta = self.film_beta(family_features)
            final_features = gamma * final_features + beta

        # Threshold head (task 1)
        threshold_logits = self.threshold_head(final_features) + self.threshold_shortcut(data.global_features)
        if (not disable_family) and self.use_circuit_family and self.use_family_residual_heads and family_features is not None:
            threshold_logits = threshold_logits + self.family_threshold_residual(family_features)

        # Runtime head (task 2): final features + optional threshold embedding
        if self.use_threshold_input:
            threshold_emb = self.threshold_embedding(data.input_threshold_idx)
            runtime_input = torch.cat([final_features, threshold_emb], dim=1)
        else:
            runtime_input = final_features
        log_runtime = self.runtime_head(runtime_input).squeeze(-1)
        if (not disable_family) and self.use_circuit_family and self.use_family_residual_heads and family_features is not None:
            log_runtime = log_runtime + self.family_runtime_residual(family_features).squeeze(-1)

        return threshold_logits, log_runtime


def create_model(config):
    """
    Factory function to create model from config.

    Args:
        config: dict with model configuration

    Returns:
        QCircuitThresholdModel
    """
    model = QCircuitThresholdModel(
        global_feature_dim=config.get('global_feature_dim', 22),
        hidden_dim=config.get('hidden_dim', 64),
        num_gnn_layers=config.get('num_layers', 2),
        use_graph_features=config.get('use_graph_features', True),
        num_threshold_classes=config.get('num_threshold_classes', 10),
        gate_embedding_dim=config.get('gate_embedding_dim', 16),
        dropout=config.get('dropout', 0.2),
        use_circuit_family=config.get('use_circuit_family', False),
        num_circuit_families=config.get('num_circuit_families', 21),
        circuit_family_embedding_dim=config.get('circuit_family_embedding_dim', 8),
        family_hidden_dim=config.get('family_hidden_dim'),
        use_film=config.get('use_film', False),
        use_family_residual_heads=config.get('use_family_residual_heads', False),
        threshold_embedding_dim=config.get('threshold_embedding_dim', 8),
        use_threshold_input=config.get('use_threshold_input', False),
        use_layernorm=config.get('use_layernorm', False),
        use_residual=config.get('use_residual', False),
        gnn_type=config.get('gnn_type', 'transformer_conv'),
        gps_heads=config.get('gps_heads', 4),
        attn_type=config.get('attn_type', 'multihead'),
    )

    return model
