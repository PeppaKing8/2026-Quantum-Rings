"""
Bipartite (heterogeneous) GNN model for iQuHACK 2026 challenge.

Same task as QCircuitThresholdModel but uses a bipartite graph representation:
- Gate nodes: quantum operations with learned embeddings
- Qubit nodes: state carriers with degree features
- HeteroConv (SAGEConv) for message passing between node types
- Pools both gate and qubit representations for graph-level features
- Optional FiLM conditioning: circuit family modulates backbone features
- Optional family residual heads: family-specific baselines for threshold + runtime
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, global_mean_pool

from .features import NUM_GATE_TYPES


class QCircuitBipartiteModel(nn.Module):
    """
    Bipartite GNN model for threshold classification + runtime regression.

    Gate nodes: gate_type_embedding + arity + is_directional + gate_index_norm
    Qubit nodes: qubit_degree_norm
    Pooling: mean+std for both gate and qubit nodes (4 * hidden_dim total)
    """

    def __init__(
        self,
        global_feature_dim=32,
        hidden_dim=64,
        num_gnn_layers=4,
        num_threshold_classes=10,
        gate_embedding_dim=16,
        dropout=0.2,
        use_circuit_family=False,
        num_circuit_families=21,
        circuit_family_embedding_dim=8,
        family_hidden_dim=0,
        use_film=False,
        use_family_residual_heads=False,
        threshold_embedding_dim=8,
        use_threshold_input=False,
        use_global_features=True,
    ):
        super().__init__()

        self.use_circuit_family = use_circuit_family
        self.use_threshold_input = use_threshold_input
        self.use_global_features = use_global_features
        self.use_film = use_film and use_circuit_family
        self.use_family_residual_heads = use_family_residual_heads and use_circuit_family

        # Gate embedding: 19 known + 1 unknown = 20
        self.gate_embedding = nn.Embedding(NUM_GATE_TYPES + 1, gate_embedding_dim)

        # Input projections
        gate_input_dim = gate_embedding_dim + 3  # emb + arity + directional + index_norm
        self.gate_input_proj = nn.Linear(gate_input_dim, hidden_dim)
        self.qubit_input_proj = nn.Linear(1, hidden_dim)  # degree_norm

        # Heterogeneous GNN layers
        self.convs = nn.ModuleList()
        self.gate_norms = nn.ModuleList()
        self.qubit_norms = nn.ModuleList()
        for _ in range(num_gnn_layers):
            conv = HeteroConv({
                ('gate', 'acts_on', 'qubit'): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                ('qubit', 'feeds_into', 'gate'): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
            }, aggr='sum')
            self.convs.append(conv)
            self.gate_norms.append(nn.LayerNorm(hidden_dim))
            self.qubit_norms.append(nn.LayerNorm(hidden_dim))

        # Pooling: mean+std for gate nodes + mean+std for qubit nodes
        gnn_output_dim = hidden_dim * 4

        # Global feature processor (optional)
        if use_global_features:
            self.global_mlp = nn.Sequential(
                nn.Linear(global_feature_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
            global_dim = hidden_dim
        else:
            global_dim = 0

        # Circuit family embedding + processing
        family_concat_dim = 0
        if use_circuit_family:
            self.circuit_family_embedding = nn.Embedding(
                num_circuit_families, circuit_family_embedding_dim
            )
            fh_dim = family_hidden_dim if family_hidden_dim > 0 else circuit_family_embedding_dim
            self.family_mlp = nn.Sequential(
                nn.Linear(circuit_family_embedding_dim, fh_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            )

            if self.use_film:
                # FiLM: generate gamma and beta to modulate backbone output
                # gamma initialized near 0 so (1+gamma)*x + beta ≈ x at init
                self.film_generator = nn.Linear(fh_dim, hidden_dim * 2)
            else:
                # Plain concat: family features go into backbone input
                family_concat_dim = fh_dim

            if self.use_family_residual_heads:
                # Family-specific baselines added to head outputs
                self.family_threshold_head = nn.Linear(fh_dim, num_threshold_classes)
                self.family_runtime_head = nn.Linear(fh_dim, 1)

        # Threshold embedding for runtime head (optional)
        if use_threshold_input:
            self.threshold_embedding = nn.Embedding(9, threshold_embedding_dim)
            thr_emb_dim = threshold_embedding_dim
        else:
            thr_emb_dim = 0

        # Total feature dim: GNN pool + [global] + backend + precision + [family_concat]
        total_feature_dim = (
            gnn_output_dim +
            global_dim +
            1 +             # backend bit
            1 +             # precision bit
            family_concat_dim
        )

        # Backbone
        self.backbone = nn.Sequential(
            nn.Linear(total_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Heads
        self.threshold_head = nn.Linear(hidden_dim, num_threshold_classes)
        self.runtime_head = nn.Linear(hidden_dim + thr_emb_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=1.0 / math.sqrt(module.embedding_dim))
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Zero-init heads for neutral initial predictions
        nn.init.zeros_(self.threshold_head.weight)
        nn.init.zeros_(self.threshold_head.bias)
        nn.init.zeros_(self.runtime_head.weight)
        nn.init.zeros_(self.runtime_head.bias)

        # Zero-init FiLM so it starts as identity: (1+0)*x + 0 = x
        if self.use_film:
            nn.init.zeros_(self.film_generator.weight)
            nn.init.zeros_(self.film_generator.bias)

        # Zero-init family residual heads so they start neutral
        if self.use_family_residual_heads:
            nn.init.zeros_(self.family_threshold_head.weight)
            nn.init.zeros_(self.family_threshold_head.bias)
            nn.init.zeros_(self.family_runtime_head.weight)
            nn.init.zeros_(self.family_runtime_head.bias)

    def forward(self, data):
        """
        Forward pass on bipartite HeteroData batch.

        Returns:
            tuple: (threshold_logits [B, 10], log_runtime [B])
        """
        disable_family = bool(getattr(self, "disable_family", False))

        # Build gate node features
        gate_emb = self.gate_embedding(data['gate'].gate_type_idx)
        gate_x = torch.cat([
            gate_emb,
            data['gate'].gate_arity.float().unsqueeze(-1),
            data['gate'].gate_is_directional.float().unsqueeze(-1),
            data['gate'].gate_index_norm.unsqueeze(-1),
        ], dim=1)
        gate_x = self.gate_input_proj(gate_x)

        # Build qubit node features
        qubit_x = self.qubit_input_proj(data['qubit'].qubit_degree_norm.unsqueeze(-1))

        x_dict = {'gate': gate_x, 'qubit': qubit_x}
        edge_index_dict = {
            ('gate', 'acts_on', 'qubit'): data['gate', 'acts_on', 'qubit'].edge_index,
            ('qubit', 'feeds_into', 'gate'): data['qubit', 'feeds_into', 'gate'].edge_index,
        }

        # HeteroGNN forward with LayerNorm + residual
        for conv, gate_norm, qubit_norm in zip(self.convs, self.gate_norms, self.qubit_norms):
            gate_residual = x_dict['gate']
            qubit_residual = x_dict['qubit']

            x_dict_new = conv(x_dict, edge_index_dict)

            x_dict = {
                'gate': gate_norm(F.silu(x_dict_new['gate'])) + gate_residual,
                'qubit': qubit_norm(F.silu(x_dict_new['qubit'])) + qubit_residual,
            }

        # Pool: mean+std for each node type
        gate_batch = data['gate'].batch
        qubit_batch = data['qubit'].batch
        gate_out = x_dict['gate']
        qubit_out = x_dict['qubit']

        gate_mean = global_mean_pool(gate_out, gate_batch)
        gate_mean_sq = global_mean_pool(gate_out * gate_out, gate_batch)
        gate_std = (gate_mean_sq - gate_mean * gate_mean).clamp(min=1e-6).sqrt()

        qubit_mean = global_mean_pool(qubit_out, qubit_batch)
        qubit_mean_sq = global_mean_pool(qubit_out * qubit_out, qubit_batch)
        qubit_std = (qubit_mean_sq - qubit_mean * qubit_mean).clamp(min=1e-6).sqrt()

        gnn_features = torch.cat([gate_mean, gate_std, qubit_mean, qubit_std], dim=1)

        # Process family embedding (shared by FiLM, residual heads, and/or concat)
        family_h = None
        if self.use_circuit_family:
            family_emb = self.circuit_family_embedding(data.circuit_family_idx)
            family_h = self.family_mlp(family_emb)
        family_h_for_concat = None
        if family_h is not None:
            family_h_for_concat = torch.zeros_like(family_h) if disable_family else family_h

        # Combine: GNN + [global] + backend + precision + [family_concat]
        features = [gnn_features]
        if self.use_global_features:
            features.append(self.global_mlp(data.global_features))
        features.extend([data.backend_bit.unsqueeze(-1), data.precision_bit.unsqueeze(-1)])
        if self.use_circuit_family and not self.use_film:
            # Plain concat path (no FiLM)
            features.append(family_h_for_concat)

        combined = torch.cat(features, dim=1)

        # Backbone
        final_features = self.backbone(combined)

        # FiLM conditioning: family modulates backbone output
        if (not disable_family) and self.use_film:
            film_params = self.film_generator(family_h)
            gamma, beta = film_params.chunk(2, dim=-1)
            final_features = (1 + gamma) * final_features + beta

        # Heads
        threshold_logits = self.threshold_head(final_features)

        if self.use_threshold_input:
            threshold_emb = self.threshold_embedding(data.input_threshold_idx)
            runtime_input = torch.cat([final_features, threshold_emb], dim=1)
        else:
            runtime_input = final_features
        log_runtime = self.runtime_head(runtime_input).squeeze(-1)

        # Family residual heads: add family-specific baselines
        if (not disable_family) and self.use_family_residual_heads:
            threshold_logits = threshold_logits + self.family_threshold_head(family_h)
            log_runtime = log_runtime + self.family_runtime_head(family_h).squeeze(-1)

        return threshold_logits, log_runtime


def create_bipartite_model(config):
    """Factory function to create bipartite model from config."""
    return QCircuitBipartiteModel(
        global_feature_dim=config.get('global_feature_dim', 32),
        hidden_dim=config.get('hidden_dim', 64),
        num_gnn_layers=config.get('num_layers', 4),
        num_threshold_classes=config.get('num_threshold_classes', 10),
        gate_embedding_dim=config.get('gate_embedding_dim', 16),
        dropout=config.get('dropout', 0.2),
        use_circuit_family=config.get('use_circuit_family', False),
        num_circuit_families=config.get('num_circuit_families', 21),
        circuit_family_embedding_dim=config.get('circuit_family_embedding_dim', 8),
        family_hidden_dim=config.get('family_hidden_dim', 0),
        use_film=config.get('use_film', False),
        use_family_residual_heads=config.get('use_family_residual_heads', False),
        threshold_embedding_dim=config.get('threshold_embedding_dim', 8),
        use_threshold_input=config.get('use_threshold_input', False),
        use_global_features=config.get('use_global_features', True),
    )
