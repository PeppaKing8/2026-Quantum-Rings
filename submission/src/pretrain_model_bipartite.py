"""
Bipartite (heterogeneous) GNN pretraining model with masked gate prediction
and local auxiliary tasks.

Architecture:
- Gate nodes: gate_type_embedding + scalar features → gate_input_proj → hidden_dim
- Qubit nodes: degree_norm → qubit_input_proj → hidden_dim
- HeteroConv layers: separate message passing for gate→qubit and qubit→gate
- LayerNorm + residual connections per layer per node type

Pretraining tasks:
1. Masked gate type prediction (CE) — BERT-style 15% masking
2. Graph-level stats prediction (MSE) — predict normalized global features
3. Gate arity prediction (CE) — predict arity of masked gates (also masked)
4. Gate position prediction (MSE) — predict position of masked gates (also masked)
5. Qubit degree prediction (MSE) — predict degree of randomly masked qubits
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, global_mean_pool

from .features import NUM_GATE_TYPES

MASK_TOKEN_IDX = NUM_GATE_TYPES + 1  # index 20 = mask token


class BipartiteGNNPretrainModel(nn.Module):
    """
    Heterogeneous bipartite GNN for masked gate prediction + auxiliary tasks.

    Two node types (gate, qubit) with bidirectional edges.
    SAGEConv for message passing with LayerNorm + residual.
    """

    def __init__(
        self,
        hidden_dim=64,
        num_layers=4,
        gate_embedding_dim=16,
        dropout=0.1,
        mask_ratio=0.15,
        global_feature_dim=0,
        stats_loss_weight=1.0,
        arity_loss_weight=0.5,
        position_loss_weight=0.5,
        degree_loss_weight=0.5,
        qubit_mask_ratio=0.15,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.mask_ratio = mask_ratio
        self.qubit_mask_ratio = qubit_mask_ratio
        self.global_feature_dim = global_feature_dim
        self.stats_loss_weight = stats_loss_weight
        self.arity_loss_weight = arity_loss_weight
        self.position_loss_weight = position_loss_weight
        self.degree_loss_weight = degree_loss_weight

        # Gate embedding: 19 known + 1 unknown + 1 mask = 21
        self.gate_embedding = nn.Embedding(NUM_GATE_TYPES + 2, gate_embedding_dim)

        # Gate input projection: gate_emb + arity + is_directional + index_norm → hidden_dim
        gate_input_dim = gate_embedding_dim + 3
        self.gate_input_proj = nn.Linear(gate_input_dim, hidden_dim)

        # Qubit input projection: degree_norm → hidden_dim
        self.qubit_input_proj = nn.Linear(1, hidden_dim)

        # Heterogeneous GNN layers
        self.convs = nn.ModuleList()
        self.gate_norms = nn.ModuleList()
        self.qubit_norms = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('gate', 'acts_on', 'qubit'): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                ('qubit', 'feeds_into', 'gate'): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
            }, aggr='sum')
            self.convs.append(conv)
            self.gate_norms.append(nn.LayerNorm(hidden_dim))
            self.qubit_norms.append(nn.LayerNorm(hidden_dim))

        # Task 1: Masked gate type prediction head (20 classes)
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, NUM_GATE_TYPES + 1),  # 20 classes (no mask token)
        )

        # Task 2: Graph-level stats prediction head
        if global_feature_dim > 0:
            pool_dim = hidden_dim * 4  # mean+std for gate + qubit
            self.stats_head = nn.Sequential(
                nn.Linear(pool_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, global_feature_dim),
            )

        # Task 3: Gate arity prediction (3 classes: arity 1, 2, 3)
        self.arity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 3),
        )

        # Task 4: Gate position prediction (regression: gate_index_norm)
        self.position_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Task 5: Qubit degree prediction (regression: qubit_degree_norm)
        self.degree_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
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

    def _apply_masking(self, gate_type_idx, gate_arity, gate_index_norm):
        """
        BERT-style masking on gate nodes.

        For masked gates: gate_type → MASK token, gate_arity → 0, gate_index_norm → 0.
        This forces the model to predict all three from graph context alone.

        Returns:
            masked_gate_type_idx, masked_gate_arity, masked_gate_index_norm,
            mask_positions, type_labels, arity_labels, position_labels
        """
        device = gate_type_idx.device
        N = gate_type_idx.size(0)

        rand = torch.rand(N, device=device)
        mask_positions = rand < self.mask_ratio

        if not mask_positions.any():
            mask_positions[torch.randint(N, (1,), device=device)] = True

        # Clone inputs
        masked_gate_type_idx = gate_type_idx.clone()
        masked_gate_arity = gate_arity.clone().float()
        masked_gate_index_norm = gate_index_norm.clone()

        # Gate type masking (BERT-style: 80% mask, 10% random, 10% unchanged)
        mask_rand = torch.rand(N, device=device)

        mask_token_mask = mask_positions & (mask_rand < 0.8)
        masked_gate_type_idx[mask_token_mask] = MASK_TOKEN_IDX

        random_mask = mask_positions & (mask_rand >= 0.8) & (mask_rand < 0.9)
        if random_mask.any():
            random_gates = torch.randint(0, NUM_GATE_TYPES, (random_mask.sum(),), device=device)
            masked_gate_type_idx[random_mask] = random_gates

        # Also mask arity and position for masked gates (set to 0)
        masked_gate_arity[mask_positions] = 0.0
        masked_gate_index_norm[mask_positions] = 0.0

        # Labels: -100 for non-masked positions (ignored by CE/MSE)
        type_labels = torch.full((N,), -100, dtype=torch.long, device=device)
        type_labels[mask_positions] = gate_type_idx[mask_positions]

        # Arity labels: class 0=arity1, 1=arity2, 2=arity3
        arity_labels = torch.full((N,), -100, dtype=torch.long, device=device)
        arity_labels[mask_positions] = gate_arity[mask_positions] - 1  # shift to 0-indexed

        # Position labels: continuous, only for masked positions
        position_labels = torch.full((N,), -1.0, device=device)
        position_labels[mask_positions] = gate_index_norm[mask_positions]

        return (masked_gate_type_idx, masked_gate_arity, masked_gate_index_norm,
                mask_positions, type_labels, arity_labels, position_labels)

    def _apply_qubit_masking(self, qubit_degree_norm):
        """Randomly mask qubit degree features. Returns masked input and labels."""
        device = qubit_degree_norm.device
        N = qubit_degree_norm.size(0)

        rand = torch.rand(N, device=device)
        mask_positions = rand < self.qubit_mask_ratio

        if not mask_positions.any():
            mask_positions[torch.randint(N, (1,), device=device)] = True

        masked_degree = qubit_degree_norm.clone()
        masked_degree[mask_positions] = 0.0

        # Labels: only for masked positions
        degree_labels = torch.full((N,), -1.0, device=device)
        degree_labels[mask_positions] = qubit_degree_norm[mask_positions]

        return masked_degree, mask_positions, degree_labels

    def forward(self, data):
        """
        Forward pass with masked prediction on bipartite graph.

        Returns:
            (total_loss, loss_dict, accuracy) tuple
            loss_dict has: mask_loss, stats_loss, arity_loss, position_loss, degree_loss
        """
        # Apply masking on gate nodes (type + arity + position)
        gate_type_idx = data['gate'].gate_type_idx
        gate_arity = data['gate'].gate_arity
        gate_index_norm = data['gate'].gate_index_norm

        (masked_type, masked_arity, masked_position,
         gate_mask, type_labels, arity_labels, position_labels) = \
            self._apply_masking(gate_type_idx, gate_arity, gate_index_norm)

        # Apply masking on qubit nodes (degree)
        qubit_degree_norm = data['qubit'].qubit_degree_norm
        masked_degree, qubit_mask, degree_labels = self._apply_qubit_masking(qubit_degree_norm)

        # Build gate node features (with masked values)
        gate_emb = self.gate_embedding(masked_type)
        gate_x = torch.cat([
            gate_emb,
            masked_arity.unsqueeze(-1),
            data['gate'].gate_is_directional.float().unsqueeze(-1),
            masked_position.unsqueeze(-1),
        ], dim=1)
        gate_x = self.gate_input_proj(gate_x)

        # Build qubit node features (with masked degree)
        qubit_x = self.qubit_input_proj(masked_degree.unsqueeze(-1))

        x_dict = {'gate': gate_x, 'qubit': qubit_x}
        edge_index_dict = {
            ('gate', 'acts_on', 'qubit'): data['gate', 'acts_on', 'qubit'].edge_index,
            ('qubit', 'feeds_into', 'gate'): data['qubit', 'feeds_into', 'gate'].edge_index,
        }

        # Heterogeneous GNN forward with LayerNorm + residual
        for conv, gate_norm, qubit_norm in zip(self.convs, self.gate_norms, self.qubit_norms):
            gate_residual = x_dict['gate']
            qubit_residual = x_dict['qubit']

            x_dict_new = conv(x_dict, edge_index_dict)

            x_dict = {
                'gate': gate_norm(F.silu(x_dict_new['gate'])) + gate_residual,
                'qubit': qubit_norm(F.silu(x_dict_new['qubit'])) + qubit_residual,
            }

        gate_out = x_dict['gate']
        qubit_out = x_dict['qubit']

        # === Task 1: Masked gate type prediction ===
        type_logits = self.prediction_head(gate_out)
        mask_loss = F.cross_entropy(type_logits, type_labels, ignore_index=-100)

        with torch.no_grad():
            masked_logits = type_logits[gate_mask]
            masked_labels = type_labels[gate_mask]
            if masked_labels.numel() > 0:
                preds = masked_logits.argmax(dim=1)
                accuracy = (preds == masked_labels).float().mean().item()
            else:
                accuracy = 0.0

        # === Task 2: Graph-level stats prediction ===
        stats_loss = torch.tensor(0.0, device=gate_out.device)
        if self.global_feature_dim > 0 and hasattr(data, 'global_features'):
            gate_batch = data['gate'].batch
            qubit_batch = data['qubit'].batch

            gate_mean = global_mean_pool(gate_out, gate_batch)
            gate_mean_sq = global_mean_pool(gate_out * gate_out, gate_batch)
            gate_std = (gate_mean_sq - gate_mean * gate_mean).clamp(min=1e-6).sqrt()

            qubit_mean = global_mean_pool(qubit_out, qubit_batch)
            qubit_mean_sq = global_mean_pool(qubit_out * qubit_out, qubit_batch)
            qubit_std = (qubit_mean_sq - qubit_mean * qubit_mean).clamp(min=1e-6).sqrt()

            pooled = torch.cat([gate_mean, gate_std, qubit_mean, qubit_std], dim=1)
            predicted_stats = self.stats_head(pooled)
            target_stats = data.global_features.reshape(-1, self.global_feature_dim)
            stats_loss = F.mse_loss(predicted_stats, target_stats)

        # === Task 3: Gate arity prediction (on masked gates) ===
        arity_logits = self.arity_head(gate_out)
        arity_loss = F.cross_entropy(arity_logits, arity_labels, ignore_index=-100)

        # === Task 4: Gate position prediction (on masked gates) ===
        position_pred = self.position_head(gate_out).squeeze(-1)
        pos_mask = position_labels >= 0
        if pos_mask.any():
            position_loss = F.mse_loss(position_pred[pos_mask], position_labels[pos_mask])
        else:
            position_loss = torch.tensor(0.0, device=gate_out.device)

        # === Task 5: Qubit degree prediction (on masked qubits) ===
        degree_pred = self.degree_head(qubit_out).squeeze(-1)
        deg_mask = degree_labels >= 0
        if deg_mask.any():
            degree_loss = F.mse_loss(degree_pred[deg_mask], degree_labels[deg_mask])
        else:
            degree_loss = torch.tensor(0.0, device=gate_out.device)

        # Total loss
        total_loss = (
            mask_loss
            + self.stats_loss_weight * stats_loss
            + self.arity_loss_weight * arity_loss
            + self.position_loss_weight * position_loss
            + self.degree_loss_weight * degree_loss
        )

        loss_dict = {
            'mask': mask_loss.item(),
            'stats': stats_loss.item(),
            'arity': arity_loss.item(),
            'position': position_loss.item() if isinstance(position_loss, torch.Tensor) else position_loss,
            'degree': degree_loss.item() if isinstance(degree_loss, torch.Tensor) else degree_loss,
        }

        return total_loss, loss_dict, accuracy

    def get_gnn_state_dict(self):
        """
        Extract GNN weights for transfer.

        Returns state dict with:
        - gate_embedding (trimmed to 20 rows, dropping mask token)
        - gate_input_proj, qubit_input_proj
        - convs (HeteroConv layers)
        - gate_norms, qubit_norms
        """
        state_dict = {}

        # Gate embedding: trim from 21 to 20 rows
        emb_weight = self.gate_embedding.weight.data[:NUM_GATE_TYPES + 1]
        state_dict['gate_embedding.weight'] = emb_weight

        for k, v in self.gate_input_proj.state_dict().items():
            state_dict[f'gate_input_proj.{k}'] = v

        for k, v in self.qubit_input_proj.state_dict().items():
            state_dict[f'qubit_input_proj.{k}'] = v

        for k, v in self.convs.state_dict().items():
            state_dict[f'convs.{k}'] = v

        for k, v in self.gate_norms.state_dict().items():
            state_dict[f'gate_norms.{k}'] = v

        for k, v in self.qubit_norms.state_dict().items():
            state_dict[f'qubit_norms.{k}'] = v

        return state_dict
