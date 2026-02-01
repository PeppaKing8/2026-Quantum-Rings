"""
GNN pretraining model with masked gate prediction.

BERT-style masking: 15% of gate tokens are corrupted
  - 80% replaced with [MASK] token
  - 10% replaced with random gate type
  - 10% kept unchanged

The model predicts the original gate type at masked positions.

Supports two GNN backends:
  - 'transformer_conv': plain TransformerConv + LayerNorm + residual (original)
  - 'gps': GPSConv (local TransformerConv + global multi-head attention)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GPSConv, global_mean_pool

from .features import NUM_GATE_TYPES

MASK_TOKEN_IDX = NUM_GATE_TYPES + 1  # index 20 = mask token (19 = unknown)


class GNNPretrainModel(nn.Module):
    """
    GNN encoder with masked gate prediction head.

    Architecture:
    - gate_embedding: nn.Embedding(NUM_GATE_TYPES + 2, gate_embedding_dim)
      (+1 unknown, +1 mask token)
    - Node features: concat(gate_emb, gate_arity, is_directional, gate_index_norm)
    - input_proj: Linear(node_feature_dim, hidden_dim)
    - gnn_layers: GPSConv or TransformerConv layers
    - prediction_head: predicts original gate type (20 classes, no mask)
    """

    def __init__(
        self,
        hidden_dim=64,
        num_gnn_layers=4,
        gate_embedding_dim=16,
        dropout=0.1,
        mask_ratio=0.15,
        global_feature_dim=0,
        stats_loss_weight=1.0,
        gnn_type='gps',
        gps_heads=4,
        attn_type='multihead',
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_gnn_layers = num_gnn_layers
        self.mask_ratio = mask_ratio
        self.global_feature_dim = global_feature_dim
        self.stats_loss_weight = stats_loss_weight
        self.gnn_type = gnn_type

        # Gate embedding: NUM_GATE_TYPES (19 known) + 1 unknown + 1 mask = 21
        self.gate_embedding = nn.Embedding(NUM_GATE_TYPES + 2, gate_embedding_dim)

        # Node feature dim: gate_emb + gate_arity + is_directional + gate_index_norm
        node_feature_dim = gate_embedding_dim + 3

        # Project node features to hidden_dim (required for both GPS and residual)
        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)

        # GNN layers
        self.gnn_layers = nn.ModuleList()
        if gnn_type == 'gps':
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
            # Plain TransformerConv with manual LayerNorm + residual
            for _ in range(num_gnn_layers):
                self.gnn_layers.append(TransformerConv(hidden_dim, hidden_dim))
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim) for _ in range(num_gnn_layers)
            ])

        # Prediction head: predict original gate type (20 classes: 19 known + 1 unknown)
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, NUM_GATE_TYPES + 1),  # 20 classes (no mask token)
        )

        # Graph-level stats prediction head (mean+max+std pooling → predict global features)
        if global_feature_dim > 0:
            gnn_pool_dim = hidden_dim * 2  # mean + std
            self.stats_head = nn.Sequential(
                nn.Linear(gnn_pool_dim, hidden_dim),
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

    def _apply_masking(self, gate_type_idx):
        """
        BERT-style masking on gate type indices.

        Args:
            gate_type_idx: (N,) original gate type indices

        Returns:
            masked_gate_type_idx: (N,) with masking applied
            mask_positions: (N,) bool tensor of which positions are masked
            labels: (N,) original gate types at masked positions (-100 elsewhere)
        """
        device = gate_type_idx.device
        N = gate_type_idx.size(0)

        # Select 15% of positions to predict
        rand = torch.rand(N, device=device)
        mask_positions = rand < self.mask_ratio

        # Ensure at least 1 position is masked
        if not mask_positions.any():
            mask_positions[torch.randint(N, (1,), device=device)] = True

        masked_gate_type_idx = gate_type_idx.clone()

        # Of the masked positions:
        # 80% -> mask token
        # 10% -> random gate type
        # 10% -> unchanged
        mask_rand = torch.rand(N, device=device)

        # Mask token (80% of masked)
        mask_token_mask = mask_positions & (mask_rand < 0.8)
        masked_gate_type_idx[mask_token_mask] = MASK_TOKEN_IDX

        # Random gate type (10% of masked)
        random_mask = mask_positions & (mask_rand >= 0.8) & (mask_rand < 0.9)
        if random_mask.any():
            random_gates = torch.randint(0, NUM_GATE_TYPES, (random_mask.sum(),), device=device)
            masked_gate_type_idx[random_mask] = random_gates

        # Remaining 10%: unchanged

        # Labels: original gate type at masked positions, -100 elsewhere
        labels = torch.full((N,), -100, dtype=torch.long, device=device)
        labels[mask_positions] = gate_type_idx[mask_positions]

        return masked_gate_type_idx, mask_positions, labels

    def forward(self, data):
        """
        Forward pass with masked gate prediction.

        Args:
            data: PyG Data batch with gate_type_idx, gate_arity, is_directional,
                  gate_index_norm, edge_index

        Returns:
            (loss, accuracy) tuple
        """
        # Apply masking
        masked_gate_type_idx, mask_positions, labels = self._apply_masking(data.gate_type_idx)

        # Build node features with masked gate types
        gate_emb = self.gate_embedding(masked_gate_type_idx)
        x = torch.cat([
            gate_emb,
            data.gate_arity.float().unsqueeze(-1),
            data.is_directional.float().unsqueeze(-1),
            data.gate_index_norm.unsqueeze(-1),
        ], dim=1)

        edge_index = data.edge_index
        batch = data.batch

        # Project to hidden_dim
        x = self.input_proj(x)

        # GNN forward
        if self.gnn_type == 'gps':
            # GPSConv handles LayerNorm, residual, and activation internally
            for gps_layer in self.gnn_layers:
                x = gps_layer(x, edge_index, batch)
        else:
            # Plain TransformerConv with manual LayerNorm + residual
            for i, (gnn_layer, layer_norm) in enumerate(zip(self.gnn_layers, self.layer_norms)):
                residual = x
                x = gnn_layer(x, edge_index)
                x = layer_norm(x)
                x = F.silu(x)
                x = x + residual

        # Predict gate type at masked positions
        logits = self.prediction_head(x)  # (N, 20)

        # CE loss only at masked positions
        mask_loss = F.cross_entropy(logits, labels, ignore_index=-100)

        # Accuracy at masked positions
        with torch.no_grad():
            masked_logits = logits[mask_positions]
            masked_labels = labels[mask_positions]
            if masked_labels.numel() > 0:
                preds = masked_logits.argmax(dim=1)
                accuracy = (preds == masked_labels).float().mean().item()
            else:
                accuracy = 0.0

        # Graph-level stats prediction
        stats_loss = torch.tensor(0.0, device=x.device)
        if self.global_feature_dim > 0 and hasattr(data, 'global_features'):
            # Pool node features to graph level (mean + std, no max to avoid slow scatter)
            gnn_mean = global_mean_pool(x, batch)
            mean_sq = global_mean_pool(x * x, batch)
            gnn_std = (mean_sq - gnn_mean * gnn_mean).clamp(min=1e-6).sqrt()
            gnn_pooled = torch.cat([gnn_mean, gnn_std], dim=1)

            predicted_stats = self.stats_head(gnn_pooled)
            # PyG batches graph-level tensors; reshape to (B, dim)
            target_stats = data.global_features.reshape(-1, self.global_feature_dim)
            stats_loss = F.mse_loss(predicted_stats, target_stats)

        total_loss = mask_loss + self.stats_loss_weight * stats_loss

        return total_loss, mask_loss.item(), stats_loss.item(), accuracy

    def get_gnn_state_dict(self):
        """
        Extract GNN weights for transfer to downstream model.

        Returns state dict with:
        - gate_embedding (trimmed to 20 rows, dropping mask token)
        - input_proj
        - gnn_layers (includes LayerNorms when using GPS)
        - layer_norms (only for transformer_conv mode)
        """
        state_dict = {}

        # Gate embedding: trim from 21 rows to 20 (drop mask token row)
        emb_weight = self.gate_embedding.weight.data[:NUM_GATE_TYPES + 1]  # rows 0..19
        state_dict['gate_embedding.weight'] = emb_weight

        # input_proj
        for k, v in self.input_proj.state_dict().items():
            state_dict[f'input_proj.{k}'] = v

        # GNN layers (for GPS, this includes conv + attn + norms + mlp)
        for k, v in self.gnn_layers.state_dict().items():
            state_dict[f'gnn_layers.{k}'] = v

        # Layer norms (only exists for transformer_conv mode)
        if hasattr(self, 'layer_norms'):
            for k, v in self.layer_norms.state_dict().items():
                state_dict[f'layer_norms.{k}'] = v

        return state_dict
