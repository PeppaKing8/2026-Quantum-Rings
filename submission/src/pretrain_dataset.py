"""
Dataset for GNN pretraining on circuit graphs.

Loads pre-extracted graph features from a .pt file (list of dicts).
Each dict contains: gate_type_idx, gate_arity, is_directional,
gate_index_norm, edge_index, num_nodes.
"""

import torch
from torch_geometric.data import Data, Dataset


class PretrainGraphDataset(Dataset):
    """
    PyG Dataset for pretraining graphs.

    Loads from a .pt file containing a list of feature dicts.
    Filters graphs with more than max_nodes nodes.
    Global features are z-score normalized to stabilize stats prediction loss.
    """

    def __init__(self, data_path, max_nodes=10000):
        super().__init__()

        raw_graphs = torch.load(data_path, map_location='cpu')

        # Filter by max_nodes
        self.graphs = [g for g in raw_graphs if g['num_nodes'] <= max_nodes]

        print(f"Loaded {len(self.graphs)} graphs from {data_path} "
              f"(filtered from {len(raw_graphs)}, max_nodes={max_nodes})")

        # Compute z-score normalization stats for global features
        self.global_mean = None
        self.global_std = None
        gf_list = [g['global_features'] for g in self.graphs if 'global_features' in g]
        if gf_list:
            all_gf = torch.stack(gf_list)  # (N, dim)
            self.global_mean = all_gf.mean(dim=0)
            self.global_std = all_gf.std(dim=0).clamp(min=1e-6)
            print(f"  Global features: dim={all_gf.shape[1]}, "
                  f"mean range=[{self.global_mean.min():.2f}, {self.global_mean.max():.2f}], "
                  f"std range=[{self.global_std.min():.4f}, {self.global_std.max():.2f}]")

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        g = self.graphs[idx]

        data = Data(
            num_nodes=g['num_nodes'],
            gate_type_idx=g['gate_type_idx'],
            gate_arity=g['gate_arity'],
            is_directional=g['is_directional'],
            gate_index_norm=g['gate_index_norm'],
            edge_index=g['edge_index'],
        )

        # Global features as prediction target (z-score normalized)
        if 'global_features' in g and self.global_mean is not None:
            data.global_features = (g['global_features'] - self.global_mean) / self.global_std

        return data
