"""
Dataset for bipartite GNN pretraining on circuit graphs.

Loads pre-extracted bipartite graph features from a .pt file.
Returns HeteroData objects with gate and qubit node types.
Global features are z-score normalized.
"""

import torch
from torch_geometric.data import HeteroData, Dataset


class BipartitePretrainDataset(Dataset):
    """
    PyG Dataset for bipartite pretraining graphs.

    Each graph has:
    - Gate nodes: gate_type_idx, gate_arity, gate_is_directional, gate_index_norm
    - Qubit nodes: qubit_degree_norm
    - Edges: gate <-> qubit (bidirectional)
    - Optional: global_features (z-score normalized)
    """

    def __init__(self, data_path, max_nodes=10000):
        super().__init__()

        raw_graphs = torch.load(data_path, map_location='cpu')

        # Filter by max gate nodes
        self.graphs = [g for g in raw_graphs if g['num_gates'] <= max_nodes]

        print(f"Loaded {len(self.graphs)} bipartite graphs from {data_path} "
              f"(filtered from {len(raw_graphs)}, max_nodes={max_nodes})")

        # Z-score normalization for global features
        self.global_mean = None
        self.global_std = None
        gf_list = [g['global_features'] for g in self.graphs if 'global_features' in g]
        if gf_list:
            all_gf = torch.stack(gf_list)
            self.global_mean = all_gf.mean(dim=0)
            self.global_std = all_gf.std(dim=0).clamp(min=1e-6)
            print(f"  Global features: dim={all_gf.shape[1]}, "
                  f"mean range=[{self.global_mean.min():.2f}, {self.global_mean.max():.2f}], "
                  f"std range=[{self.global_std.min():.4f}, {self.global_std.max():.2f}]")

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        g = self.graphs[idx]

        data = HeteroData()

        # Gate node attributes
        data['gate'].gate_type_idx = g['gate_type_idx']
        data['gate'].gate_arity = g['gate_arity']
        data['gate'].gate_is_directional = g['gate_is_directional']
        data['gate'].gate_index_norm = g['gate_index_norm']
        data['gate'].num_nodes = g['num_gates']

        # Qubit node attributes
        # Normalize degree: degree / max_degree_in_this_circuit
        degree = g['qubit_degree'].float()
        max_deg = degree.max().clamp(min=1.0)
        data['qubit'].qubit_degree_norm = degree / max_deg
        data['qubit'].num_nodes = g['num_qubits']

        # Bidirectional edges
        edge_index = g['edge_index_gate_to_qubit']
        data['gate', 'acts_on', 'qubit'].edge_index = edge_index
        data['qubit', 'feeds_into', 'gate'].edge_index = edge_index.flip(0)

        # Global features (z-score normalized)
        if 'global_features' in g and self.global_mean is not None:
            data.global_features = (g['global_features'] - self.global_mean) / self.global_std

        return data
