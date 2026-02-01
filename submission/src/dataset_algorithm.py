"""
Dataset loader for algorithm/family classification.

One sample per QASM file. Labels are derived from filename prefix or a label map.
"""

import json
from pathlib import Path
import torch
from torch_geometric.data import Data, Dataset

from .features import qasm_file_to_features


def load_label_map(label_map_path):
    """Load label map JSON and normalize fields."""
    label_map_path = Path(label_map_path)
    with label_map_path.open("r", encoding="utf-8") as f:
        label_map = json.load(f)

    label_to_idx = label_map.get("label_to_idx", {})
    idx_to_label = label_map.get("idx_to_label", {})
    labels = label_map.get("labels", {})
    return label_map, labels, label_to_idx, idx_to_label


class AlgorithmClassificationDataset(Dataset):
    """
    PyTorch Geometric Dataset for algorithm/family classification.

    Each sample contains:
    - Circuit features (GNN + global)
    - Target: algorithm class index
    """

    def __init__(
        self,
        circuits_dir,
        label_map_path,
        split="train",
        use_graph_features=True,
        feature_registry=None,
    ):
        super().__init__()

        self.circuits_dir = Path(circuits_dir)
        self.split = split
        self.use_graph_features = use_graph_features
        self.feature_registry = feature_registry

        self.label_map, self.labels, self.label_to_idx, self.idx_to_label = load_label_map(
            label_map_path
        )

        split_dir = self.circuits_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        self.samples = []
        for qasm_path in sorted(split_dir.glob("*.qasm")):
            rel = f"{split}/{qasm_path.name}"
            label = self.labels.get(rel) or self.labels.get(qasm_path.name)
            if label is None:
                label = qasm_path.name.split("_", 1)[0]
            if label not in self.label_to_idx:
                raise ValueError(f"Label '{label}' not found in label_map for {rel}")
            self.samples.append(
                {
                    "path": qasm_path,
                    "label": label,
                    "label_idx": int(self.label_to_idx[label]),
                }
            )

        print(
            f"Loaded {len(self.samples)} samples for {split} split "
            f"from {split_dir}"
        )

    def len(self):
        return len(self.samples)

    def get(self, idx):
        sample = self.samples[idx]

        qasm_path = sample["path"]
        augment = self.split == "train"

        features = qasm_file_to_features(
            qasm_path,
            backend="CPU",
            precision="double",
            registry=self.feature_registry,
            augment=augment,
        )

        y_label = torch.tensor(sample["label_idx"], dtype=torch.long)

        if self.use_graph_features:
            num_nodes = len(features["gate_type_idx"])
            data = Data(
                num_nodes=num_nodes,
                gate_type_idx=features["gate_type_idx"],
                qubit_indices=features["qubit_indices"],
                is_directional=features["is_directional"],
                gate_arity=features["gate_arity"],
                gate_index_norm=features["gate_index_norm"],
                edge_index=features["edge_index"],
                global_features=features["global_features"],
                y_label=y_label,
            )
        else:
            data = Data(
                global_features=features["global_features"],
                y_label=y_label,
            )

        return data


def get_algorithm_dataloaders(
    circuits_dir,
    label_map_path,
    batch_size=32,
    use_graph_features=True,
    num_workers=0,
    feature_registry=None,
):
    """Create train/val dataloaders for algorithm classification."""
    from torch_geometric.loader import DataLoader

    datasets = {
        "train": AlgorithmClassificationDataset(
            circuits_dir,
            label_map_path,
            split="train",
            use_graph_features=use_graph_features,
            feature_registry=feature_registry,
        ),
        "val": AlgorithmClassificationDataset(
            circuits_dir,
            label_map_path,
            split="val",
            use_graph_features=use_graph_features,
            feature_registry=feature_registry,
        ),
    }

    loaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
        )
        for split, dataset in datasets.items()
    }

    return loaders
