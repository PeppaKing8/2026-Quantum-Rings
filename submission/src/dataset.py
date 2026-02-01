"""
Dataset loader for iQuHACK 2026 challenge.

Threshold classification: one sample per (circuit, backend, precision).
- "ok" status: threshold_class = index of forward.threshold in ladder (0-8)
- "no_threshold_met" status: threshold_class = 9
"""

import json
import random
import torch
import numpy as np
from pathlib import Path
from torch_geometric.data import Data, HeteroData, Dataset
from .features import qasm_file_to_features, extract_global_features
from .features_bipartite import extract_bipartite_features


# Threshold ladder
THRESHOLD_LADDER = [1, 2, 4, 8, 16, 32, 64, 128, 256]
THRESHOLD_TO_IDX = {t: i for i, t in enumerate(THRESHOLD_LADDER)}
NUM_THRESHOLD_CLASSES = 10  # 9 threshold bins + 1 no_threshold_met

# Circuit families (sorted alphabetically)
CIRCUIT_FAMILIES = [
    'Amplitude_Estimation', 'CutBell', 'Deutsch_Jozsa', 'GHZ', 'GraphState',
    'Ground_State', 'Grover_NoAncilla', 'Grover_V_Chain', 'Portfolio_QAOA',
    'Portfolio_VQE', 'Pricing_Call', 'QAOA', 'QFT', 'QFT_Entangled', 'QNN',
    'QPE_Exact', 'Shor', 'TwoLocalRandom', 'VQE', 'W_State',
]
FAMILY_TO_IDX = {f: i for i, f in enumerate(CIRCUIT_FAMILIES)}
NUM_CIRCUIT_FAMILIES = len(CIRCUIT_FAMILIES) + 1  # +1 for unknown


class IQuHACKDataset(Dataset):
    """
    PyTorch Geometric Dataset for iQuHACK challenge.

    Each sample contains:
    - Circuit features (GNN + global)
    - Backend and precision (as bits)
    - Target: threshold class (0-9) and log runtime
    """

    def __init__(self, data_file, circuits_dir, split='train',
                 use_graph_features=True, feature_registry=None,
                 gnn_type='homogeneous', cache_key=None):
        super().__init__()

        self.data_file = Path(data_file)
        self.circuits_dir = Path(circuits_dir)
        self.split = split
        self.use_graph_features = use_graph_features
        self.feature_registry = feature_registry
        self.gnn_type = gnn_type
        self.cache_key = cache_key

        # Load data
        with open(data_file, 'r') as f:
            raw_data = json.load(f)

        self.circuits_meta = {c['file']: c for c in raw_data['circuits']}
        self.results = raw_data['results']

        # Fixed train/val split from CIRCUITS.MD
        test_files = {
            'ae_indep_qiskit_130.qasm', 'dj_indep_qiskit_30.qasm',
            'ghz_indep_qiskit_30.qasm', 'ghz_indep_qiskit_130.qasm',
            'grover-noancilla_indep_qiskit_11.qasm', 'grover-v-chain_indep_qiskit_17.qasm',
            'portfolioqaoa_indep_qiskit_17.qasm', 'portfoliovqe_indep_qiskit_18.qasm',
            'qft_indep_qiskit_15.qasm', 'qftentangled_indep_qiskit_30.qasm',
            'qpeexact_indep_qiskit_30.qasm', 'wstate_indep_qiskit_130.qasm',
        }

        train_files = {
            'ae_indep_qiskit_20.qasm', 'cutbell_n30_k6.qasm',
            'dj_indep_qiskit_15.qasm', 'dj_indep_qiskit_130.qasm',
            'ghz_indep_qiskit_15.qasm', 'ghz_indep_qiskit_100.qasm',
            'graphstate_indep_qiskit_15.qasm', 'groundstate_large_indep_qiskit_14.qasm',
            'grover-noancilla_indep_qiskit_7.qasm', 'grover-v-chain_indep_qiskit_7.qasm',
            'portfolioqaoa_indep_qiskit_10.qasm', 'portfoliovqe_indep_qiskit_10.qasm',
            'pricingcall_indep_qiskit_17.qasm', 'qaoa_indep_qiskit_16.qasm',
            'qft_indep_qiskit_30.qasm', 'qft_indep_qiskit_130.qasm',
            'qftentangled_indep_qiskit_15.qasm', 'qnn_indep_qiskit_20.qasm',
            'qpeexact_indep_qiskit_100.qasm', 'shor_15_4_indep_qiskit_18.qasm',
            'twolocalrandom_indep_qiskit_30.qasm', 'vqe_indep_qiskit_16.qasm',
            'wstate_indep_qiskit_15.qasm', 'wstate_indep_qiskit_30.qasm',
        }

        all_circuit_files = set(r['file'] for r in self.results)
        train_files = train_files & all_circuit_files
        val_files = test_files & all_circuit_files

        if split == 'train':
            self.results = [r for r in self.results if r['file'] in train_files]
        elif split == 'val':
            self.results = [r for r in self.results if r['file'] in val_files]
        else:
            raise ValueError(f"Invalid split: {split}")

        # Create samples from results
        self.samples = self._create_samples()

        n_ok = sum(1 for s in self.samples if not s['is_no_threshold'])
        n_no_thr = sum(1 for s in self.samples if s['is_no_threshold'])
        print(f"Loaded {len(self.samples)} samples for {split} split")
        print(f"  - OK samples (threshold 0-8): {n_ok}")
        print(f"  - No threshold met (class 9): {n_no_thr}")

    def _create_samples(self):
        """
        Create one sample per result entry.

        - Task 1 target: smallest threshold where fidelity >= 0.75 (from threshold_sweep)
        - Task 2 input/target: forward threshold and runtime (always present)
        """
        samples = []
        n_075_changed = 0  # debug: count differences from 0.99 to 0.75

        for result in self.results:
            file = result['file']
            backend = result['backend']
            precision = result['precision']
            status = result.get('status', 'ok')

            if status not in ('ok', 'no_threshold_met'):
                continue

            # Always look at forward section regardless of status
            forward = result.get('forward', {})
            if not forward or forward.get('returncode') != 0:
                continue

            runtime = forward.get('run_wall_s')
            if runtime is None:
                continue

            forward_threshold = forward.get('threshold')
            if forward_threshold not in THRESHOLD_TO_IDX:
                continue

            # Task 1: find smallest threshold with fidelity >= 0.75 from sweep
            threshold_075_class = 9  # default: no threshold met at 0.75
            is_no_075_threshold = True
            threshold_sweep = result.get('threshold_sweep', [])
            sweep_sorted = sorted(threshold_sweep, key=lambda x: x.get('threshold', float('inf')))
            for entry in sweep_sorted:
                thr_val = entry.get('threshold')
                fidelity = entry.get('sdk_get_fidelity')
                if thr_val is None or fidelity is None:
                    continue
                if thr_val not in THRESHOLD_TO_IDX:
                    continue
                if fidelity >= 0.75:
                    threshold_075_class = THRESHOLD_TO_IDX[thr_val]
                    is_no_075_threshold = False
                    break

            # Debug: count changes from 0.99 to 0.75
            if status == 'ok':
                old_class = THRESHOLD_TO_IDX[forward_threshold]
                if threshold_075_class != old_class:
                    n_075_changed += 1
            elif status == 'no_threshold_met':
                if not is_no_075_threshold:
                    n_075_changed += 1

            # Circuit family
            circuit_meta = self.circuits_meta.get(file, {})
            family = circuit_meta.get('family', 'unknown')
            family_idx = FAMILY_TO_IDX.get(family, len(CIRCUIT_FAMILIES))

            samples.append({
                'file': file,
                'backend': backend,
                'precision': precision,
                'threshold_class': threshold_075_class,
                'is_no_threshold': is_no_075_threshold,
                'input_threshold_idx': THRESHOLD_TO_IDX[forward_threshold],
                'runtime': runtime,
                'circuit_family_idx': family_idx,
            })

        print(f"  - Threshold changed (0.99 → 0.75): {n_075_changed}")
        return samples

    def len(self):
        return len(self.samples)

    def get(self, idx):
        """Get a single sample."""
        sample = self.samples[idx]

        qasm_path = self.circuits_dir / sample['file']
        backend = sample['backend']
        precision = sample['precision']

        # Encode backend and precision as bits
        backend_bit = 1.0 if backend == 'GPU' else 0.0
        precision_bit = 1.0 if precision == 'single' else 0.0

        # Log runtime (natural log)
        y_log_runtime = torch.tensor(
            np.log(max(sample['runtime'], 1e-6)), dtype=torch.float32
        )

        # Task 2 input: forward threshold index
        input_threshold_idx = torch.tensor(sample['input_threshold_idx'], dtype=torch.long)
        # Circuit family index
        circuit_family_idx = torch.tensor(sample['circuit_family_idx'], dtype=torch.long)

        if self.gnn_type == 'bipartite' and self.use_graph_features:
            return self._get_bipartite(
                qasm_path, backend, precision, backend_bit, precision_bit,
                y_log_runtime, input_threshold_idx, circuit_family_idx, sample,
            )

        # Get features (GNN + global), augment during training
        augment = (self.split == 'train')
        features = qasm_file_to_features(qasm_path, backend, precision,
                                        registry=self.feature_registry,
                                        augment=augment,
                                        cache_key=self.cache_key)

        if self.use_graph_features:
            num_nodes = len(features['gate_type_idx'])
            data = Data(
                # Explicit num_nodes so PyG can build batch vector correctly
                num_nodes=num_nodes,
                # Graph features (raw indices for learned embeddings)
                gate_type_idx=features['gate_type_idx'],
                qubit_indices=features['qubit_indices'],
                is_directional=features['is_directional'],
                gate_arity=features['gate_arity'],
                gate_index_norm=features['gate_index_norm'],
                edge_index=features['edge_index'],
                global_features=features['global_features'],

                # Context features
                backend_bit=torch.tensor(backend_bit, dtype=torch.float32),
                precision_bit=torch.tensor(precision_bit, dtype=torch.float32),
                input_threshold_idx=input_threshold_idx,
                circuit_family_idx=circuit_family_idx,

                # Targets
                y_threshold=torch.tensor(sample['threshold_class'], dtype=torch.long),
                y_log_runtime=y_log_runtime,
                is_no_threshold=torch.tensor(sample['is_no_threshold'], dtype=torch.bool),
            )
        else:
            data = Data(
                global_features=features['global_features'],

                # Context features
                backend_bit=torch.tensor(backend_bit, dtype=torch.float32),
                precision_bit=torch.tensor(precision_bit, dtype=torch.float32),
                input_threshold_idx=input_threshold_idx,
                circuit_family_idx=circuit_family_idx,

                # Targets
                y_threshold=torch.tensor(sample['threshold_class'], dtype=torch.long),
                y_log_runtime=y_log_runtime,
                is_no_threshold=torch.tensor(sample['is_no_threshold'], dtype=torch.bool),
            )

        return data

    def _get_bipartite(self, qasm_path, backend, precision, backend_bit, precision_bit,
                       y_log_runtime, input_threshold_idx, circuit_family_idx, sample):
        """Build a HeteroData sample with bipartite graph features."""
        from qiskit import QuantumCircuit

        circ = QuantumCircuit.from_qasm_file(str(qasm_path))

        # Bipartite graph features
        bp = extract_bipartite_features(circ)

        # Global features
        global_features = extract_global_features(
            circ, backend, precision, registry=self.feature_registry
        )

        # Normalize qubit degree
        degree = bp['qubit_degree'].float()
        max_deg = degree.max().clamp(min=1.0)
        qubit_degree_norm = degree / max_deg

        data = HeteroData()

        # Gate nodes
        data['gate'].gate_type_idx = bp['gate_type_idx']
        data['gate'].gate_arity = bp['gate_arity']
        data['gate'].gate_is_directional = bp['gate_is_directional']
        data['gate'].gate_index_norm = bp['gate_index_norm']
        data['gate'].num_nodes = bp['num_gates']

        # Qubit nodes
        data['qubit'].qubit_degree_norm = qubit_degree_norm
        data['qubit'].num_nodes = bp['num_qubits']

        # Bidirectional edges
        edge_index = bp['edge_index_gate_to_qubit']
        data['gate', 'acts_on', 'qubit'].edge_index = edge_index
        data['qubit', 'feeds_into', 'gate'].edge_index = edge_index.flip(0)

        # Graph-level attributes
        data.global_features = global_features
        data.backend_bit = torch.tensor(backend_bit, dtype=torch.float32)
        data.precision_bit = torch.tensor(precision_bit, dtype=torch.float32)
        data.input_threshold_idx = input_threshold_idx
        data.circuit_family_idx = circuit_family_idx
        data.y_threshold = torch.tensor(sample['threshold_class'], dtype=torch.long)
        data.y_log_runtime = y_log_runtime
        data.is_no_threshold = torch.tensor(sample['is_no_threshold'], dtype=torch.bool)

        return data


def get_dataloaders(data_file, circuits_dir, batch_size=32, use_graph_features=True,
                    num_workers=0, feature_registry=None, gnn_type='homogeneous',
                    cache_key=None, pin_memory=False):
    """
    Create train/val dataloaders.

    Args:
        gnn_type: 'homogeneous' (default) or 'bipartite'

    Returns:
        dict with 'train', 'val' DataLoaders
    """
    from torch_geometric.loader import DataLoader

    datasets = {
        'train': IQuHACKDataset(data_file, circuits_dir, split='train',
                                use_graph_features=use_graph_features,
                                feature_registry=feature_registry,
                                gnn_type=gnn_type, cache_key=cache_key),
        'val': IQuHACKDataset(data_file, circuits_dir, split='val',
                              use_graph_features=use_graph_features,
                              feature_registry=feature_registry,
                              gnn_type=gnn_type, cache_key=cache_key),
    }

    loaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        for split, dataset in datasets.items()
    }

    return loaders
