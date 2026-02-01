"""
Feature extraction from QASM circuits for iQuHACK 2026.

Node features use raw indices for learned embeddings:
- gate_type_idx: integer gate type (for nn.Embedding lookup)
- qubit_indices: qubit indices per gate (for nn.Embedding lookup)
- positional role: directional vs non-directional gates
- gate_index_norm: normalized gate position

Random qubit permutation augmentation for training.
Uses modular feature extractors from feature_extractors.py for easy customization,
including algorithm fingerprints and interaction-graph metrics.
"""

import re
import random as _random
import torch
import networkx as nx
from pathlib import Path
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode
from torch_geometric.utils.convert import from_networkx
from .feature_extractors import create_default_registry


# Common gate types in OpenQASM 2.0
GATE_TYPES = {
    "h": 0,
    "x": 1,
    "y": 2,
    "z": 3,
    "s": 4,
    "t": 5,
    "sdg": 6,
    "tdg": 7,
    "rx": 8,
    "ry": 9,
    "rz": 10,
    "u1": 11,
    "u2": 12,
    "u3": 13,
    "cx": 14,
    "cy": 15,
    "cz": 16,
    "swap": 17,
    "ccx": 18,
}
NUM_GATE_TYPES = len(GATE_TYPES)

# Gates where qubit ordering matters (control vs target)
# cx/cy: control-target; ccx: control-control-target
# cz and swap are symmetric
DIRECTIONAL_GATES = {'cx', 'cy', 'ccx'}

MAX_GATE_ARITY = 3  # ccx has 3 qubits


def random_permute_qubits(circ):
    """
    Apply random global qubit permutation for data augmentation.

    This only relabels qubit indices globally - the circuit structure
    (which gates act on which qubits) is preserved. Drawing the circuit
    with the new indices produces the exact same diagram.

    Args:
        circ: Qiskit QuantumCircuit

    Returns:
        New QuantumCircuit with permuted qubit indices
    """
    num_qubits = circ.num_qubits
    perm = list(range(num_qubits))
    _random.shuffle(perm)

    new_circ = QuantumCircuit(num_qubits, circ.num_clbits)
    for instruction in circ.data:
        old_qubits = instruction.qubits
        new_qubits = [new_circ.qubits[perm[circ.find_bit(q).index]] for q in old_qubits]
        # Preserve classical bit arguments (e.g., for measure gates)
        clbits = [new_circ.clbits[circ.find_bit(c).index] for c in instruction.clbits]
        new_circ.append(instruction.operation, new_qubits, clbits)

    return new_circ


def extract_global_features(circ, backend='CPU', precision='double', registry=None):
    """
    Extract circuit-level global features using modular extractors.

    Args:
        circ: Qiskit QuantumCircuit
        backend: 'CPU' or 'GPU'
        precision: 'single' or 'double'
        registry: GlobalFeatureRegistry (if None, uses default)

    Returns:
        torch.Tensor of shape (1, feature_dim)
    """
    if registry is None:
        registry = create_default_registry()

    return registry.extract_all(circ, backend, precision)


def extract_graph_features(circ, max_gate_arity=MAX_GATE_ARITY):
    """
    Extract node-level raw indices and graph structure from circuit DAG.

    Returns raw indices for learned embeddings in the model:
    - gate_type_idx: (N,) integer gate type index
    - qubit_indices: (N, max_gate_arity) qubit indices, padded with 0
    - is_directional: (N,) bool - whether gate has meaningful qubit ordering
    - gate_arity: (N,) integer - number of qubits per gate
    - gate_index_norm: (N,) float - normalized gate position in circuit
    - edge_index: (2, E) graph edges

    Args:
        circ: Qiskit QuantumCircuit
        max_gate_arity: Maximum number of qubit arguments per gate

    Returns:
        dict with the above tensors
    """
    dag = circuit_to_dag(circ)

    gate_type_indices = []
    qubit_indices_list = []
    is_directional_list = []
    gate_arity_list = []
    gate_index_norms = []
    node_list = []

    total_gates = max(dag.size(), 1)

    for node in dag.topological_op_nodes():
        if not isinstance(node, DAGOpNode):
            continue

        node_list.append(node)

        gate_name = node.op.name.lower()

        # Gate type index
        gate_idx = GATE_TYPES.get(gate_name, NUM_GATE_TYPES)  # NUM_GATE_TYPES = unknown
        gate_type_indices.append(gate_idx)

        # Qubit indices (padded to max_gate_arity)
        qubits = [circ.find_bit(q).index for q in node.qargs]
        arity = len(qubits)
        padded_qubits = qubits[:max_gate_arity] + [0] * (max_gate_arity - min(arity, max_gate_arity))
        qubit_indices_list.append(padded_qubits)

        # Directional flag
        is_directional_list.append(gate_name in DIRECTIONAL_GATES)

        # Gate arity
        gate_arity_list.append(min(arity, max_gate_arity))

        # Normalized gate index
        gate_index_norms.append(len(gate_index_norms) / total_gates)

    if not node_list:
        # Empty circuit - return dummy features
        return {
            'gate_type_idx': torch.zeros(1, dtype=torch.long),
            'qubit_indices': torch.zeros(1, max_gate_arity, dtype=torch.long),
            'is_directional': torch.zeros(1, dtype=torch.bool),
            'gate_arity': torch.ones(1, dtype=torch.long),
            'gate_index_norm': torch.zeros(1, dtype=torch.float32),
            'edge_index': torch.zeros((2, 0), dtype=torch.long),
        }

    # Extract edges from DAG
    edges = []
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}

    for idx, node in enumerate(node_list):
        for successor in dag.successors(node):
            if successor in node_to_idx:
                edges.append([idx, node_to_idx[successor]])

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return {
        'gate_type_idx': torch.tensor(gate_type_indices, dtype=torch.long),
        'qubit_indices': torch.tensor(qubit_indices_list, dtype=torch.long),
        'is_directional': torch.tensor(is_directional_list, dtype=torch.bool),
        'gate_arity': torch.tensor(gate_arity_list, dtype=torch.long),
        'gate_index_norm': torch.tensor(gate_index_norms, dtype=torch.float32),
        'edge_index': edge_index,
    }


# Cache for feature extraction (validation only, augment=False)
# Key: (path_str, backend, precision, registry_name) -> features dict
_FEATURE_CACHE = {}


def qasm_file_to_features(qasm_path, backend='CPU', precision='double',
                           registry=None, augment=False, cache_key=None):
    """
    Load QASM file and extract all features.

    Args:
        qasm_path: Path to QASM file
        backend: 'CPU' or 'GPU'
        precision: 'single' or 'double'
        registry: GlobalFeatureRegistry (if None, uses default)
        augment: Whether to apply random qubit permutation (training only)
        cache_key: If set and augment=False, use cache (speeds up validation)

    Returns:
        dict with 'global_features' and graph feature tensors
    """
    path_str = str(Path(qasm_path).resolve())
    use_cache = cache_key is not None and not augment
    if use_cache:
        key = (path_str, backend, precision, cache_key)
        if key in _FEATURE_CACHE:
            return _FEATURE_CACHE[key]

    circ = QuantumCircuit.from_qasm_file(qasm_path)

    if augment:
        circ = random_permute_qubits(circ)

    global_features = extract_global_features(
        circ, backend, precision, registry=registry
    )
    graph_data = extract_graph_features(circ)

    result = {
        'global_features': global_features,
        **graph_data,
    }
    if use_cache:
        _FEATURE_CACHE[key] = result
    return result


def quick_qasm_features(qasm_path):
    """
    Quick lightweight feature extraction without Qiskit parsing.
    Useful for fast baseline models.

    Returns:
        dict of simple features (all numeric)
    """
    text = Path(qasm_path).read_text(encoding="utf-8")

    # Line counts
    lines = [
        ln.strip()
        for ln in text.splitlines()
        if ln.strip() and not ln.strip().startswith("//")
    ]

    # Gate counts (simple regex)
    features = {
        "n_lines": len(lines),
        "n_qubits": 0,  # parsed from qreg line
        "n_h": len(re.findall(r"\bh\b", text)),
        "n_x": len(re.findall(r"\bx\b", text)),
        "n_y": len(re.findall(r"\by\b", text)),
        "n_z": len(re.findall(r"\bz\b", text)),
        "n_s": len(re.findall(r"\bs\b", text)),
        "n_t": len(re.findall(r"\bt\b", text)),
        "n_rx": len(re.findall(r"\brx\b", text)),
        "n_ry": len(re.findall(r"\bry\b", text)),
        "n_rz": len(re.findall(r"\brz\b", text)),
        "n_cx": len(re.findall(r"\bcx\b", text)),
        "n_cy": len(re.findall(r"\bcy\b", text)),
        "n_cz": len(re.findall(r"\bcz\b", text)),
        "n_swap": len(re.findall(r"\bswap\b", text)),
        "n_measure": len(re.findall(r"\bmeasure\b", text)),
    }

    # Parse qubit count from qreg line
    qreg_match = re.search(r"qreg\s+\w+\[(\d+)\]", text)
    if qreg_match:
        features["n_qubits"] = int(qreg_match.group(1))

    return features
