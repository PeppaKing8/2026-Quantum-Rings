"""
Bipartite (incidence) graph feature extraction from quantum circuits.

Represents circuits as heterogeneous bipartite graphs:
- Gate nodes: represent quantum operations (H, CNOT, RZ, etc.)
- Qubit nodes: represent physical qubits / state carriers
- Edges: gate <-> qubit (a gate connects to each qubit it acts on)

This separates operators from state carriers, giving the GNN
explicit access to qubit-level structure (degree, connectivity)
without needing global feature engineering.
"""

import torch
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode

from .features import GATE_TYPES, NUM_GATE_TYPES, DIRECTIONAL_GATES, MAX_GATE_ARITY


def extract_bipartite_features(circ):
    """
    Extract bipartite graph features from a quantum circuit.

    Gate nodes get: gate_type_idx, gate_arity, is_directional, gate_index_norm
    Qubit nodes get: qubit_degree_norm (normalized number of gates on this qubit)
    Edges: gate <-> qubit incidence

    Args:
        circ: Qiskit QuantumCircuit

    Returns:
        dict with:
        - gate_type_idx: (num_gates,) long
        - gate_arity: (num_gates,) long
        - gate_is_directional: (num_gates,) bool
        - gate_index_norm: (num_gates,) float
        - num_gates: int
        - num_qubits: int
        - qubit_degree: (num_qubits,) long — raw gate count per qubit
        - edge_index_gate_to_qubit: (2, num_edges) long
    """
    dag = circuit_to_dag(circ)

    gate_type_indices = []
    gate_arity_list = []
    gate_is_directional_list = []
    gate_index_norms = []

    edges_src = []  # gate indices
    edges_dst = []  # qubit indices

    num_qubits = circ.num_qubits
    qubit_degree = [0] * num_qubits

    op_nodes = list(dag.topological_op_nodes())
    total_gates = max(len(op_nodes), 1)

    for gate_idx, node in enumerate(op_nodes):
        if not isinstance(node, DAGOpNode):
            continue

        gate_name = node.op.name.lower()

        # Gate features
        gate_type_indices.append(GATE_TYPES.get(gate_name, NUM_GATE_TYPES))
        arity = len(node.qargs)
        gate_arity_list.append(min(arity, MAX_GATE_ARITY))
        gate_is_directional_list.append(gate_name in DIRECTIONAL_GATES)
        gate_index_norms.append(gate_idx / total_gates)

        # Bipartite edges: gate <-> qubit
        for qubit in node.qargs:
            q_idx = circ.find_bit(qubit).index
            edges_src.append(gate_idx)
            edges_dst.append(q_idx)
            qubit_degree[q_idx] += 1

    num_gates = len(gate_type_indices)

    if num_gates == 0:
        return None

    # Edge index for gate -> qubit
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)

    return {
        'gate_type_idx': torch.tensor(gate_type_indices, dtype=torch.long),
        'gate_arity': torch.tensor(gate_arity_list, dtype=torch.long),
        'gate_is_directional': torch.tensor(gate_is_directional_list, dtype=torch.bool),
        'gate_index_norm': torch.tensor(gate_index_norms, dtype=torch.float32),
        'num_gates': num_gates,
        'num_qubits': num_qubits,
        'qubit_degree': torch.tensor(qubit_degree, dtype=torch.long),
        'edge_index_gate_to_qubit': edge_index,
    }
