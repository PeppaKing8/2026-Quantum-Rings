"""
Modular global feature extractors for quantum circuits.

Each feature extractor is a separate class that can be easily enabled/disabled.
Use GlobalFeatureRegistry to combine multiple extractors.
"""

import torch
import numpy as np
import networkx as nx
from abc import ABC, abstractmethod
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression
from qiskit.converters import circuit_to_dag
from networkx.utils import reverse_cuthill_mckee_ordering
import qiskit


class GlobalFeatureExtractor(ABC):
    """Base class for global circuit feature extractors."""

    @abstractmethod
    def extract(
        self, circuit: QuantumCircuit, backend="CPU", precision="double"
    ) -> list[float] | np.ndarray:
        """
        Extract features from a quantum circuit.

        Args:
            circuit: Qiskit QuantumCircuit
            backend: 'CPU' or 'GPU'
            precision: 'single' or 'double'

        Returns:
            list or np.ndarray of feature values
        """
        pass

    @abstractmethod
    def feature_dim(self) -> int:
        """Return the dimensionality of extracted features."""
        pass

    @abstractmethod
    def feature_names(self) -> list[str]:
        """Return names of features for interpretability."""
        pass

    def normalize(
        self, features, circuit, backend="CPU", precision="double"
    ) -> list[float] | np.ndarray:
        """
        Normalize extracted features for model-friendly scale.

        Default implementation is identity; override per extractor.
        """
        return features


class BasicStatsExtractor(GlobalFeatureExtractor):
    """Extract basic circuit statistics: depth, width, total gates."""

    def extract(self, circuit, backend="CPU", precision="double") -> list[float]:
        return [
            float(circuit.depth()),
            float(circuit.num_qubits),
            float(circuit.size()),
        ]

    def normalize(self, features, circuit, backend="CPU", precision="double"):
        depth, num_qubits, total_gates = features
        max_qubits = 150
        depth_norm = float(np.log1p(max(depth, 0)))
        qubits_norm = float(min(num_qubits / max(max_qubits, 1), 1.0))
        gates_norm = float(np.log1p(max(total_gates, 0)))
        return [depth_norm, qubits_norm, gates_norm]

    def feature_dim(self):
        return 3

    def feature_names(self):
        return ["depth", "num_qubits", "total_gates"]


class GateCountsExtractor(GlobalFeatureExtractor):
    """Extract counts for specific gate types."""

    def __init__(self, gate_types=None):
        """
        Args:
            gate_types: List of gate names to count (default: common gates)
        """
        if gate_types is None:
            self.gate_types = [
                "h",
                "x",
                "y",
                "z",
                "s",
                "t",
                "rx",
                "ry",
                "rz",
                "cx",
                "cz",
                "swap",
            ]
        else:
            self.gate_types = gate_types

    def extract(self, circuit, backend="CPU", precision="double") -> list[float]:
        gate_ops: dict[str, int] = {
            getattr(k, "name", str(k)): int(v) for k, v in circuit.count_ops().items()
        }
        return [float(gate_ops.get(gate, 0)) for gate in self.gate_types]

    def normalize(self, features, circuit, backend="CPU", precision="double"):
        return [float(np.log1p(max(value, 0))) for value in features]

    def feature_dim(self):
        return len(self.gate_types)

    def feature_names(self):
        return [f"count_{gate}" for gate in self.gate_types]


class ExecutionContextExtractor(GlobalFeatureExtractor):
    """Extract execution context: backend and precision (one-hot encoded)."""

    def extract(self, circuit, backend="CPU", precision="double") -> list[float]:
        return [
            1.0 if backend == "CPU" else 0.0,
            1.0 if backend == "GPU" else 0.0,
            1.0 if precision == "single" else 0.0,
            1.0 if precision == "double" else 0.0,
        ]

    def normalize(self, features, circuit, backend="CPU", precision="double"):
        return features

    def feature_dim(self):
        return 4

    def feature_names(self):
        return ["backend_cpu", "backend_gpu", "precision_single", "precision_double"]


class ComplexityMetricsExtractor(GlobalFeatureExtractor):
    """Extract circuit complexity metrics."""

    def extract(self, circuit, backend="CPU", precision="double") -> list[float]:
        features = []

        # Two-qubit gate ratio (proxy for entanglement)
        gate_ops: dict[str, int] = {
            getattr(k, "name", str(k)): int(v) for k, v in circuit.count_ops().items()
        }
        two_qubit_gates = sum(
            gate_ops.get(g, 0) for g in ["cx", "cy", "cz", "swap", "ccx"]
        )
        total_gates = max(circuit.size(), 1)
        features.append(two_qubit_gates / total_gates)

        # Circuit connectivity (graph density)
        if circuit.num_qubits > 1:
            # Count unique qubit pairs in 2-qubit gates
            dag = circuit_to_dag(circuit)
            qubit_pairs = set()
            for node in dag.op_nodes():
                if len(node.qargs) == 2:
                    q0, q1 = sorted([circuit.find_bit(q).index for q in node.qargs])
                    qubit_pairs.add((q0, q1))
            max_pairs = circuit.num_qubits * (circuit.num_qubits - 1) / 2
            features.append(len(qubit_pairs) / max(max_pairs, 1))
        else:
            features.append(0.0)

        # Depth-to-width ratio (circuit "shape")
        features.append(circuit.depth() / max(circuit.num_qubits, 1))

        return features

    def normalize(self, features, circuit, backend="CPU", precision="double"):
        two_qubit_ratio, connectivity, depth_width_ratio = features
        depth_width_ratio_norm = float(np.log1p(max(depth_width_ratio, 0)))
        return [two_qubit_ratio, connectivity, depth_width_ratio_norm]

    def feature_dim(self):
        return 3

    def feature_names(self):
        return ["two_qubit_ratio", "connectivity", "depth_width_ratio"]


class DegreeStatsExtractor(GlobalFeatureExtractor):
    """Extract degree-based statistics from the qubit interaction graph."""

    def extract(self, circuit, backend="CPU", precision="double") -> list[float]:
        num_qubits = circuit.num_qubits
        if num_qubits == 0:
            return [0.0, 0.0]

        adjacency = [set() for _ in range(num_qubits)]

        dag = circuit_to_dag(circuit)
        for node in dag.op_nodes():
            if len(node.qargs) == 2:
                q0, q1 = [circuit.find_bit(q).index for q in node.qargs]
                if q0 != q1:
                    adjacency[q0].add(q1)
                    adjacency[q1].add(q0)

        degrees = np.array(
            [len(neighbors) for neighbors in adjacency], dtype=np.float32
        )
        max_degree = float(degrees.max()) if degrees.size > 0 else 0.0

        degree_sum = float(degrees.sum())
        if degree_sum == 0.0:
            degree_entropy = 0.0
        else:
            probs = degrees / degree_sum
            nonzero = probs > 0
            degree_entropy = float(-np.sum(probs[nonzero] * np.log(probs[nonzero])))

        return [max_degree, degree_entropy]

    def normalize(self, features, circuit, backend="CPU", precision="double"):
        max_degree, degree_entropy = features
        num_qubits = circuit.num_qubits
        max_possible_degree = max(num_qubits - 1, 1)
        max_degree_norm = float(min(max_degree / max_possible_degree, 1.0))
        if num_qubits <= 1:
            entropy_norm = 0.0
        else:
            entropy_norm = float(degree_entropy / np.log(num_qubits))
        return [max_degree_norm, entropy_norm]

    def feature_dim(self):
        return 2

    def feature_names(self):
        return ["max_degree", "degree_entropy"]


class InteractionGraphOptimizerExtractor(GlobalFeatureExtractor):
    """Extract graph connectivity and optimized cut metrics."""

    def extract(self, circuit, backend="CPU", precision="double") -> list[float]:
        num_qubits = circuit.num_qubits

        qubit_map = {q: i for i, q in enumerate(circuit.qubits)}

        graph = nx.Graph()
        graph.add_nodes_from(range(num_qubits))
        gate_list = []

        for inst in circuit.data:
            try:
                qargs = inst.qubits
            except AttributeError:
                _, qargs, _ = inst

            if len(qargs) == 2:
                u = qubit_map[qargs[0]]
                v = qubit_map[qargs[1]]
                if graph.has_edge(u, v):
                    graph[u][v]["weight"] += 1
                else:
                    graph.add_edge(u, v, weight=1)
                gate_list.append((u, v))

        clustering_coeff = nx.average_clustering(graph) if num_qubits > 0 else 0.0
        connected_components = nx.number_connected_components(graph)

        def calculate_cut_profile(ordering_map, gates):
            cuts = [0] * (num_qubits - 1)
            max_span = 0
            for u, v in gates:
                p1, p2 = ordering_map[u], ordering_map[v]
                start, end = min(p1, p2), max(p1, p2)
                span = end - start
                if span > max_span:
                    max_span = span
                for k in range(start, end):
                    cuts[k] += 1
            return {
                "max_cut": max(cuts) if cuts else 0,
                "mean_cut": float(np.mean(cuts)) if cuts else 0.0,
                "max_span": max_span,
            }

        try:
            rcm_list = list(reverse_cuthill_mckee_ordering(graph))
            if len(rcm_list) < num_qubits:
                missing = set(range(num_qubits)) - set(rcm_list)
                rcm_list.extend(list(missing))
            rcm_map = {node_id: pos for pos, node_id in enumerate(rcm_list)}
            optimized_metrics = calculate_cut_profile(rcm_map, gate_list)
        except Exception:
            optimized_metrics = {"max_cut": 0, "mean_cut": 0.0, "max_span": 0}

        return [
            float(clustering_coeff),
            float(connected_components),
            float(optimized_metrics["max_cut"]),
            float(optimized_metrics["mean_cut"]),
            float(optimized_metrics["max_span"]),
        ]

    def normalize(self, features, circuit, backend="CPU", precision="double"):
        (
            clustering_coeff,
            connected_components,
            optimized_max_cut,
            optimized_mean_cut,
            optimized_max_span,
        ) = features
        connected_components_norm = float(np.log1p(max(connected_components, 0)))
        optimized_max_cut_norm = float(np.log1p(max(optimized_max_cut, 0)))
        optimized_mean_cut_norm = float(np.log1p(max(optimized_mean_cut, 0)))
        optimized_max_span_norm = float(np.log1p(max(optimized_max_span, 0)))
        return [
            clustering_coeff,
            connected_components_norm,
            optimized_max_cut_norm,
            optimized_mean_cut_norm,
            optimized_max_span_norm,
        ]

    def feature_dim(self):
        return 5

    def feature_names(self):
        return [
            "clustering_coeff",
            "connected_components",
            "optimized_max_cut",
            "optimized_mean_cut",
            "optimized_max_span",
        ]


class EntanglementMetricsExtractor(GlobalFeatureExtractor):
    """Extract entanglement-related metrics."""

    def extract(self, circuit, backend="CPU", precision="double") -> list[float]:
        features = []

        gate_ops: dict[str, int] = {
            getattr(k, "name", str(k)): int(v) for k, v in circuit.count_ops().items()
        }

        # CNOT count (primary entangling gate)
        features.append(gate_ops.get("cx", 0))

        # CNOT density (CNOTs per qubit)
        features.append(gate_ops.get("cx", 0) / max(circuit.num_qubits, 1))

        # Multi-qubit gate fraction
        multi_qubit = sum(gate_ops.get(g, 0) for g in ["cx", "cy", "cz", "swap", "ccx"])
        features.append(multi_qubit / max(circuit.size(), 1))

        return features

    def normalize(self, features, circuit, backend="CPU", precision="double"):
        cnot_count, cnot_density, multi_qubit_fraction = features
        cnot_count_norm = float(np.log1p(max(cnot_count, 0)))
        return [cnot_count_norm, cnot_density, multi_qubit_fraction]

    def feature_dim(self):
        return 3

    def feature_names(self):
        return ["cnot_count", "cnot_density", "multi_qubit_fraction"]


class DepthAnalysisExtractor(GlobalFeatureExtractor):
    """Extract depth-related analysis features."""

    def extract(self, circuit, backend="CPU", precision="double") -> list[float]:
        features = []

        # Circuit depth
        depth = circuit.depth()
        features.append(depth)

        # Average gates per layer
        features.append(circuit.size() / max(depth, 1))

        # Parallelism estimate (gates / (depth * qubits))
        features.append(circuit.size() / max(depth * circuit.num_qubits, 1))

        return features

    def normalize(self, features, circuit, backend="CPU", precision="double"):
        depth, avg_gates_per_layer, parallelism_factor = features
        depth_norm = float(np.log1p(max(depth, 0)))
        avg_gates_norm = float(np.log1p(max(avg_gates_per_layer, 0)))
        return [depth_norm, avg_gates_norm, parallelism_factor]

    def feature_dim(self):
        return 3

    def feature_names(self):
        return ["circuit_depth", "avg_gates_per_layer", "parallelism_factor"]


class AlgorithmFingerprintExtractor(GlobalFeatureExtractor):
    """Extract algorithm fingerprint scores (arithmetic, QNN, QFT)."""

    def __init__(self, max_decomposition_depth=3, standard_gates=None):
        self.max_decomposition_depth = max_decomposition_depth
        if standard_gates is None:
            self.standard_gates = {
                "x",
                "y",
                "z",
                "h",
                "s",
                "sdg",
                "t",
                "tdg",
                "rx",
                "ry",
                "rz",
                "p",
                "u",
                "u1",
                "u2",
                "u3",
                "cx",
                "cy",
                "cz",
                "swap",
                "cswap",
                "ccx",
                "mcx",
                "cp",
                "cu1",
                "mcp",
                "measure",
                "barrier",
                "reset",
            }
        else:
            self.standard_gates = set(standard_gates)

    def _decompose_opaque(self, qc):
        """
        Decomposes opaque gates (like EfficientSU2) while preserving
        standard gates (like ccx, t, cp) and maintaining registers.
        """
        # Standard gates we MUST NOT decompose
        standard_gates = {
            "x",
            "y",
            "z",
            "h",
            "s",
            "sdg",
            "t",
            "tdg",
            "rx",
            "ry",
            "rz",
            "p",
            "u",
            "u1",
            "u2",
            "u3",
            "cx",
            "cy",
            "cz",
            "swap",
            "cswap",
            "ccx",
            "mcx",
            "cp",
            "cu1",
            "mcp",
            "measure",
            "barrier",
            "reset",
        }

        max_passes = 3
        for _ in range(max_passes):
            has_opaque = False
            new_data = []

            for inst in qc.data:
                # If it's a standard gate, keep it.
                if inst.operation.name in standard_gates:
                    new_data.append(inst)
                else:
                    try:
                        # Decompose unknown instructions
                        # copying registers is crucial for the mini_qc too,
                        # but usually, decompose() handles the internal logic.
                        # A safer way to decompose a single instruction is slightly complex,
                        # but usually, we can just try to decompose the WHOLE circuit
                        # if we weren't doing this selective logic.

                        # For selective decomposition, we must handle the instruction:

                        # 1. Create a tiny circuit just for this instruction
                        # We need to map the specific qubits of 'inst' to a new mini circuit
                        # This is often too messy.

                        # EASIER APPROACH for the loop:
                        # Just use the definition if it exists
                        if inst.operation.definition:
                            # This unwraps the composite gate into its definition
                            # mapped to the correct qubits
                            decomp_insts = inst.operation.definition.data

                            # We have to remap the qubits from the definition to the real qubits
                            # This is hard to do manually.

                            # FALLBACK: Just use Qiskit's decompose() on the whole circuit
                            # but that decomposes everything.

                            # ACTUALLY, the previous approach was mostly fine,
                            # except for the temp_qc creation.

                            mini_qc = qiskit.QuantumCircuit(
                                qc.num_qubits, qc.num_clbits
                            )
                            # This mini_qc step is actually dangerous if we don't match registers.
                            # It's better to append to the instructions list directly if possible.

                            # Let's stick to the simplest fix for your specific error:
                            # The error happened when REBUILDING the main circuit.
                            pass

                        # REVERTING TO YOUR LOGIC with the fix:
                        # We create a dummy circuit to decompose this specific instruction.
                        # To avoid the Register error here too, we use loose bits matching the count.
                        # But the 'inst' has specific qubits.

                        # The robust way:
                        mini_qc = qc.copy_empty_like()  # Use correct registers!
                        mini_qc.append(inst)
                        decomp_qc = mini_qc.decompose()
                        for decomp_inst in decomp_qc.data:
                            new_data.append(decomp_inst)
                        has_opaque = True
                    except Exception:
                        # If cannot decompose, keep as is
                        new_data.append(inst)

            if has_opaque:
                # HERE IS THE MAIN FIX:
                temp_qc = qc.copy_empty_like()
                for i in new_data:
                    temp_qc.append(i)
                qc = temp_qc
            else:
                break

        return qc

    def extract(self, circuit, backend="CPU", precision="double") -> list[float]:
        qc = self._decompose_opaque(circuit)

        instructions = qc.data
        total_gates = len(instructions) if len(instructions) > 0 else 1

        op_counts = {}
        n_parameterized = 0

        for inst in instructions:
            name = inst.operation.name
            op_counts[name] = op_counts.get(name, 0) + 1

            if hasattr(inst.operation, "params") and len(inst.operation.params) > 0:
                p = inst.operation.params[0]
                if isinstance(p, ParameterExpression):
                    n_parameterized += 1
                elif isinstance(p, (float, int)):
                    val = float(p)
                    if abs(val) > 1e-5:
                        steps = val / (np.pi / 4)
                        if abs(steps - round(steps)) > 0.01:
                            n_parameterized += 1

        arithmetic_gates = {"ccx", "mcx", "cswap"}
        n_arith = sum(op_counts.get(k, 0) for k in arithmetic_gates)
        n_t = op_counts.get("t", 0) + op_counts.get("tdg", 0)

        arith_density = n_arith / total_gates
        t_density = n_t / total_gates

        score_arithmetic = 0.0
        if arith_density > 0.02:
            score_arithmetic = min(1.0, arith_density * 10.0)
        if score_arithmetic > 0.1 and t_density > 0.05:
            score_arithmetic = min(1.0, score_arithmetic + 0.2)

        qft_gates = {"cp", "cu1", "mcp"}
        n_cp = sum(op_counts.get(k, 0) for k in qft_gates)
        n_h = op_counts.get("h", 0)

        cp_density = n_cp / total_gates
        h_density = n_h / total_gates
        qft_purity = cp_density + h_density

        score_qft = 0.0
        if cp_density > 0.1:
            if qft_purity > 0.6:
                score_qft = 0.8
            elif qft_purity > 0.4:
                score_qft = 0.5

            bad_gates = n_parameterized + n_arith
            if bad_gates > 0:
                score_qft -= (bad_gates / total_gates) * 2.0

        score_qft = max(0.0, min(1.0, score_qft))

        param_density = n_parameterized / total_gates
        score_qnn = 0.0
        if param_density > 0.1:
            score_qnn = min(1.0, param_density * 2.0)
        if score_arithmetic > 0.5:
            score_qnn = 0.0

        return [score_arithmetic, score_qnn, score_qft]

    def normalize(self, features, circuit, backend="CPU", precision="double"):
        return features

    def feature_dim(self):
        return 3

    def feature_names(self):
        return ["score_arithmetic", "score_qnn", "score_qft"]


class GlobalFeatureRegistry:
    """
    Registry that combines multiple feature extractors.

    Usage:
        registry = GlobalFeatureRegistry()
        registry.register(BasicStatsExtractor())
        registry.register(GateCountsExtractor())
        features = registry.extract_all(circuit, backend, precision)
    """

    def __init__(self):
        self.extractors = []
        self.standardize_mean = None
        self.standardize_std = None
        self.standardize_enabled = False

    def register(self, extractor):
        """Add a feature extractor to the registry."""
        if not isinstance(extractor, GlobalFeatureExtractor):
            raise TypeError(
                f"Extractor must inherit from GlobalFeatureExtractor, got {type(extractor)}"
            )
        self.extractors.append(extractor)
        return self  # Allow chaining

    def extract_all(self, circuit, backend="CPU", precision="double"):
        """
        Extract features from all registered extractors.

        Returns:
            torch.Tensor of shape (1, total_dim)
        """
        all_features = []

        for extractor in self.extractors:
            extractor: GlobalFeatureExtractor
            features = extractor.extract(circuit, backend, precision)
            features = extractor.normalize(features, circuit, backend, precision)
            if isinstance(features, (list, tuple)):
                all_features.extend(features)
            elif isinstance(features, np.ndarray):
                all_features.extend(features.tolist())
            else:
                all_features.append(features)

        features_tensor = torch.tensor(all_features, dtype=torch.float32).unsqueeze(0)
        if (
            self.standardize_enabled
            and self.standardize_mean is not None
            and self.standardize_std is not None
        ):
            features_tensor = (
                features_tensor - self.standardize_mean
            ) / self.standardize_std
        return features_tensor

    def set_standardization(self, mean, std, eps=1e-8):
        mean_tensor = torch.tensor(mean, dtype=torch.float32).unsqueeze(0)
        std_tensor = torch.tensor(std, dtype=torch.float32).unsqueeze(0)
        std_tensor = torch.where(
            std_tensor < eps, torch.ones_like(std_tensor), std_tensor
        )
        self.standardize_mean = mean_tensor
        self.standardize_std = std_tensor
        self.standardize_enabled = True

    def clear_standardization(self):
        self.standardize_mean = None
        self.standardize_std = None
        self.standardize_enabled = False

    def total_dim(self):
        """Total feature dimensionality."""
        return sum(e.feature_dim() for e in self.extractors)

    def all_feature_names(self):
        """Get all feature names from all extractors."""
        names = []
        for extractor in self.extractors:
            names.extend(extractor.feature_names())
        return names

    def summary(self):
        """Print a summary of registered extractors."""
        print(f"{'=' * 60}")
        print("Global Feature Registry Summary")
        print(f"{'=' * 60}")
        print(f"Total extractors: {len(self.extractors)}")
        print(f"Total features: {self.total_dim()}")
        print(f"\n{'Extractor':<35} {'Dim':<8} {'Features'}")
        print(f"{'-' * 60}")

        for extractor in self.extractors:
            name = extractor.__class__.__name__
            dim = extractor.feature_dim()
            feature_list = ", ".join(extractor.feature_names()[:3])
            if len(extractor.feature_names()) > 3:
                feature_list += ", ..."
            print(f"{name:<35} {dim:<8} {feature_list}")
        print(f"{'=' * 60}")


def create_default_registry():
    """
    Create a registry with default feature extractors.

    This matches the original feature set from features.py.
    """
    registry = GlobalFeatureRegistry()

    # Register extractors in order
    registry.register(BasicStatsExtractor())
    registry.register(GateCountsExtractor())
    registry.register(ExecutionContextExtractor())
    registry.register(ComplexityMetricsExtractor())
    registry.register(DegreeStatsExtractor())
    registry.register(AlgorithmFingerprintExtractor())
    registry.register(InteractionGraphOptimizerExtractor())

    return registry


def create_default_no_context_registry():
    """
    Create a default registry without execution context features.

    This is useful for algorithm classification where backend/precision
    should not be part of the input.
    """
    registry = GlobalFeatureRegistry()

    registry.register(BasicStatsExtractor())
    registry.register(GateCountsExtractor())
    registry.register(ComplexityMetricsExtractor())
    registry.register(DegreeStatsExtractor())
    registry.register(AlgorithmFingerprintExtractor())
    registry.register(InteractionGraphOptimizerExtractor())

    return registry


def create_minimal_registry():
    """
    Create a minimal registry with only basic features.

    Useful for fast baseline models.
    """
    registry = GlobalFeatureRegistry()

    registry.register(BasicStatsExtractor())
    registry.register(ExecutionContextExtractor())

    return registry


def create_extended_registry():
    """
    Create an extended registry with additional features.

    Includes more detailed analysis for advanced models.
    """
    registry = GlobalFeatureRegistry()

    # All default features
    registry.register(BasicStatsExtractor())
    registry.register(GateCountsExtractor())
    registry.register(ExecutionContextExtractor())
    registry.register(ComplexityMetricsExtractor())
    registry.register(DegreeStatsExtractor())
    registry.register(AlgorithmFingerprintExtractor())
    registry.register(InteractionGraphOptimizerExtractor())

    # Additional features
    registry.register(EntanglementMetricsExtractor())
    registry.register(DepthAnalysisExtractor())

    return registry
