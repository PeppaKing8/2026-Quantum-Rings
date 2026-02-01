import numpy as np
import qiskit, random

def permute_qubits(qc: qiskit.QuantumCircuit) -> qiskit.QuantumCircuit:
    """
    Returns a copy of the circuit with qubits randomly permuted.
    """
    # 1. Generate a random permutation of indices [0, ..., N-1]
    n_qubits = qc.num_qubits
    perm = np.random.permutation(n_qubits)
    
    # Map: Old Index -> New Index
    # mapping[i] = j means the qubit at index i moves to index j
    index_map = {i: perm[i] for i in range(n_qubits)}
    
    # 2. Create a new empty circuit with the same structure (registers, name, etc.)
    new_qc = qc.copy_empty_like()
    
    # 3. Rebuild the circuit instruction by instruction
    for instruction in qc.data:
        op = instruction.operation
        old_qubits = instruction.qubits
        clbits = instruction.clbits  # We usually keep classical bits as is
        
        new_qubits = []
        for q in old_qubits:
            # Get the flat index of the qubit in the original circuit
            old_idx = qc.find_bit(q).index
            
            # Find the new index using our random permutation
            new_idx = index_map[old_idx]
            
            # Get the actual Qubit object from the new circuit at that index
            new_q = new_qc.qubits[new_idx]
            new_qubits.append(new_q)
        
        # Append the operation with the remapped qubits
        new_qc.append(op, new_qubits, clbits)
        
    return new_qc

def circuit_augmentation(circuit: qiskit.QuantumCircuit) -> qiskit.QuantumCircuit:
    """
    Augments the given circuit by randomly permuting its qubits.
    """
    return permute_qubits(circuit)

FAMILY_TO_FILE_NAME = {
    "Deutsch_Jozsa": "dj",
    "QFT_Entangled": "qftentangled",
    # "TwoLocalRandom": "twolocalrandom",
    "QNN": "qnn",
    # "Grover_V_Chain": None, # not supporting!
    # "Portfolio_VQE": "vqiskit_application_finance.portfoliovqe", 
    "Grover_NoAncilla": "grover",
    "GHZ": "ghz",
    # "CutBell": None, # not supporting!
    "GraphState": "graphstate",
    "Portfolio_QAOA": "qaoa",
    "QPE_Exact": "qpeexact",
    "QAOA": "qaoa",
    # "Pricing_Call": "vqiskit_application_finance.pricingcall",
    "QFT": "qft",
    "Amplitude_Estimation": "ae",
    "VQE": "vqe",
    "W_State": "wstate",
    "Shor": "shor",
    # "Ground_State": "qiskit_application_nature.groundstate",
}

def generate_circuit_from_family(family: str, num_qubits: int = None) -> qiskit.QuantumCircuit:
    """
    Generates a quantum circuit based on the specified family and number of qubits.
    Applies circuit augmentation.
    """
    
    # first, find the module
    # it's external/bench/src/mqt/bench/benchmarks/{family}.py
    file_name = FAMILY_TO_FILE_NAME.get(family)
    if file_name is None:
        raise ValueError(f"Unknown family: {family}")
    
    if isinstance(file_name, set):
        # random choice from the set
        file_name = np.random.choice(list(file_name))
    
    # import the module
    module_path = f"external.bench.src.mqt.bench.benchmarks.{file_name}"
    module = __import__(module_path, fromlist=[''])
    
    # now, find the function that generates the circuit
    func_name = f"create_circuit"
    if not hasattr(module, func_name):
        raise ValueError(f"Module {module_path} does not have function {func_name}")
    
    create_circuit_func = getattr(module, func_name)
    
    # Generate the circuit
    if num_qubits is not None:
        circuit = create_circuit_func(num_qubits)
    else:
        # random num_qubits between 7 and 128
        num_qubits = np.random.randint(7, 129)
        circuit = create_circuit_func(num_qubits)
    
    # Apply augmentation
    augmented_circuit = circuit_augmentation(circuit)
    return augmented_circuit

def find_all_families() -> list[str]:
    """
    Returns a list of all available circuit families.
    """
    file_path = "data/hackathon_public.json"
    import json
    with open(file_path, 'r') as f:
        data = json.load(f)
    circuits = data["circuits"]
    print(f"Find {len(circuits)} circuits in total.")
    
    families = set()
    for circuit in circuits:    
        family = circuit["family"]
        families.add(family)
    return list(families)

def preprocess_data():
    file_path = "data/hackathon_public.json"
    import json
    with open(file_path, 'r') as f:
        data = json.load(f)
    circuits = data["circuits"]
    results = data["results"]
    
    print("Total circuits:", len(circuits))
    print("Total results:", len(results))
    
    for circuit in circuits:
        circuit['results'] = {}
        if 'source' in circuit:
            del circuit['source']
    
    for result in results:
        filename = result["file"]
        matching_circuits = [c for c in circuits if c["file"] == filename]
        if not matching_circuits:
            print(f"No matching circuit found for result file: {filename}")
            continue
        assert len(matching_circuits) == 1, f"Multiple matching circuits found for result file: {filename}"
        info = matching_circuits[0]
        backend, precision = result["backend"], result["precision"]
        key = backend + "_" + precision
        info['results'][key] = {}
        cur = info['results'][key]
        cur['status'] = result["status"]
        cur['fidelity'] = []
        for item in result["threshold_sweep"]:
            thres = item["threshold"]
            fid = item["sdk_get_fidelity"]
            cur['fidelity'].append((thres, fid))
        
        cur['runtime'] = result["forward"]["run_wall_s"]
        cur['thres'] = result["forward"]["threshold"]
        
    with open("data/hackathon_processed.json", 'w') as f:
        json.dump(circuits, f, indent=2)
        
    # categorized by family
    family_dict = {}
    for circuit in circuits:
        family = circuit["family"]
        if family not in family_dict:
            family_dict[family] = []
        circ = circuit.copy()
        del circ['family']
        family_dict[family].append(circ)
    
    with open("data/hackathon_by_family.json", 'w') as f:
        json.dump(family_dict, f, indent=2)

def _bind_params_for_qasm2(circuit: qiskit.QuantumCircuit) -> qiskit.QuantumCircuit:
    """Bind unbound parameters so the circuit can be exported to OpenQASM 2."""
    if circuit.parameters:
        param_bind = {p: random.uniform(0, 2 * np.pi) for p in circuit.parameters}
        return circuit.assign_parameters(param_bind)
    return circuit


if __name__ == "__main__":
    family_name = "Deutsch_Jozsa"
    num_qubits = random.randint(3, 12)
    circuit = generate_circuit_from_family(family_name, num_qubits=num_qubits)
    circuit = _bind_params_for_qasm2(circuit)
    output_qasm = f"{family_name}_{num_qubits}_{random.randint(100, 999)}.qasm"
    try:
        from qiskit.qasm2 import dump as qasm2_dump
        with open(output_qasm, "w") as f:
            qasm2_dump(circuit, f)
    except (ImportError, AttributeError):
        from qiskit.qasm3 import dumps as qasm3_dumps
        with open(output_qasm, "w") as f:
            f.write(qasm3_dumps(circuit))
    print(f"Wrote circuit to {output_qasm}")
    
    # families = find_all_families()
    # print("Available families:")
    # for fam in families:
    #     print(f"- {fam}")
    
    preprocess_data()