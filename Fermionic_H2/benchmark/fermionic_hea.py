'Particle-number-conserving fermionic HEA circuits.'
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

def givens_block(qc: QuantumCircuit, q1: int, q2: int, theta):
    'Two-qubit particle-conserving excitation gate. '
    # Outer CX
    qc.cx(q1, q2)

    # Elementary decomposition of CRY(2*theta, control=q2, target=q1)
    qc.ry(theta, q1)
    qc.cx(q2, q1)
    qc.ry(-theta, q1)
    qc.cx(q2, q1)

    # Outer CX
    qc.cx(q1, q2)

def make_fermionic_hea(num_qubits: int = 4, nelec: int = 2, layers: int = 2):
    'Constructs an N-particle preserving fermionic ansatz. '
    qc = QuantumCircuit(num_qubits)

    # Prepare Hartree-Fock reference state |1100>
    for q in range(nelec):
        qc.x(q)

    # Parameterized particle-conserving layers
    params = ParameterVector("theta", layers * 3)
    pidx = 0

    for _ in range(layers):
        qc.barrier()
        # Even-parity hopping
        givens_block(qc, 0, 1, params[pidx])
        givens_block(qc, 2, 3, params[pidx + 1])
        pidx += 2

        # Odd-parity hopping (inter-orbital coupling)
        givens_block(qc, 1, 2, params[pidx])
        pidx += 1

    return qc

def get_smooth_params(qc: QuantumCircuit, layer_depth: int, seed: int = 42):
    'Generates coherent variational angles so the target energy evolves smoothly with depth.'
    rng = np.random.default_rng(seed)
    base_angles = rng.uniform(0.15, 0.35, size=qc.num_parameters)
    scaled_angles = base_angles / np.sqrt(layer_depth)
    return dict(zip(qc.parameters, scaled_angles))
