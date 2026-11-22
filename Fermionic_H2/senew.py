import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

def default_single_qubit_pool(num_qubits: int) -> list[SparsePauliOp]:
    """Creates a pool of single-qubit X and Z operators for every qubit."""
    pool = []
    for q in range(num_qubits):
        for pauli in ("X", "Z"):
            label = ["I"] * num_qubits
            label[num_qubits - 1 - q] = pauli
            pool.append(SparsePauliOp.from_list([("".join(label), 1.0)]))
    return pool

class SEMitigator:
    def __init__(self, operator_pool: list[SparsePauliOp], eigenvalue_cutoff: float = 1e-6):
        self.operator_pool = operator_pool
        self.eigenvalue_cutoff = eigenvalue_cutoff

    def run(self, circuit: QuantumCircuit, observable: SparsePauliOp, noise_model) -> dict:
        num_qubits = circuit.num_qubits
        identity = SparsePauliOp.from_list([("I" * num_qubits, 1.0)])
        pool = [identity] + self.operator_pool
        k = len(pool)

        # Simulate the noisy density matrix
        backend = AerSimulator(noise_model=noise_model, method="density_matrix")
        tqc = transpile(circuit, backend, optimization_level=0)
        tqc.save_density_matrix()
        rho = np.asarray(backend.run(tqc).result().data(0)['density_matrix'])

        # Build S (Overlap) and H (Hamiltonian/Observable) Matrices
        S = np.zeros((k, k), dtype=complex)
        H = np.zeros((k, k), dtype=complex)
        H_op = observable.to_matrix()
        pool_ops = [p.to_matrix() for p in pool]

        for i in range(k):
            Pi_dag = pool_ops[i].conj().T
            for j in range(k):
                Pj = pool_ops[j]
                S[i, j] = np.trace(rho @ Pi_dag @ Pj)
                H[i, j] = np.trace(rho @ Pi_dag @ H_op @ Pj)

        # Enforce exact Hermiticity
        S = 0.5 * (S + S.conj().T)
        H = 0.5 * (H + H.conj().T)

        # Canonical Orthogonalization
        s_eigvals, s_eigvecs = np.linalg.eigh(S)
        order = np.argsort(s_eigvals)[::-1]
        s_eigvals, s_eigvecs = s_eigvals[order], s_eigvecs[:, order]

        threshold = self.eigenvalue_cutoff * max(s_eigvals[0], 1e-300)
        keep = s_eigvals > threshold

        if np.sum(keep) == 0:
            raise RuntimeError("All pool directions discarded. Lower cutoff or check noise levels.")

        X = s_eigvecs[:, keep] @ np.diag(1.0 / np.sqrt(s_eigvals[keep]))
        H_prime = X.conj().T @ H @ X
        H_prime = 0.5 * (H_prime + H_prime.conj().T)

        # Solve the regularized eigenproblem
        eigenvalues = np.linalg.eigvalsh(H_prime)

        # Extract Energies (SE strictly takes the lowest eigenvalue for ground states)
        raw_energy = float(np.real(H[0, 0]))
        mitigated_energy = float(np.real(eigenvalues[0]))

        return {
            "mitigated_value": mitigated_energy,
            "noisy_value": raw_energy,
            "kept_dimension": int(np.sum(keep))
        }
