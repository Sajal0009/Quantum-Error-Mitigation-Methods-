import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

class MEMitigator:
    def __init__(self, num_qubits: int, shots: int = 8192):
        self.num_qubits = num_qubits
        self.shots = shots
        self.calibration_matrices = None

    def calibrate(self, backend: AerSimulator):
        """Calibrate each qubit's readout matrix independently."""
        self.calibration_matrices = []

        for q in range(self.num_qubits):
            A_q = np.zeros((2, 2), dtype=float)

            for true_bit in (0, 1):
                qc = QuantumCircuit(self.num_qubits, 1)

                if true_bit == 1:
                    qc.x(q)

                qc.measure(q, 0)
                tqc = transpile(
                    qc, backend, optimization_level=0
                )
                counts = backend.run(
                    tqc, shots=self.shots
                ).result().get_counts()

                for outcome, count in counts.items():
                    A_q[int(outcome), true_bit] = (
                        count / self.shots
                    )

            self.calibration_matrices.append(A_q)

    def run(
        self,
        circuit: QuantumCircuit,
        observable: SparsePauliOp,
        noise_model
    ) -> dict:
        """Estimate a diagonal I/Z observable with readout mitigation."""
        if circuit.num_qubits != self.num_qubits:
            raise ValueError("Circuit qubit count mismatch.")

        backend = AerSimulator(noise_model=noise_model)

        if self.calibration_matrices is None:
            self.calibrate(backend)

        # Execute the circuit in the computational basis.
        qc = circuit.copy()
        qc.measure_all()
        tqc = transpile(qc, backend, optimization_level=0)

        counts = backend.run(
            tqc, shots=self.shots
        ).result().get_counts()

        # Qiskit count strings use q_(n-1)...q_0 ordering.
        dim = 2 ** self.num_qubits
        p_noisy = np.zeros(dim, dtype=float)

        for bitstring, count in counts.items():
            bits = bitstring.replace(" ", "")
            idx = int(bits, 2)
            p_noisy[idx] += count / self.shots

        # Matrix order must match q_(n-1)...q_0 in count strings.
        A_inv_full = np.array([[1.0]])

        for q in reversed(range(self.num_qubits)):
            A_inv_q = np.linalg.inv(
                self.calibration_matrices[q]
            )
            A_inv_full = np.kron(A_inv_full, A_inv_q)

        p_corrected = A_inv_full @ p_noisy

        # Do not clip or renormalize the corrected distribution:
        # linear inversion can produce negative quasi-probabilities.
        noisy_ev = 0.0
        mitigated_ev = 0.0

        for label, coeff in zip(
            observable.paulis.to_labels(),
            observable.coeffs
        ):
            if set(label) - {"I", "Z"}:
                raise ValueError(
                    "MEMitigator.run requires an I/Z observable. "
                    "Rotate X/Y terms to the Z basis first."
                )

            coeff = float(np.real(coeff))
            term_noisy = 0.0
            term_mitigated = 0.0

            for idx in range(dim):
                bits = format(idx, f"0{self.num_qubits}b")

                # label and bits both follow q_(n-1)...q_0.
                eigenvalue = 1
                for bit, pauli in zip(bits, label):
                    if pauli == "Z" and bit == "1":
                        eigenvalue *= -1

                term_noisy += eigenvalue * p_noisy[idx]
                term_mitigated += (
                    eigenvalue * p_corrected[idx]
                )

            noisy_ev += coeff * term_noisy
            mitigated_ev += coeff * term_mitigated

        return {
            "noisy_value": float(noisy_ev),
            "mitigated_value": float(mitigated_ev)
        }
