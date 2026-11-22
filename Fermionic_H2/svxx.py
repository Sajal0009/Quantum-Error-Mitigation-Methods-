import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

class SymmetryVerifiedMitigator:
    def __init__(self, symmetry_type: str = "parity", expected_sector: int = 1, shots: int = 8192):
        self.symmetry_type = symmetry_type
        self.expected_sector = expected_sector
        self.shots = shots

    def _get_label(self, bitstring: str) -> int:
        """Evaluates the symmetry label of a given measured bitstring."""
        bits = bitstring.replace(" ", "")

        if self.symmetry_type == "parity":
            # X-Parity: XX = +1 for equal bits, -1 for different bits
            return 1 if bits[-1] == bits[-2] else -1

        elif self.symmetry_type == "hamming":
            # Particle Conservation: Exact count of 1s
            return bits.count('1')

        return 1

    def _compute_ev(self, counts: dict, pauli_string: str) -> float:
        """Calculates the ZZ expectation value from Bell-basis counts."""
        total = sum(counts.values())
        if total == 0:
            return 0.0

        ev = 0.0

        for bitstring, count in counts.items():
            bits = bitstring.replace(" ", "")

            # In Bell-basis measurement:
            # left bit  -> ZZ eigenvalue
            parity = 1
            if pauli_string == "ZZ":
                if bits[-2] == '1':
                    parity *= -1

            ev += parity * count / total

        return ev

    def run(self, circuit: QuantumCircuit, observable: SparsePauliOp, noise_model) -> dict:
        backend = AerSimulator(noise_model=noise_model)

        # Execute noisy circuit
        qc = circuit.copy()

        # Bell-basis measurement for XX symmetry and ZZ observable
        qc.cx(0, 1)
        qc.h(0)
        qc.measure_all()

        tqc = transpile(qc, backend, optimization_level=0)
        raw_counts = backend.run(tqc, shots=self.shots).result().get_counts()

        # Filter counts based on the expected XX symmetry sector
        verified_counts = {
            bs: count for bs, count in raw_counts.items()
            if self._get_label(bs) == self.expected_sector
        }

        kept_shots = sum(verified_counts.values())
        survival_fraction = kept_shots / self.shots if self.shots > 0 else 0.0

        # Calculate expectation values
        pauli_str = observable.paulis[0].to_label()
        raw_ev = self._compute_ev(raw_counts, pauli_str)
        sv_ev = self._compute_ev(verified_counts, pauli_str) if kept_shots > 0 else 0.0

        return {
            "mitigated_value": float(sv_ev),
            "noisy_value": float(raw_ev),
            "survival_fraction": float(survival_fraction)
        }
