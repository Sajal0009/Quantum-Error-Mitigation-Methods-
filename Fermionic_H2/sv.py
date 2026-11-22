'Measurement-based symmetry verification with QWC grouping.'

from typing import Optional, Sequence
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator

class SymmetryVerifiedMitigator:
    def __init__(
        self,
        symmetry_type: str = "none",
        expected_sector: int = 1,
        shots: int = 8192,
        symmetry_operators: Optional[Sequence] = None,
        seed_simulator: Optional[int] = None,
    ):
        if symmetry_type not in {"none", "parity", "hamming", "pauli"}:
            raise ValueError("symmetry_type must be none, parity, hamming, or pauli")
        if shots <= 0:
            raise ValueError("shots must be positive")
        self.symmetry_type = symmetry_type
        self.expected_sector = int(expected_sector)
        self.shots = int(shots)
        self.seed_simulator = seed_simulator
        self.symmetry_operators = self._normalise_symmetries(symmetry_operators or [])

        if symmetry_type == "pauli" and not self.symmetry_operators:
            raise ValueError("Provide symmetry_operators for symmetry_type='pauli'")
        if symmetry_type == "parity" and self.expected_sector not in (-1, 1):
            raise ValueError("Parity expected_sector must be +1 or -1")

    @staticmethod
    def _normalise_symmetries(items):
        out = []
        for item in items:
            if isinstance(item, dict):
                label, eig = str(item["pauli"]).upper(), int(item["eigenvalue"])
            else:
                label, eig = str(item[0]).upper(), int(item[1])
            if not label or any(c not in "IXYZ" for c in label) or eig not in (-1, 1):
                raise ValueError(f"Invalid Pauli symmetry: {(label, eig)}")
            out.append((label, eig))
        return out

    @staticmethod
    def _qchar(label, q):
        # Qiskit Pauli labels are written q[n-1]...q[0].
        return label[-1-q]

    @classmethod
    def _groups(cls, observable):
        terms = []
        for p, c in zip(observable.paulis, observable.coeffs):
            c = complex(c)
            if abs(c.imag) > 1e-10:
                raise ValueError("Hamiltonian coefficients must be real.")
            terms.append((p.to_label(), float(c.real)))

        groups, bases = [], []
        for label, coeff in terms:
            for k, basis in enumerate(bases):
                if all(
                    cls._qchar(label, q) == "I"
                    or basis[q] == "I"
                    or cls._qchar(label, q) == basis[q]
                    for q in range(len(label))
                ):
                    groups[k].append((label, coeff))
                    for q in range(len(label)):
                        p = cls._qchar(label, q)
                        if basis[q] == "I" and p != "I":
                            basis[q] = p
                    break
            else:
                basis = [cls._qchar(label, q) for q in range(len(label))]
                groups.append([(label, coeff)])
                bases.append(basis)
        return groups, bases

    @classmethod
    def _merge_basis(cls, group_basis, symmetries):
        basis = list(group_basis)
        compatible = []
        for label, eig in symmetries:
            candidate = list(basis)
            for q in range(len(label)):
                p = cls._qchar(label, q)
                if p != "I" and candidate[q] not in ("I", p):
                    break
                if p != "I":
                    candidate[q] = p
            else:
                basis = candidate
                compatible.append((label, eig))
        return basis, compatible

    @staticmethod
    def _rotate(qc, basis):
        for q, p in enumerate(basis):
            if p == "X":
                qc.h(q)
            elif p == "Y":
                qc.sdg(q)
                qc.h(q)

    @staticmethod
    def _bit(bits, q):
        # For measure_all() with one classical register, q0 is rightmost.
        return int(bits.replace(" ", "")[-1-q])

    @classmethod
    def _eigenvalue(cls, bits, label):
        value = 1
        for q in range(len(label)):
            if cls._qchar(label, q) != "I" and cls._bit(bits, q):
                value *= -1
        return value

    @classmethod
    def _energy_stats(cls, counts, terms, nqubits):
        n = sum(counts.values())
        if n == 0:
            return np.nan, np.nan, 0
        samples = []
        for bits, count in counts.items():
            e = sum(c * cls._eigenvalue(bits, label) for label, c in terms)
            samples.extend([e] * count)
        x = np.asarray(samples, dtype=float)
        se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        return float(np.mean(x)), se, n

    def _symmetries(self, n):
        if self.symmetry_type == "parity":
            return [("Z" * n, self.expected_sector)]
        if self.symmetry_type == "pauli":
            return self.symmetry_operators
        return []

    @staticmethod
    def _ideal_value(circuit, observable):
        try:
            clean = circuit.remove_final_measurements(inplace=False)
            value = complex(Statevector.from_instruction(clean).expectation_value(observable))
            return float(value.real) if abs(value.imag) < 1e-9 else None
        except Exception:
            return None

    def run(self, circuit: QuantumCircuit, observable: SparsePauliOp, noise_model=None):
        if not isinstance(observable, SparsePauliOp):
            raise TypeError("observable must be a SparsePauliOp")
        if circuit.num_qubits != observable.num_qubits:
            raise ValueError("Circuit and observable must have the same number of qubits.")

        nqubits = circuit.num_qubits
        groups, bases = self._groups(observable)
        backend = AerSimulator(noise_model=noise_model)
        diagnostics = []
        raw_energy, mitigated_energy = 0.0, 0.0
        raw_var, mitigated_var = 0.0, 0.0
        total_shots = accepted_total = 0

        for gi, (terms, group_basis) in enumerate(zip(groups, bases)):
            measure_basis, compatible = self._merge_basis(
                group_basis, self._symmetries(nqubits)
            )
            hamming_ok = (
                self.symmetry_type == "hamming"
                and all(p in ("I", "Z") for p in measure_basis)
            )

            qc = circuit.copy()
            self._rotate(qc, measure_basis)
            qc.measure_all()
            tqc = transpile(qc, backend, optimization_level=0)
            kwargs = {"shots": self.shots}
            if self.seed_simulator is not None:
                kwargs["seed_simulator"] = self.seed_simulator + gi
            counts = backend.run(tqc, **kwargs).result().get_counts()

            raw_e, raw_se, nraw = self._energy_stats(counts, terms, nqubits)

            if self.symmetry_type == "none":
                kept = dict(counts)
                checks = []
            elif self.symmetry_type == "hamming":
                if hamming_ok:
                    kept = {
                        b: c for b, c in counts.items()
                        if sum(self._bit(b, q) for q in range(nqubits)) == self.expected_sector
                    }
                    checks = [f"Hamming weight={self.expected_sector}"]
                else:
                    # Rotated X/Y outcomes are not occupation bitstrings.
                    kept, checks = dict(counts), []
            else:
                checks = compatible
                kept = {
                    b: c for b, c in counts.items()
                    if all(self._eigenvalue(b, label) == eig for label, eig in checks)
                }

            mit_e, mit_se, nkeep = self._energy_stats(kept, terms, nqubits)
            raw_energy += raw_e
            mitigated_energy += mit_e
            if np.isfinite(raw_se):
                raw_var += raw_se ** 2
            if np.isfinite(mit_se):
                mitigated_var += mit_se ** 2

            total_shots += nraw
            accepted_total += nkeep
            diagnostics.append({
                "group": gi,
                "pauli_terms": [label for label, _ in terms],
                "measurement_basis_q0_to_qn": "".join(measure_basis),
                "verification_applied": bool(checks),
                "active_symmetries": checks,
                "shots": nraw,
                "accepted_shots": nkeep,
                "rejected_shots": nraw - nkeep,
                "survival_fraction": nkeep / nraw if nraw else 0.0,
                "raw_energy_contribution": raw_e,
                "mitigated_energy_contribution": mit_e,
                "raw_standard_error": raw_se,
                "mitigated_standard_error": mit_se,
            })

        return {
            "ideal_value": self._ideal_value(circuit, observable),
            "noisy_value": float(raw_energy),
            "mitigated_value": float(mitigated_energy),
            "raw_standard_error": float(np.sqrt(raw_var)),
            "standard_error": float(np.sqrt(mitigated_var)),
            "survival_fraction": accepted_total / total_shots if total_shots else 0.0,
            "accepted_shots": int(accepted_total),
            "rejected_shots": int(total_shots - accepted_total),
            "total_shots": int(total_shots),
            "num_measurement_groups": len(groups),
            "num_groups_with_verification": sum(d["verification_applied"] for d in diagnostics),
            "group_diagnostics": diagnostics,
        }

# ---------------- Circuit-specific configuration helpers ----------------

def make_fermionic_h2_sv(shots=8192, seed_simulator=None):
    """Exact N=2 postselection, only on Z/occupation-basis groups."""
    return SymmetryVerifiedMitigator(
        symmetry_type="hamming", expected_sector=2,
        shots=shots, seed_simulator=seed_simulator
    )

def make_pauli_sv(symmetry_operators, shots=8192, seed_simulator=None):
    'Generic Pauli symmetry configuration. '
    return SymmetryVerifiedMitigator(
        symmetry_type="pauli", symmetry_operators=symmetry_operators,
        shots=shots, seed_simulator=seed_simulator
    )

def make_parity_sv(expected_parity=1, shots=8192, seed_simulator=None):
    """Global Z-parity check, used only on QWC-compatible measurement groups."""
    return SymmetryVerifiedMitigator(
        symmetry_type="parity", expected_sector=expected_parity,
        shots=shots, seed_simulator=seed_simulator
    )

def make_no_postselection_sv(shots=8192, seed_simulator=None):
    """Full-Hamiltonian measurement baseline without symmetry filtering."""
    return SymmetryVerifiedMitigator(
        symmetry_type="none", shots=shots, seed_simulator=seed_simulator
    )
