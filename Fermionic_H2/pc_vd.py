import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

class VDMitigator:
    def __init__(self):
        pass

    def run(self, circuit: QuantumCircuit, observable: SparsePauliOp, noise_model) -> dict:
        """Executes exact Virtual Distillation (M=2) via Density Matrix simulation."""
        backend = AerSimulator(noise_model=noise_model, method="density_matrix")

        tqc = transpile(circuit, backend, optimization_level=0)
        tqc.save_density_matrix()

        result = backend.run(tqc).result()
        rho = np.asarray(result.data(0)['density_matrix'])

        O_matrix = observable.to_matrix()

        # M=1: Standard noisy expectation value
        ev_1 = np.real(np.trace(O_matrix @ rho))

        # M=2: Purity-weighted distilled expectation value
        rho2 = rho @ rho
        purity = np.real(np.trace(rho2))
        ev_2 = np.real(np.trace(O_matrix @ rho2)) / purity if purity > 0 else 0.0

        return {
            "mitigated_value": float(ev_2),
            "noisy_value": float(ev_1),
            "purity": float(purity)
        }
