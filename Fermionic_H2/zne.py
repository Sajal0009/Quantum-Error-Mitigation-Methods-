import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import EstimatorV2 as AerEstimator

def fold_gates_local(circuit: QuantumCircuit, scale_factor: float) -> QuantumCircuit:
    if scale_factor < 1:
        raise ValueError("scale_factor must be >= 1")

    gates= list(circuit.data)
    n =len(gates)
    if n == 0:
        return circuit.copy()

    k =int((scale_factor -1)/2)
    r =int(round((scale_factor-(2*k+1))*n/2))

    folded = QuantumCircuit(*circuit.qregs, *circuit.cregs)

    for idx, inst in enumerate(gates):
        op, qargs, cargs = inst.operation, inst.qubits, inst.clbits
        folded.append(op, qargs, cargs)

        for _ in range(k):
            folded.append(op.inverse(), qargs, cargs)
            folded.append(op, qargs, cargs)

        if idx >= n - r and r > 0:
            folded.append(op.inverse(), qargs, cargs)
            folded.append(op, qargs, cargs)

    return folded

def fold_circuit_global(circuit: QuantumCircuit, scale_factor: int) -> QuantumCircuit:
    if scale_factor < 1 or scale_factor %2 == 0:
        raise ValueError("Global folding requires odd integer scale factors: 1, 3, 5, ...")

    k = (scale_factor -1)//2
    inv = circuit.inverse()

    folded = circuit.copy()
    for _ in range(k):
        folded = folded.compose(inv)
        folded = folded.compose(circuit)

    return folded

def estimate_expectation_value(
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
    noise_model: NoiseModel,
    shots: int = 8192
) -> float:
    backend = AerSimulator(noise_model=noise_model)
    tcirc = transpile(circuit, backend, optimization_level=0)

    opts = {"run_options": {"shots": shots}}
    estimator = AerEstimator.from_backend(backend, options=opts)

    pub = (tcirc, observable)
    result = estimator.run([pub]).result()

    return float(result[0].data.evs)

def collect_noise_scaled_expectations(
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
    scale_factors: list,
    noise_model: NoiseModel,
    shots: int = 8192,
    fold_method: str = "local"
) -> tuple[np.ndarray, np.ndarray]:
    expectations = []
    for c in scale_factors:
        if fold_method == "local":
            folded = fold_gates_local(circuit, c)
        else:
            c_odd = int(c) if int(c) % 2 == 1 else int(c) + 1
            folded = fold_circuit_global(circuit, c_odd)

        ev = estimate_expectation_value(folded, observable, noise_model, shots)
        expectations.append(ev)
        print(f"    c = {c:.1f}  ->  <O> = {ev:.6f}")

    return np.array(scale_factors, dtype=float), np.array(expectations)

def richardson_extrapolation(
    scale_factors: np.ndarray,
    expectations: np.ndarray
) -> tuple[float, np.ndarray]:
    c = scale_factors.astype(float)
    y = expectations.astype(float)
    m = len(c)

    gamma = np.ones(m)
    for i in range(m):
        for j in range(m):
            if j != i:
                gamma[i] *= c[j] / (c[j] - c[i])

    mitigated = float(np.dot(gamma, y))
    return mitigated, gamma

def polynomial_extrapolation(
    scale_factors: np.ndarray,
    expectations: np.ndarray,
    degree: int = None
) -> float:
    if degree is None:
        degree = len(scale_factors) - 1

    coeffs = np.polyfit(scale_factors, expectations, degree)
    poly = np.poly1d(coeffs)
    return float(poly(0))

def exponential_extrapolation(
    scale_factors: np.ndarray,
    expectations: np.ndarray
) -> tuple[float, dict]:
    def exp_model(c, A, B, Gamma):
        return A + B * np.exp(-Gamma * c)

    y = expectations
    c = scale_factors
    p0 = [y.min(), y.max() - y.min(), 0.5]

    try:
        popt, pcov = curve_fit(exp_model, c, y, p0=p0, maxfev=10000)
        A, B, Gamma = popt
        mitigated = float(A + B)
        params = {"A": A, "B": B, "Gamma": Gamma}
        return mitigated, params
    except RuntimeError:
        print("    WARNING: exp fit failed, falling back to linear extrapolation")
        return polynomial_extrapolation(scale_factors, expectations, degree=1), {}

class ZNEMitigator:
    def __init__(
        self,
        scale_factors: list = [1, 2, 3],
        extrapolator: str = "richardson",
        poly_degree: int = None,
        fold_method: str = "local",
        shots: int = 8192
    ):
        if 1 not in scale_factors and 1.0 not in scale_factors:
            raise ValueError("scale_factors must include 1 (unscaled native noise)")

        self.scale_factors = sorted(scale_factors)
        self.extrapolator = extrapolator
        self.poly_degree = poly_degree
        self.fold_method = fold_method
        self.shots = shots

        self._last_c = None
        self._last_y = None
        self._last_fit = None

    def run(
        self,
        circuit: QuantumCircuit,
        observable: SparsePauliOp,
        noise_model: NoiseModel
    ) -> dict:
        sep = "=" * 64
        print(f"\n{sep}")
        print(f"  ZNE Pipeline")
        print(f"  Extrapolator : {self.extrapolator}")
        print(f"  Fold method  : {self.fold_method}")
        print(f"  Scale factors: {self.scale_factors}")
        print(f"  Shots        : {self.shots}")
        print(sep)

        sv = Statevector(circuit)
        ideal = float(sv.expectation_value(observable).real)
        print(f"\n  Ideal <O> (noiseless statevector) : {ideal:.6f}")

        print(f"\n  Collecting expectation values:")
        c_arr, y_arr = collect_noise_scaled_expectations(
            circuit, observable, self.scale_factors,
            noise_model, self.shots, self.fold_method
        )
        noisy_raw = y_arr[np.argmin(np.abs(c_arr - 1.0))]
        print(f"\n  Noisy <O> at c=1 (unmitigated)    : {noisy_raw:.6f}")

        self._last_c = c_arr
        self._last_y = y_arr

        gamma = None

        if self.extrapolator == "richardson":
            mitigated, gamma = richardson_extrapolation(c_arr, y_arr)
            print(f"\n  Richardson weights gamma : {np.round(gamma, 4)}")
            print(f"  Sum(gamma_i) (should be 1)    : {gamma.sum():.6f}")
            print(f"  Sum(gamma_i^2) (variance factor): {(gamma**2).sum():.4f}")

        elif self.extrapolator == "polynomial":
            deg = self.poly_degree or len(c_arr) - 1
            mitigated = polynomial_extrapolation(c_arr, y_arr, degree=deg)
            print(f"\n  Polynomial degree: {deg}")

        elif self.extrapolator == "exponential":
            mitigated, params = exponential_extrapolation(c_arr, y_arr)
            self._last_fit = params
            print(f"\n  Exponential fit params: {params}")

        else:
            raise ValueError(f"Unknown extrapolator '{self.extrapolator}'")

        noisy_err = abs(noisy_raw - ideal)
        mitigated_err = abs(mitigated - ideal)
        improvement = ((noisy_err - mitigated_err) / noisy_err * 100
                        if noisy_err > 1e-12 else 0.0)

        print(f"\n{'-'*64}")
        print(f"  Ideal      : {ideal:.6f}")
        print(f"  Noisy      : {noisy_raw:.6f}   |error| = {noisy_err:.6f}")
        print(f"  ZNE result : {mitigated:.6f}   |error| = {mitigated_err:.6f}")
        print(f"  Error reduction : {improvement:.1f}%")
        print(f"{'-'*64}\n")

        return {
            "mitigated_value": mitigated,
            "ideal_value": ideal,
            "noisy_value": noisy_raw,
            "scale_factors": c_arr,
            "raw_values": y_arr,
            "error_reduction": improvement,
            "gamma_weights": gamma,
        }

    def plot_extrapolation(
        self,
        ideal_value: float = None,
        title: str = "ZNE Result",
        save_path: str = "zne_plot.png"
    ):
        if self._last_c is None:
            raise RuntimeError("Call .run() before plotting.")

        c = self._last_c
        y = self._last_y
        c_dense = np.linspace(0, max(c), 100)

        plt.figure(figsize=(6, 4))

        # Plot the ideal expectation value as a reference line
        if ideal_value is not None:
            plt.axhline(ideal_value, color='black', linestyle='--', label='Ideal')

        # Calculate and plot the fitted extrapolation curve
        if self.extrapolator in ("richardson", "polynomial"):
            deg = self.poly_degree or len(c) - 1
            coeffs = np.polyfit(c, y, deg)
            poly = np.poly1d(coeffs)
            y_fit = poly(c_dense)
            y_at_0 = float(poly(0))
            plt.plot(c_dense, y_fit, color='gray', linestyle='-', alpha=0.7, label='Fit')

        elif self.extrapolator == "exponential" and self._last_fit:
            p = self._last_fit
            y_fit = p["A"] + p["B"] * np.exp(-p["Gamma"] * c_dense)
            y_at_0 = float(p["A"] + p["B"])
            plt.plot(c_dense, y_fit, color='gray', linestyle='-', alpha=0.7, label='Fit')

        else:
            y_at_0 = polynomial_extrapolation(c, y, degree=len(c)-1)

        # Plot the raw noisy data points
        plt.plot(c, y, 'o', color='blue', label='Noisy')

        # Plot the ZNE mitigated point at c=0
        plt.plot(0, y_at_0, 'o', color='red', label='ZNE')

        # Clean, human-readable formatting
        plt.xlabel("Noise Scale Factor")
        plt.ylabel("Expectation Value")
        plt.title(title)
        plt.legend()
        plt.tight_layout()

        plt.savefig(save_path, dpi=300)
        plt.show()
        print(f"Plot saved -> {save_path}")
