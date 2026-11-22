from __future__ import annotations

import json
import os
import warnings
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from benchmark.lih_sqd import (
    build_lih_cas22,
    make_lih_hamiltonian,
    run_sqd_from_counts,
)
from benchmark.models import build_noise_model
from pc_vd import virtual_distillation
from se import subspace_expansion, number_conserving_pool
from sv import symmetry_verify
from pec2 import invert_global_depolarizing
from nrep import NRepMitigator
from zne import fold_global, richardson_extrapolate
from mem import mitigate_probabilities

SHOTS = 2048
SEED = 17
BOND_LENGTH_ANGSTROM = 1.60
BASIS = "sto-3g"
NOISE_P1 = 0.001
NOISE_P2 = 0.010
READOUT_ERROR = 0.020
ZNE_SCALES = [1, 3, 5]
OUTPUT_DIR = "results_final"

def make_ansatz(n_qubits: int = 4) -> QuantumCircuit:

    qc = QuantumCircuit(n_qubits, name="LiH_CAS22_trial_ansatz")
    
    qc.x(0)
    qc.x(2)
    for q, theta in enumerate((0.24, 0.41, 0.24, 0.41)):
        qc.ry(theta, q)
    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.cx(1, 2)
    qc.ry(0.17, 0)
    qc.ry(-0.13, 1)
    qc.ry(0.17, 2)
    qc.ry(-0.13, 3)
    return qc

def density_matrix_from_circuit(
    circuit: QuantumCircuit,
    noise_model=None,
) -> np.ndarray:
    backend = AerSimulator(method="density_matrix", noise_model=noise_model)
    qc = circuit.copy()
    qc.save_density_matrix()
    tqc = transpile(qc, backend, optimization_level=0)
    result = backend.run(
        tqc, shots=1, seed_simulator=SEED
    ).result()
    return np.asarray(result.data(0)["density_matrix"], dtype=complex)

def sampled_counts(
    circuit: QuantumCircuit,
    noise_model,
    shots: int = SHOTS,
    seed: int = SEED,
) -> dict:
    backend = AerSimulator(noise_model=noise_model)
    qc = circuit.copy()
    qc.measure_all()
    tqc = transpile(qc, backend, optimization_level=0)
    return backend.run(
        tqc, shots=shots, seed_simulator=seed
    ).result().get_counts()

def normalize_rho(rho: np.ndarray) -> np.ndarray:
    rho = np.asarray(rho, dtype=complex)
    rho = 0.5 * (rho + rho.conj().T)
    tr = np.trace(rho)
    if abs(tr) < 1e-14:
        raise ValueError("Cannot normalize a density matrix with zero trace.")
    rho = rho / tr
    return rho

def energy_from_rho(rho: np.ndarray, h_matrix: np.ndarray) -> float:
    return float(np.real(np.trace(h_matrix @ normalize_rho(rho))))

def particle_number_projector(
    n_qubits: int = 4,
    n_alpha: int = 1,
    n_beta: int = 1,
) -> np.ndarray:

    dim = 2**n_qubits
    diagonal = np.zeros(dim, dtype=float)
    for basis_index in range(dim):
        alpha_count = (basis_index & 0b0011).bit_count()
        beta_count = ((basis_index >> 2) & 0b0011).bit_count()
        if alpha_count == n_alpha and beta_count == n_beta:
            diagonal[basis_index] = 1.0
    return np.diag(diagonal).astype(complex)

def apply_symmetry_verification(rho: np.ndarray) -> tuple[np.ndarray, float]:
    P = particle_number_projector()
    projected = P @ rho @ P
    acceptance = float(np.real(np.trace(projected)))
    if acceptance <= 1e-14:
        raise RuntimeError("No population survived the target particle-number sector.")
    return normalize_rho(projected), acceptance

def apply_virtual_distillation(rho: np.ndarray) -> tuple[np.ndarray, float]:

    rho = normalize_rho(rho)
    rho2 = rho @ rho
    purity = float(np.real(np.trace(rho2)))
    if purity <= 1e-14:
        raise RuntimeError("Virtual-distillation purity is numerically zero.")
    return normalize_rho(rho2), purity

def apply_pec_surrogate(
    rho: np.ndarray,
    p_effective: float,
) -> tuple[np.ndarray, np.ndarray, dict]:

    pec = invert_global_depolarizing(normalize_rho(rho), p_effective)
    rho_linear = normalize_rho(pec["rho_linear_inverse"])
    rho_physical = normalize_rho(pec["rho_physical_projected"])
    return rho_linear, rho_physical, pec

def run_subspace_expansion(
    rho: np.ndarray,
    h_matrix: np.ndarray,
    pool,
) -> dict:

    result = subspace_expansion(
        normalize_rho(rho), h_matrix, pool
    )
    if isinstance(result, dict):
        return result
    return {"mitigated_value": float(result)}

def get_energy_from_method_result(result: Any) -> float:

    if isinstance(result, (int, float, np.floating)):
        return float(result)
    if isinstance(result, dict):
        for key in (
            "mitigated_value",
            "energy",
            "energy_total",
            "projected_energy",
        ):
            value = result.get(key)
            if isinstance(value, (int, float, np.floating)):
                return float(value)
    raise ValueError(
        "Could not extract an energy from method result. "
        f"Returned object: {type(result).__name__}: {result}"
    )

def add_energy_result(
    rows: list,
    name: str,
    energy: float,
    category: str,
    description: str,
    ideal_energy: float,
    extra: dict | None = None,
) -> None:
    row = {
        "Workflow": name,
        "Category": category,
        "Energy (Ha)": float(energy),
        "Absolute error vs ideal ansatz (mHa)": abs(
            float(energy) - ideal_energy
        ) * 1000.0,
        "Description": description,
    }
    if extra:
        row.update(extra)
    rows.append(row)

def save_report(
    rows: list,
    ideal_energy: float,
    zne_scales: list,
    zne_energies: list,
    mem_diagnostics: dict,
    nrep_diagnostics: dict,
    zne_combo_results: dict | None = None,
) -> pd.DataFrame:

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.DataFrame(rows)
    df = df.sort_values(
        "Absolute error vs ideal ansatz (mHa)",
        ascending=True,
        kind="stable",
    ).reset_index(drop=True)

    
    csv_path = os.path.join(OUTPUT_DIR, "lih_qem_energy_comparison.csv")
    xlsx_path = os.path.join(OUTPUT_DIR, "lih_qem_energy_comparison.xlsx")
    json_path = os.path.join(OUTPUT_DIR, "lih_qem_results.json")
    df.to_csv(csv_path, index=False)
    try:
        df.to_excel(xlsx_path, index=False)
    except Exception as exc:
        print("Excel export unavailable; CSV is saved. Reason:", repr(exc))

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "reference": {
                    "name": "Noiseless expectation of the same trial ansatz",
                    "energy_hartree": float(ideal_energy),
                },
                "energy_results": rows,
                "zne": {
                    "scale_factors": zne_scales,
                    "energies_hartree": zne_energies,
                },
                "mem_population_diagnostic": mem_diagnostics,
                "n_representability_diagnostic": nrep_diagnostics,
                "zne_combination_details": zne_combo_results or {},
            },
            f,
            indent=2,
            default=lambda x: float(x) if isinstance(x, np.generic) else str(x),
        )

    
    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 19,
        "axes.titleweight": "bold",
        "axes.labelsize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "figure.titlesize": 21,
        "savefig.dpi": 350,
    })

    plot_df = df.copy()
    plot_df["Signed deviation (mHa)"] = (
        plot_df["Energy (Ha)"].astype(float) - float(ideal_energy)
    ) * 1000.0

    
    
    plot_df = plot_df.sort_values("Signed deviation (mHa)", kind="stable")
    x = np.arange(len(plot_df))
    deviations = plot_df["Signed deviation (mHa)"].to_numpy(dtype=float)

    fig_width = max(15, 0.95 * len(plot_df))
    fig, ax = plt.subplots(figsize=(fig_width, 9))
    colors = [
        "tab:green" if c == "Reference" else
        "tab:orange" if c == "Combination" else
        "tab:blue" if c == "Individual" else "tab:gray"
        for c in plot_df["Category"]
    ]
    bars = ax.bar(x, deviations, width=0.72, color=colors)
    ax.axhline(
        0.0, color="black", linewidth=1.8, linestyle="--",
        label="Ideal ansatz reference (0 mHa deviation)",
    )
    ax.set_yscale("symlog", linthresh=1.0, linscale=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["Workflow"], rotation=48, ha="right")
    ax.set_ylabel("Energy deviation from ideal ansatz (mHa)", labelpad=12)
    ax.set_title("LiH CAS(2,2): QEM energy comparison", pad=18)
    ax.grid(axis="y", which="both", alpha=0.28)
    ax.legend(loc="best")
    ax.tick_params(axis="both", which="major", labelsize=12)
    for bar, value in zip(bars, deviations):
        offset = 5 if value >= 0 else -7
        ax.annotate(
            f"{value:+.3g}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
            rotation=0,
        )
    fig.tight_layout()
    energy_fig_path = os.path.join(
        OUTPUT_DIR, "lih_qem_energy_comparison.png"
    )
    fig.savefig(energy_fig_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    
    
    errors = df["Absolute error vs ideal ansatz (mHa)"].to_numpy(dtype=float)
    positive_errors = errors[errors > 0]
    plot_floor = (
        max(float(np.min(positive_errors)) * 0.35, 1e-8)
        if positive_errors.size else 1e-8
    )
    plotted_errors = np.where(errors > 0, errors, plot_floor)

    fig, ax = plt.subplots(figsize=(fig_width, 9))
    bar_colors = [
        "tab:green" if c == "Reference" else
        "tab:orange" if c == "Combination" else
        "tab:blue" if c == "Individual" else "tab:gray"
        for c in df["Category"]
    ]
    bars = ax.bar(
        np.arange(len(df)), plotted_errors, width=0.72, color=bar_colors
    )
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(df["Workflow"], rotation=48, ha="right")
    ax.set_ylabel("Absolute error vs ideal ansatz (mHa)", labelpad=12)
    ax.set_title(
        "LiH CAS(2,2): QEM absolute error (logarithmic scale)", pad=18
    )
    ax.grid(axis="y", which="both", alpha=0.28)
    ax.tick_params(axis="both", which="major", labelsize=12)
    for bar, error in zip(bars, errors):
        label = "0" if error == 0 else f"{error:.3g}"
        ax.annotate(
            label,
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="tab:green", label="Reference"),
        Patch(color="tab:blue", label="Individual method"),
        Patch(color="tab:orange", label="Combination"),
    ], loc="best")
    fig.tight_layout()
    error_fig_path = os.path.join(
        OUTPUT_DIR, "lih_qem_absolute_error_log.png"
    )
    fig.savefig(error_fig_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    
    print("\n=== QEM ENERGY COMPARISON ===")
    print("Reference: noiseless expectation of the same trial ansatz.")
    try:
        from IPython.display import display
        display(df)
    except Exception:
        print(df.to_string(index=False))

    print(f"\nAll final outputs saved in: {OUTPUT_DIR}/")
    print(f"- {os.path.basename(csv_path)}")
    if os.path.exists(xlsx_path):
        print(f"- {os.path.basename(xlsx_path)}")
    print(f"- {os.path.basename(json_path)}")
    print(f"- {os.path.basename(energy_fig_path)}")
    print(f"- {os.path.basename(error_fig_path)}")
    return df

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("\n=== LiH CAS(2,2): STO-3G / Jordan-Wigner / 4 qubits ===")

    data = build_lih_cas22(
        bond_length=BOND_LENGTH_ANGSTROM,
        basis=BASIS,
    )
    H, H_pauli = make_lih_hamiltonian(data)
    H = np.asarray(H, dtype=complex)
    print(
        f"RHF converged; active orbitals={data['norb']}; "
        f"active electrons={sum(data['nelec'])}"
    )
    print(f"Pauli terms in mapped Hamiltonian: {len(H_pauli)}")

    circuit = make_ansatz(4)
    ideal_rho = density_matrix_from_circuit(circuit, None)
    ideal_energy = energy_from_rho(ideal_rho, H)

    noise = build_noise_model(p1=NOISE_P1, p2=NOISE_P2)
    noisy_rho = density_matrix_from_circuit(circuit, noise)
    noisy_energy = energy_from_rho(noisy_rho, H)

    print(f"Ideal ansatz expectation: {ideal_energy:.12f} Ha")
    print(f"Noisy expectation:        {noisy_energy:.12f} Ha")
    print("Reference for every error below: ideal ansatz expectation.")
    print("No FCI/CASCI energy is used in the results table or plots.")

    rows: list[dict] = []
    add_energy_result(
        rows, "Ideal ansatz reference", ideal_energy, "Reference",
        "Noiseless expectation of the same trial ansatz", ideal_energy,
    )
    add_energy_result(
        rows, "Noisy baseline", noisy_energy, "Baseline",
        "Noisy density-matrix expectation", ideal_energy,
    )

    
    rho_sv, sv_acceptance = apply_symmetry_verification(noisy_rho)
    sv_energy = energy_from_rho(rho_sv, H)
    add_energy_result(
        rows, "Symmetry verification", sv_energy, "Individual",
        "Project onto the (1,1) particle-number sector", ideal_energy,
        {"Sector acceptance": sv_acceptance},
    )

    
    rho_vd, vd_purity = apply_virtual_distillation(noisy_rho)
    vd_energy = energy_from_rho(rho_vd, H)
    add_energy_result(
        rows, "Virtual distillation", vd_energy, "Individual",
        "Exact M=2 density-matrix virtual distillation", ideal_energy,
        {"Purity": vd_purity},
    )

    
    pool = number_conserving_pool(4)
    se_result = run_subspace_expansion(noisy_rho, H, pool)
    se_energy = get_energy_from_method_result(se_result)
    add_energy_result(
        rows, "Subspace expansion", se_energy, "Individual",
        "Density-matrix subspace expansion", ideal_energy,
    )

    
    zne_energies = []
    for scale in ZNE_SCALES:
        folded = fold_global(circuit, scale)
        rho_scale = density_matrix_from_circuit(folded, noise)
        e_scale = energy_from_rho(rho_scale, H)
        zne_energies.append(e_scale)
        print(f"ZNE scale {scale}: {e_scale:.12f} Ha")
    zne_zero, zne_coeff = richardson_extrapolate(
        ZNE_SCALES, zne_energies, degree=2
    )
    add_energy_result(
        rows, "ZNE", zne_zero, "Individual",
        "Quadratic extrapolation of noisy energy at scales 1, 3, 5",
        ideal_energy,
    )

    
    n1 = sum(1 for inst in circuit.data if inst.operation.num_qubits == 1)
    n2 = sum(1 for inst in circuit.data if inst.operation.num_qubits == 2)
    p_effective = 1.0 - (1.0 - NOISE_P1) ** n1 * (1.0 - NOISE_P2) ** n2
    rho_pec_linear, rho_pec_physical, pec_info = apply_pec_surrogate(
        noisy_rho, p_effective
    )
    pec_energy = energy_from_rho(rho_pec_physical, H)
    add_energy_result(
        rows, "PEC", pec_energy, "Individual",
        "Physical projection of a global-depolarizing inverse surrogate",
        ideal_energy,
        {"Effective p (approx.)": p_effective},
    )

    
    
    counts_ro = sampled_counts(
        circuit,
        build_noise_model(0.0, 0.0, READOUT_ERROR),
        SHOTS,
        SEED + 10,
    )
    p_raw = np.zeros(16, dtype=float)
    for bitstring, count in counts_ro.items():
        p_raw[int(bitstring.replace(" ", ""), 2)] += count / SHOTS
    readout_matrix = np.array(
        [
            [1.0 - READOUT_ERROR, READOUT_ERROR],
            [READOUT_ERROR, 1.0 - READOUT_ERROR],
        ],
        dtype=float,
    )
    p_mem = mitigate_probabilities(p_raw, [readout_matrix] * 4)
    mem_diagnostics = {
        "raw_probability_sum": float(np.sum(p_raw)),
        "corrected_probability_sum": float(np.sum(p_mem)),
        "raw_distribution": p_raw.tolist(),
        "corrected_distribution": np.asarray(p_mem).tolist(),
        "scope": "Computational-basis population correction only; not a Hamiltonian energy estimate.",
    }

    
    
    nrep_diagnostics = {}
    try:
        nr = NRepMitigator(
            data["h1"],
            data["eri"],
            data["ecore"],
            shots=SHOTS,
            seed=SEED,
        )
        nr_result = nr.run_from_density_matrix(
            noisy_rho,
            add_sampling_noise=True,
        )
        nr_energy = float(nr_result["mitigated_value"])
        add_energy_result(
            rows, "N-representability", nr_energy, "Individual",
            "SDP projection of synthetically sampled RDMs", ideal_energy,
        )
        nrep_diagnostics = {
            "noisy_value": nr_result.get("noisy_value"),
            "mitigated_value": nr_energy,
            "scope": nr_result.get("scope", "Synthetic RDM sampling; simulator-only"),
        }
    except Exception as exc:
        nrep_diagnostics = {"status": "FAILED", "error": repr(exc)}
        print("N-representability stage failed:", repr(exc))

    
    
    
    
    print("\n=== Sequential QEM combinations ===")

    
    rho_sv_vd, purity_sv_vd = apply_virtual_distillation(rho_sv)
    e_sv_vd = energy_from_rho(rho_sv_vd, H)
    add_energy_result(
        rows, "SV → VD", e_sv_vd, "Combination",
        "Particle-number projection followed by M=2 virtual distillation",
        ideal_energy,
        {"SV acceptance": sv_acceptance, "Purity": purity_sv_vd},
    )

    
    se_sv = run_subspace_expansion(rho_sv, H, pool)
    e_sv_se = get_energy_from_method_result(se_sv)
    add_energy_result(
        rows, "SV → SE", e_sv_se, "Combination",
        "Subspace expansion applied to the symmetry-projected state",
        ideal_energy,
    )

    
    se_vd = run_subspace_expansion(rho_vd, H, pool)
    e_vd_se = get_energy_from_method_result(se_vd)
    add_energy_result(
        rows, "VD → SE", e_vd_se, "Combination",
        "Subspace expansion applied to the virtual-distilled state",
        ideal_energy,
    )

    
    _, rho_sv_pec_physical, pec_sv_info = apply_pec_surrogate(
        rho_sv, p_effective
    )
    e_sv_pec = energy_from_rho(rho_sv_pec_physical, H)
    add_energy_result(
        rows, "SV → PEC surrogate", e_sv_pec, "Combination",
        "Global-channel inverse applied after particle-number projection",
        ideal_energy,
    )

    
    rho_pec_sv, pec_sv_acceptance = apply_symmetry_verification(
        rho_pec_physical
    )
    e_pec_sv = energy_from_rho(rho_pec_sv, H)
    add_energy_result(
        rows, "PEC surrogate → SV", e_pec_sv, "Combination",
        "Particle-number projection applied after PEC physical projection",
        ideal_energy,
        {"SV acceptance": pec_sv_acceptance},
    )

    
    se_sv_vd = run_subspace_expansion(rho_sv_vd, H, pool)
    e_sv_vd_se = get_energy_from_method_result(se_sv_vd)
    add_energy_result(
        rows, "SV → VD → SE", e_sv_vd_se, "Combination",
        "Sequential particle-number projection, virtual distillation, and SE",
        ideal_energy,
    )

    
    
    
    
    
    
    
    
    
    
    print("\\n=== ZNE-based QEM combinations ===")

    def zne_for_pipeline(label, pipeline, description):
        scale_energies = []
        for scale in ZNE_SCALES:
            folded_circuit = fold_global(circuit, scale)
            rho_at_scale = density_matrix_from_circuit(folded_circuit, noise)

            
            rho_work = rho_at_scale
            for method in pipeline:
                if method == "SV":
                    rho_work, _ = apply_symmetry_verification(rho_work)
                elif method == "VD":
                    rho_work, _ = apply_virtual_distillation(rho_work)
                elif method == "PEC":
                    _, rho_work, _ = apply_pec_surrogate(
                        rho_work, p_effective
                    )
                elif method == "SE":
                    
                    
                    scale_result = run_subspace_expansion(
                        rho_work, H, pool
                    )
                    scale_energy = get_energy_from_method_result(scale_result)
                    rho_work = None
                    break
                else:
                    raise ValueError(f"Unknown pipeline operation: {method}")

            if rho_work is not None:
                scale_energy = energy_from_rho(rho_work, H)

            scale_energies.append(float(scale_energy))
            print(
                f"{label}, scale {scale}: "
                f"{scale_energy:.12f} Ha"
            )

        extrapolated, coefficients = richardson_extrapolate(
            ZNE_SCALES, scale_energies, degree=2
        )
        add_energy_result(
            rows,
            label,
            extrapolated,
            "Combination",
            description,
            ideal_energy,
        )
        zne_combo_results[label] = {
            "scales": list(ZNE_SCALES),
            "scale_energies_hartree": scale_energies,
            "extrapolated_energy_hartree": float(extrapolated),
            "fit_coefficients": np.asarray(coefficients).tolist(),
            "pipeline": list(pipeline),
        }

    zne_combo_results = {}
    zne_for_pipeline(
        "SV + ZNE", ["SV"],
        "At each folded noise scale: particle-number projection, then quadratic zero-noise extrapolation",
    )
    zne_for_pipeline(
        "VD + ZNE", ["VD"],
        "At each folded noise scale: M=2 virtual distillation, then quadratic zero-noise extrapolation",
    )
    zne_for_pipeline(
        "SV + VD + ZNE", ["SV", "VD"],
        "At each folded noise scale: sector projection, virtual distillation, then extrapolation",
    )
    zne_for_pipeline(
        "SV + SE + ZNE", ["SV", "SE"],
        "At each folded noise scale: sector projection, subspace expansion energy, then extrapolation",
    )
    zne_for_pipeline(
        "VD + SE + ZNE", ["VD", "SE"],
        "At each folded noise scale: virtual distillation, subspace expansion energy, then extrapolation",
    )
    zne_for_pipeline(
        "SV + VD + SE + ZNE", ["SV", "VD", "SE"],
        "At each folded noise scale: sector projection, virtual distillation, subspace expansion energy, then extrapolation",
    )

    
    
    
    
    sqd_diagnostics = {}
    try:
        sqd_counts = sampled_counts(circuit, noise, SHOTS, SEED + 20)
        sqd = run_sqd_from_counts(
            sqd_counts,
            data,
            samples_per_batch=512,
            num_batches=4,
            max_iterations=10,
            seed=SEED,
            max_dim=16,
        )
        sqd_diagnostics = {
            k: v for k, v in sqd.items() if k != "result"
        }
        print("\nSQD result:", sqd_diagnostics)
    except Exception as exc:
        sqd_diagnostics = {
            "status": "FAILED",
            "error": repr(exc),
            "note": (
                "The current helper passes a NumPy array where qiskit-addon-sqd "
                "0.12.1 expects a BitArray. No SQD energy is reported."
            ),
        }
        print("\nSQD stage failed; no SQD value is included:", repr(exc))

    
    df = save_report(
        rows,
        ideal_energy,
        ZNE_SCALES,
        zne_energies,
        mem_diagnostics,
        nrep_diagnostics,
        zne_combo_results,
    )

    with open(
        os.path.join(OUTPUT_DIR, "lih_qem_sqd_status.json"), "w"
    ) as f:
        json.dump(sqd_diagnostics, f, indent=2, default=str)

    print("\n=== Additional diagnostics ===")
    print("SV acceptance fraction:", f"{sv_acceptance:.4f}")
    print("VD purity:", f"{vd_purity:.6f}")
    print("PEC effective depolarizing parameter (approx.):", f"{p_effective:.6f}")
    print("MEM scope:", mem_diagnostics["scope"])
    print("N-representability:", nrep_diagnostics.get("status", "completed"))
    print("SQD:", sqd_diagnostics.get("status", "completed"))

    return df

if __name__ == "__main__":
    main()
