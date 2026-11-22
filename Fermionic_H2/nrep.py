'N-representability error mitigation.'
from __future__ import annotations

import warnings
import numpy as np
import scipy.sparse as sp
import cvxpy as cp
from qiskit.quantum_info import Operator, Pauli, SparsePauliOp, Statevector
from qiskit_aer import AerSimulator

# --------------------------------------------------------------------------- #
# Jordan-Wigner mapping
# --------------------------------------------------------------------------- #
_I2 = sp.identity(2, format="csr", dtype=complex)
_Z = sp.csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex))
_LOWER = sp.csr_matrix(np.array([[0, 1], [0, 0]], dtype=complex))  # |0><1|

def _embed(ops: dict[int, sp.spmatrix], n: int) -> sp.csr_matrix:
    out = sp.identity(1, format="csr", dtype=complex)
    for q in range(n - 1, -1, -1):
        out = sp.kron(out, ops.get(q, _I2), format="csr")
    return out

def jw_annihilation(n: int) -> list[sp.csr_matrix]:
    return [_embed({**{k: _Z for k in range(p)}, p: _LOWER}, n) for p in range(n)]

def _expect(rho: np.ndarray, O: sp.spmatrix) -> complex:
    return complex(O.multiply(rho.T).sum())

# --------------------------------------------------------------------------- #
# RDM construction
# --------------------------------------------------------------------------- #
def rdm_operators(n: int):
    a = jw_annihilation(n)
    ad = [x.getH().tocsr() for x in a]
    O1 = {(i, j): ad[i] @ a[j] for i in range(n) for j in range(n)}
    O2 = {
        (p, q, r, s): ad[p] @ ad[q] @ a[s] @ a[r]
        for p in range(n) for q in range(n) for r in range(n) for s in range(n)
    }
    return O1, O2

def exact_rdms(rho: np.ndarray, n: int):
    O1, O2 = rdm_operators(n)
    D1 = np.zeros((n, n), dtype=complex)
    T = np.zeros((n, n, n, n), dtype=complex)
    for (i, j), O in O1.items():
        D1[i, j] = _expect(rho, O)
    for (p, q, r, s), O in O2.items():
        T[p, q, r, s] = _expect(rho, O)
    return D1, T.reshape(n * n, n * n)

def sampled_rdms(rho: np.ndarray, n: int, shots: int, rng: np.random.Generator):
    O1, O2 = rdm_operators(n)
    keys = [("1",) + k for k in O1] + [("2",) + k for k in O2]
    ops = list(O1.values()) + list(O2.values())
    decomp, labels = [], {}

    for O in ops:
        spo = SparsePauliOp.from_operator(Operator(O.toarray()))
        terms = [(str(p), c) for p, c in zip(spo.paulis, spo.coeffs)]
        decomp.append(terms)
        for lab, _ in terms:
            labels.setdefault(lab, None)

    for lab in labels:
        if set(lab) == {"I"}:
            labels[lab] = 1.0
            continue
        e = _expect(rho, sp.csr_matrix(Pauli(lab).to_matrix(sparse=True))).real
        sigma = np.sqrt(max(1.0 - e * e, 0.0) / shots)
        labels[lab] = float(np.clip(e + rng.normal(0.0, sigma), -1.0, 1.0))

    D1 = np.zeros((n, n), dtype=complex)
    T = np.zeros((n, n, n, n), dtype=complex)
    for key, terms in zip(keys, decomp):
        val = sum(c * labels[lab] for lab, c in terms)
        if key[0] == "1":
            D1[key[1], key[2]] = val
        else:
            T[key[1], key[2], key[3], key[4]] = val

    return D1, T.reshape(n * n, n * n)

# --------------------------------------------------------------------------- #
# Permutation and positivity operators
# --------------------------------------------------------------------------- #
def _swap(n: int) -> sp.csr_matrix:
    m = n * n
    S = sp.lil_matrix((m, m))
    for p in range(n):
        for q in range(n):
            S[q * n + p, p * n + q] = 1.0
    return S.tocsr()

def _realign_perm(n: int) -> sp.csr_matrix:
    m = n * n
    rows, cols = [], []
    for p in range(n):
        for q in range(n):
            for r in range(n):
                for s in range(n):
                    rows.append((p * n + q) * m + (r * n + s))
                    cols.append((p * n + s) * m + (r * n + q))
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(m * m, m * m))

def _lib_ops(M):
    return (np.kron, "np") if isinstance(M, np.ndarray) else (cp.kron, "cp")

def g_matrix(D1, M, n: int):
    kron, tag = _lib_ops(M)
    m = n * n
    P = _realign_perm(n)
    if tag == "np":
        real_part = (P @ M.reshape(-1)).reshape(m, m)
    else:
        vec_M = cp.reshape(M, (m * m, 1), order="C")
        real_part = cp.reshape(P @ vec_M, (m, m), order="C")
    return kron(D1, np.eye(n)) - real_part

def q_matrix(D1, M, n: int):
    kron, _ = _lib_ops(M)
    m = n * n
    I_n, I_m = np.eye(n), np.eye(m)
    S = _swap(n)
    D1T = D1.T
    A = kron(D1T, I_n)
    B = kron(I_n, D1T)
    return I_m - S + (-A + B @ S + A @ S - B) + (S @ M @ S).T

# --------------------------------------------------------------------------- #
# SDP projection
# --------------------------------------------------------------------------- #
def project_rdms_sdp(
    D1_meas: np.ndarray,
    D2_meas: np.ndarray,
    n: int,
    nelec: int,
    w1: float = 1.0,
    real: bool = True,
    solver: str = "CLARABEL",
):
    assert 2 <= nelec <= n, "SDP contraction constraint requires 2 <= nelec <= n"
    m = n * n

    # Strict real float64 conversion
    D1m = np.ascontiguousarray(np.real(D1_meas), dtype=np.float64)
    D2m = np.ascontiguousarray(np.real(D2_meas), dtype=np.float64)

    D1 = cp.Variable((n, n), symmetric=True)
    D2 = cp.Variable((m, m), symmetric=True)

    S = _swap(n)
    Q = q_matrix(D1, D2, n)
    G = g_matrix(D1, D2, n)

    Q_sym = (Q + Q.T) / 2
    G_sym = (G + G.T) / 2

    # Linear constraints: strictly avoid cp.real() wrappers on real variables
    cons = [
        D2 == -S @ D2,
        D2 == -D2 @ S,
        cp.trace(D1) == float(nelec),
        cp.trace(D2) == float(nelec * (nelec - 1)),
        D1 >> 0,
        np.eye(n) - D1 >> 0,
        D2 >> 0,
        Q_sym >> 0,
        G_sym >> 0,
    ]

    # Partial trace contraction via explicit linear indexing
    for i in range(n):
        for j in range(n):
            trace_sum = cp.sum([D2[i * n + k, j * n + k] for k in range(n)])
            cons.append(trace_sum == (nelec - 1) * D1[i, j])

    # Affine Frobenius objective
    obj = w1 * cp.sum_squares(D1 - D1m) + cp.sum_squares(D2 - D2m)
    prob = cp.Problem(cp.Minimize(obj), cons)

    # Multi-solver execution fallback
    installed = cp.installed_solvers()
    candidates = [s for s in [solver, "CLARABEL", "SCS"] if s in installed]
    solved = False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for s in candidates:
            try:
                prob.solve(solver=s, verbose=False)
                if prob.status in ("optimal", "optimal_inaccurate"):
                    solved = True
                    break
            except Exception:
                continue

    if not solved:
        raise RuntimeError(f"SDP solver failed. Last status: {prob.status}")

    return np.asarray(D1.value, dtype=complex), np.asarray(D2.value, dtype=complex)

# --------------------------------------------------------------------------- #
# Energy evaluation and molecular reference
# --------------------------------------------------------------------------- #
def energy_general(h1: np.ndarray, g2: np.ndarray, D1: np.ndarray, D2: np.ndarray, e_nuc: float = 0.0) -> float:
    e_1b = np.sum(h1 * D1)
    e_2b = 0.5 * np.sum(g2.reshape(D2.shape) * D2)
    return float(np.real(e_1b + e_2b + e_nuc))

def get_h2_sto3g_hamiltonian() -> tuple[np.ndarray, np.ndarray, float]:
    """Electronic integrals for H2 (R = 0.735 A, STO-3G) on 4 spin-orbitals."""
    n = 4
    e_nuc = 0.7199689076
    h1 = np.zeros((n, n), dtype=float)
    h_spatial = np.array([[-1.252477, 0.0], [0.0, -0.475934]])
    for p in range(n):
        for q in range(n):
            if (p % 2) == (q % 2):
                h1[p, q] = h_spatial[p // 2, q // 2]

    g2 = np.zeros((n, n, n, n), dtype=float)
    for p in range(n):
        for q in range(n):
            for r in range(n):
                for s in range(n):
                    if (p % 2 == r % 2) and (q % 2 == s % 2):
                        sp_p, sp_q, sp_r, sp_s = p // 2, q // 2, r // 2, s // 2
                        val = 0.0
                        if sp_p == 0 and sp_r == 0 and sp_q == 0 and sp_s == 0:
                            val = 0.674493
                        elif sp_p == 1 and sp_r == 1 and sp_q == 1 and sp_s == 1:
                            val = 0.697397
                        elif (sp_p == 0 and sp_r == 0 and sp_q == 1 and sp_s == 1) or \
                             (sp_p == 1 and sp_r == 1 and sp_q == 0 and sp_s == 0):
                            val = 0.663472
                        elif (sp_p == 0 and sp_r == 1 and sp_q == 1 and sp_s == 0) or \
                             (sp_p == 1 and sp_r == 0 and sp_q == 0 and sp_s == 1):
                            val = 0.181287
                        g2[p, q, r, s] = val

    return h1, g2, e_nuc

# --------------------------------------------------------------------------- #
# Mitigator interface
# --------------------------------------------------------------------------- #
class NRepMitigator:
    def __init__(self, num_qubits: int = 4, nelec: int = 2, shots: int = 30_000, seed: int = 42):
        self.num_qubits = num_qubits
        self.nelec = nelec
        self.shots = shots
        self.rng = np.random.default_rng(seed)
        self.h1, self.g2, self.e_nuc = get_h2_sto3g_hamiltonian()

    def run(self, circuit, noise_model=None):
        # Ideal ground truth calculation
        psi_ideal = np.array(Statevector(circuit).data)
        rho_ideal = np.outer(psi_ideal, psi_ideal.conj())
        d1_ideal, d2_ideal = exact_rdms(rho_ideal, self.num_qubits)
        ideal_energy = energy_general(self.h1, self.g2, d1_ideal, d2_ideal, self.e_nuc)

        # Noisy density matrix execution via Aer
        qc_sim = circuit.copy()
        qc_sim.save_density_matrix()
        sim = AerSimulator(method="density_matrix", noise_model=noise_model)
        job = sim.run(qc_sim)
        rho_noisy = np.array(job.result().data()["density_matrix"])

        # Correlated Pauli tomography shot noise
        d1_meas, d2_meas = sampled_rdms(rho_noisy, self.num_qubits, self.shots, self.rng)
        noisy_energy = energy_general(self.h1, self.g2, d1_meas, d2_meas, self.e_nuc)

        # SDP N-representability projection
        d1_sdp, d2_sdp = project_rdms_sdp(d1_meas, d2_meas, self.num_qubits, self.nelec, real=True)
        mitigated_energy = energy_general(self.h1, self.g2, d1_sdp, d2_sdp, self.e_nuc)

        return {
            "ideal_value": ideal_energy,
            "noisy_value": noisy_energy,
            "mitigated_value": mitigated_energy,
        }
