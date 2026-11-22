from __future__ import annotations

import warnings
import numpy as np
import scipy.sparse as sp
import cvxpy as cp
from qiskit.quantum_info import Operator, Pauli, SparsePauliOp, Statevector
from qiskit_aer import AerSimulator

# --------------------------------------------------------------------------- #
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
# --------------------------------------------------------------------------- #
def energy_general(h1: np.ndarray, g2: np.ndarray, D1: np.ndarray, D2: np.ndarray, e_nuc: float = 0.0) -> float:
    e_1b = np.sum(h1 * D1)
    e_2b = 0.5 * np.sum(g2.reshape(D2.shape) * D2)
    return float(np.real(e_1b + e_2b + e_nuc))

def spatial_to_spin_orbital_integrals(h1_spatial, eri_spatial):
    h=np.asarray(h1_spatial,dtype=float); eri=np.asarray(eri_spatial,dtype=float)
    n=h.shape[0]; nso=2*n
    hso=np.zeros((nso,nso),dtype=float)
    gso=np.zeros((nso,nso,nso,nso),dtype=float)
    for sigma in range(2):
        for p in range(n):
            for q in range(n):
                hso[sigma*n+p,sigma*n+q]=h[p,q]
    # RDM convention here is D2[p,q,r,s] = <a†_p a†_q a_s a_r>.
    # PySCF chemist integrals multiply a†_(p,sigma) a†_(r,tau) a_(s,tau) a_(q,sigma).
    for sigma in range(2):
        for tau in range(2):
            for p in range(n):
                for q in range(n):
                    for r in range(n):
                        for s in range(n):
                            P=sigma*n+p; Q=sigma*n+q
                            R=tau*n+r; S=tau*n+s
                            gso[P,R,Q,S]=eri[p,q,r,s]
    return hso,gso

class NRepMitigator:
    """LiH-ready RDM SDP wrapper; input RDMs must use this module's JW ordering."""
    def __init__(self, h1_spatial, eri_spatial, ecore=0.0, shots=2048, seed=42):
        self.h1,self.g2=spatial_to_spin_orbital_integrals(h1_spatial,eri_spatial)
        self.ecore=float(ecore); self.n=self.h1.shape[0]
        self.nelec=2; self.shots=int(shots); self.rng=np.random.default_rng(seed)

    def run_from_density_matrix(self, rho, add_sampling_noise=True):
        d1_exact,d2_exact=exact_rdms(np.asarray(rho),self.n)
        if add_sampling_noise:
            d1_meas,d2_meas=sampled_rdms(np.asarray(rho),self.n,self.shots,self.rng)
        else:
            d1_meas,d2_meas=d1_exact,d2_exact
        e_raw=energy_general(self.h1,self.g2,d1_meas,d2_meas,self.ecore)
        d1_sdp,d2_sdp=project_rdms_sdp(d1_meas,d2_meas,self.n,self.nelec,real=True)
        e_sdp=energy_general(self.h1,self.g2,d1_sdp,d2_sdp,self.ecore)
        return {'noisy_value':e_raw,'mitigated_value':e_sdp,
                'D1_measured':d1_meas,'D2_measured':d2_meas,
                'D1_projected':d1_sdp,'D2_projected':d2_sdp,
                'scope':'synthetic Pauli-expectation sampling from simulator density matrix'}
