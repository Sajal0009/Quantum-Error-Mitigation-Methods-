'Probabilistic error cancellation with learned Pauli channels.'
from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliGate
from qiskit.quantum_info import (Clifford, Operator, Pauli, SparsePauliOp,
                                 Statevector, SuperOp)
from qiskit_aer import AerSimulator

_AXIS = {"rx": "X", "ry": "Y", "rz": "Z"}      # rotation gates and their generator
_XZ = {"I": (0, 0), "X": (1, 0), "Y": (1, 1), "Z": (0, 1)}
_CHAR = {v: k for k, v in _XZ.items()}
_TOL = 1e-9                                      # gamma - 1 below this => gate treated as noiseless

# --------------------------------------------------------------------------- #
# Pauli-label helpers
# --------------------------------------------------------------------------- #
def _labels(k: int) -> list[str]:
    return ["".join(p) for p in itertools.product("IXYZ", repeat=k)]

def _commutes(a: str, b: str) -> bool:
    return sum(x != "I" and y != "I" and x != y for x, y in zip(a, b)) % 2 == 0

def _mul(a: str, b: str) -> str:
    """Pauli product, global phase dropped."""
    return "".join(_CHAR[(_XZ[x][0] ^ _XZ[y][0], _XZ[x][1] ^ _XZ[y][1])] for x, y in zip(a, b))

def _walsh(labels: list[str]) -> np.ndarray:
    return np.array([[1.0 if _commutes(a, b) else -1.0 for b in labels] for a in labels])

# --------------------------------------------------------------------------- #
# Pauli-channel estimation
# --------------------------------------------------------------------------- #
@dataclass
class PauliChannel:
    labels: list[str]
    probs: np.ndarray        # p_P: Pauli error probabilities of the twirled channel
    lam: np.ndarray          # Pauli eigenvalues (diagonal of the Pauli transfer matrix)
    quasi: np.ndarray        # q_P: quasi-probabilities of the inverse channel (sum = 1)
    gamma: float             # sum |q_P|, per-gate sampling overhead
    twirl_residual: float    # largest off-diagonal PTM entry: the non-Pauli part a twirl discards

    @property
    def error_rate(self) -> float:
        return float(1.0 - self.probs[0])

def learn_pauli_channel(noise_model, op) -> PauliChannel:
    """Error channel Lambda of `op` under `noise_model` (noise is applied after the gate)."""
    k = op.num_qubits
    d = 2 ** k
    labels = _labels(k)

    qc = QuantumCircuit(k)
    qc.append(op, range(k))
    inv_ideal = SuperOp(Operator(qc).adjoint()).data
    qc.save_superop()
    res = AerSimulator(method="superop", noise_model=noise_model).run(qc).result()
    S_err = np.asarray(res.data(0)["superop"].data) @ inv_ideal      # (Lambda o G) o G^-1

    # Pauli transfer matrix R_ij = tr(P_i Lambda(P_j)) / d ; Qiskit SuperOps act on column-stacked vec(rho)
    mats = [Pauli(lbl).to_matrix() for lbl in labels]
    R = np.empty((len(labels), len(labels)))
    for j, Pj in enumerate(mats):
        out = (S_err @ Pj.reshape(-1, order="F")).reshape(d, d, order="F")
        for i, Pi in enumerate(mats):
            R[i, j] = np.trace(Pi @ out).real / d

    lam = np.diag(R).copy()
    if np.any(lam <= 1e-6):
        raise ValueError(f"error channel of '{op.name}' is not invertible (Pauli eigenvalue <= 0)")
    H = _walsh(labels)
    probs = H @ lam / len(labels)
    quasi = H @ (1.0 / lam) / len(labels)
    return PauliChannel(labels, probs, lam, quasi, float(np.abs(quasi).sum()),
                        float(np.max(np.abs(R - np.diag(lam)))))

# --------------------------------------------------------------------------- #
# Gate-twirl bookkeeping
# --------------------------------------------------------------------------- #
class _GateProfile:
    """Everything needed to twirl + correct one noisy gate type."""

    def __init__(self, op, channel: PauliChannel, twirl: bool):
        self.channel = channel
        k = op.num_qubits
        ident = "I" * k
        if twirl:
            self.pre = list(channel.labels)
            if op.name in _AXIS:                                     # exact: P rot(+-theta) P = rot(theta)
                self.post = list(self.pre)
                self.flip = [not _commutes(p, _AXIS[op.name]) for p in self.pre]
            else:                                                    # Clifford: propagate P through the gate
                try:
                    cliff = Clifford(op)
                except Exception as exc:
                    raise NotImplementedError(
                        f"cannot twirl non-Clifford, non-rotation gate '{op.name}'") from exc
                self.post = ["".join(c for c in Pauli(p).evolve(cliff, frame="s").to_label() if c in "IXYZ")
                             for p in self.pre]
                self.flip = [False] * len(self.pre)
        else:
            self.pre, self.post, self.flip = [ident], [ident], [False]
        self.n_twirl = len(self.pre)
        absq = np.abs(channel.quasi)
        self.sample_p = absq / absq.sum()
        self.qsign = np.sign(channel.quasi)
        # final Pauli applied after the gate = post-twirl Pauli * sampled correction Pauli
        self.final = [[_mul(self.post[t], q) for q in channel.labels] for t in range(self.n_twirl)]

# --------------------------------------------------------------------------- #
# PEC implementation
# --------------------------------------------------------------------------- #
class PECMitigator:
    'Parameters ---------- num_samples number of sampled circuits (Monte-Carlo samples of the quasi-probability distribution) shots_per_sample shots per sampled circuit (total shots = num_samples * shots_per_sample) noisy_shots shots for the unmitigated reference run twirl Pauli-twirl every noisy gate so the simulated noise is exactly the learned Pauli channel readout_flip_prob symmetric readout flip probability to correct analytically (0 = none) method Aer simulation method'

    def __init__(self, num_samples=2000, shots_per_sample=4, noisy_shots=8192, twirl=True,
                 readout_flip_prob=0.0, method="automatic", seed=None, chunk_size=500, verbose=True):
        if not 0.0 <= readout_flip_prob < 0.5:
            raise ValueError("readout_flip_prob must be in [0, 0.5)")
        self.num_samples = num_samples
        self.shots_per_sample = shots_per_sample
        self.noisy_shots = noisy_shots
        self.twirl = twirl
        self.readout_flip_prob = readout_flip_prob
        self.method = method
        self.seed = seed
        self._rng = np.random.default_rng(seed)     # advances across run() calls: reproducible, but no reused streams
        self.chunk_size = chunk_size
        self.verbose = verbose
        self.channels: dict[str, PauliChannel] = {}
        self._cache: dict = {}
        self._pauli_gates: dict[str, PauliGate] = {}

    # ---- observable / counts ------------------------------------------------ #
    @staticmethod
    def _z_terms(observable, n):
        if isinstance(observable, Pauli):
            observable = SparsePauliOp(observable)
        terms = []
        for lbl, c in zip(observable.paulis.to_labels(), observable.coeffs):
            lbl = "".join(ch for ch in lbl if ch in "IXYZ")
            if len(lbl) != n or set(lbl) - {"I", "Z"}:
                raise ValueError("observable must be a sum of I/Z strings on all circuit qubits")
            if abs(np.imag(c)) > 1e-12:
                raise ValueError("observable coefficients must be real")
            mask = sum(1 << q for q, ch in enumerate(reversed(lbl)) if ch == "Z")
            terms.append((float(np.real(c)), mask))
        return terms

    def _expect(self, counts, terms, correct_readout):
        total = sum(counts.values())
        val = 0.0
        for coef, mask in terms:
            s = sum(cnt * (1 - 2 * (bin(int(key.replace(" ", ""), 2) & mask).count("1") & 1))
                    for key, cnt in counts.items())
            f = (1 - 2 * self.readout_flip_prob) ** bin(mask).count("1") if correct_readout else 1.0
            val += coef * s / total / f
        return val

    # ---- steps 1-2: learn channels, build per-instruction plan --------------- #
    def _learn(self, circuit, noise_model):
        profiles = {}
        for inst in circuit.data:
            op = inst.operation
            if op.name in ("measure", "reset"):
                raise ValueError("PEC input circuit must be unitary (no measure/reset)")
            if op.name == "barrier":
                continue
            key = (id(noise_model), op.name, op.num_qubits, self.twirl)
            if key not in self._cache:
                chan = learn_pauli_channel(noise_model, op)
                prof = _GateProfile(op, chan, self.twirl) if chan.gamma - 1.0 > _TOL else None
                self._cache[key] = (chan, prof)
            chan, prof = self._cache[key]
            profiles[(op.name, op.num_qubits)] = prof
            self.channels[op.name] = chan
        return profiles

    def _make_plan(self, circuit, profiles):
        plan, noisy = [], []
        for inst in circuit.data:
            op = inst.operation
            qubits = [circuit.find_bit(q).index for q in inst.qubits]
            prof = None if op.name == "barrier" else profiles[(op.name, op.num_qubits)]
            flipped = None
            if prof is not None and any(prof.flip):
                flipped = copy.copy(op)
                flipped.params = [-p for p in op.params]
            rec = (op, qubits, prof, flipped)
            plan.append(rec)
            if prof is not None:
                noisy.append(rec)
        return plan, noisy

    # ---- step 3: assemble one sampled circuit -------------------------------- #
    def _pgate(self, label):
        if label not in self._pauli_gates:
            self._pauli_gates[label] = PauliGate(label)
        return self._pauli_gates[label]

    def _assemble(self, n, plan, T, J, s):
        qc = QuantumCircuit(n)
        r = 0
        for op, qubits, prof, flipped in plan:
            if prof is None:
                qc.append(op, qubits)
                continue
            t, j = T[r, s], J[r, s]
            r += 1
            pre, post = prof.pre[t], prof.final[t][j]
            if set(pre) != {"I"}:
                qc.append(self._pgate(pre), qubits)
            qc.append(flipped if prof.flip[t] else op, qubits)
            if set(post) != {"I"}:
                qc.append(self._pgate(post), qubits)
        qc.measure_all()
        return qc

    # ---- main entry point ------------------------------------------------------ #
    def run(self, circuit: QuantumCircuit, observable, noise_model) -> dict:
        n = circuit.num_qubits
        terms = self._z_terms(observable, n)
        rng = self._rng
        if self.seed is not None:
            base = int(rng.integers(0, 2 ** 30))
            seedkw = lambda off=0: {"seed_simulator": base + off}
        else:
            seedkw = lambda off=0: {}

        ideal = float(Statevector(circuit).expectation_value(observable).real)

        backend = AerSimulator(method=self.method, noise_model=noise_model)
        counts = backend.run(circuit.measure_all(inplace=False), shots=self.noisy_shots,
                             **seedkw()).result().get_counts()
        noisy = self._expect(counts, terms, False)
        noisy_ro = self._expect(counts, terms, True)

        profiles = self._learn(circuit, noise_model)
        plan, noisy_recs = self._make_plan(circuit, profiles)
        gamma_total = float(np.prod([rec[2].channel.gamma for rec in noisy_recs])) if noisy_recs else 1.0
        if self.verbose:
            self.print_channels()
            print(f"noisy gate instances: {len(noisy_recs)} | gamma_total = {gamma_total:.4f} | "
                  f"sampling overhead gamma^2 = {gamma_total ** 2:.3f}")

        N = self.num_samples
        T = np.zeros((len(noisy_recs), N), dtype=int)
        J = np.zeros((len(noisy_recs), N), dtype=int)
        sign = np.ones(N)
        for r, rec in enumerate(noisy_recs):
            prof = rec[2]
            if self.twirl:
                T[r] = rng.integers(0, prof.n_twirl, N)
            J[r] = rng.choice(len(prof.sample_p), size=N, p=prof.sample_p)
            sign *= prof.qsign[J[r]]

        vals = np.empty(N)
        for start in range(0, N, self.chunk_size):
            stop = min(N, start + self.chunk_size)
            circs = [self._assemble(n, plan, T, J, s) for s in range(start, stop)]
            res = backend.run(circs, shots=self.shots_per_sample, **seedkw(start + 1)).result()
            for i in range(stop - start):
                vals[start + i] = self._expect(res.get_counts(i), terms, True)

        est = gamma_total * sign * vals
        return {
            "ideal_value": ideal,
            "noisy_value": noisy,
            "noisy_value_readout_corrected": noisy_ro,
            "mitigated_value": float(est.mean()),
            "std_error": float(est.std(ddof=1) / np.sqrt(N)),
            "gamma_total": gamma_total,
            "sampling_overhead": gamma_total ** 2,
            "num_noisy_gates": len(noisy_recs),
            "num_samples": N,
            "total_shots": N * self.shots_per_sample,
            "twirl": self.twirl,
        }

    def print_channels(self):
        print(f"{'gate':<6}{'gamma':>10}{'error rate':>12}{'non-Pauli residual':>20}")
        for name, ch in self.channels.items():
            print(f"{name:<6}{ch.gamma:>10.5f}{ch.error_rate:>12.5f}{ch.twirl_residual:>20.2e}")
