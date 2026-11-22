import numpy as np

from pyscf import gto, scf, mcscf, ao2mo
from qiskit.quantum_info import SparsePauliOp

def build_lih_cas22(bond_length=1.60, basis="sto-3g"):

    mol = gto.M(
        atom=f"Li 0 0 0; H 0 0 {float(bond_length)}",
        basis=basis,
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=0
    )

    
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()

    if not mf.converged:
        raise RuntimeError("LiH RHF calculation did not converge.")

    
    ncas = 2
    nelecas = 2

    mc = mcscf.CASCI(mf, ncas, nelecas)
    mc.kernel()

    if not mc.converged:
        raise RuntimeError("LiH CASCI calculation did not converge.")

    
    e_tot = float(mc.e_tot)

    
    h1eff, ecore = mc.get_h1eff()

    h1eff = np.asarray(h1eff)
    ecore = float(ecore)

    
    
    
    
    

    eri_packed = mc.get_h2eff()

    eri = np.asarray(
        ao2mo.restore(1, eri_packed, ncas)
    )

    if eri.shape != (ncas, ncas, ncas, ncas):
        raise RuntimeError(
            f"Unexpected ERI shape: {eri.shape}; "
            f"expected {(ncas,) * 4}"
        )

    
    
    e_fci_total = e_tot
    e_fci_electronic = e_tot - ecore

    return {
        "mol": mol,
        "mf": mf,
        "mc": mc,
        "h1": h1eff,
        "eri": eri,
        "ecore": ecore,
        "energy_casci": e_tot,
        "energy_fci_electronic": e_fci_electronic,
        "energy_fci_total": e_fci_total,
        "norb": ncas,
        "nelec": (1, 1),
        "bond_length": float(bond_length),
        "basis": basis
    }

def _apply_fermion(state, mode, create):

    occupied = (state >> mode) & 1

    if create and occupied:
        return None, 0

    if not create and not occupied:
        return None, 0

    sign = (
        -1
        if (state & ((1 << mode) - 1)).bit_count() % 2
        else 1
    )

    if create:
        new_state = state | (1 << mode)
    else:
        new_state = state & ~(1 << mode)

    return new_state, sign

def build_jw_matrix(h1, eri, ecore=0.0):

    h1 = np.asarray(h1)
    eri = np.asarray(eri)

    n = h1.shape[0]
    nspin = 2 * n
    dim = 1 << nspin

    if eri.shape != (n, n, n, n):
        raise ValueError(
            f"ERI shape {eri.shape} is incompatible with "
            f"one-electron integrals of size {n}."
        )

    H = np.zeros((dim, dim), dtype=complex)

    H += float(ecore) * np.eye(dim)

    
    
    
    

    for sigma in range(2):

        for p in range(n):
            for q in range(n):

                coeff = h1[p, q]

                if abs(coeff) < 1e-13:
                    continue

                a = sigma * n + p
                c = sigma * n + q

                for ket in range(dim):

                    s1, z1 = _apply_fermion(
                        ket, c, False
                    )

                    if s1 is None:
                        continue

                    s2, z2 = _apply_fermion(
                        s1, a, True
                    )

                    if s2 is not None:
                        H[s2, ket] += coeff * z1 * z2

    
    
    
    
    
    
    
    
    

    for sigma in range(2):
        for tau in range(2):

            for p in range(n):
                for q in range(n):
                    for r in range(n):
                        for s in range(n):

                            coeff = 0.5 * eri[p, q, r, s]

                            if abs(coeff) < 1e-13:
                                continue

                            ap = sigma * n + p
                            aq = sigma * n + q

                            ar = tau * n + r
                            ass = tau * n + s

                            for ket in range(dim):

                                
                                st, z1 = _apply_fermion(
                                    ket, aq, False
                                )

                                if st is None:
                                    continue

                                st, z2 = _apply_fermion(
                                    st, ass, False
                                )

                                if st is None:
                                    continue

                                
                                st, z3 = _apply_fermion(
                                    st, ar, True
                                )

                                if st is None:
                                    continue

                                st, z4 = _apply_fermion(
                                    st, ap, True
                                )

                                if st is not None:
                                    H[st, ket] += (
                                        coeff * z1 * z2 * z3 * z4
                                    )

    
    hermiticity_error = np.linalg.norm(
        H - H.conj().T
    )

    if hermiticity_error > 1e-9:
        raise RuntimeError(
            f"JW Hamiltonian is not Hermitian: "
            f"error={hermiticity_error}"
        )

    return 0.5 * (H + H.conj().T)

def matrix_to_pauli(H, atol=1e-10):

    from itertools import product

    I = np.array(
        [[1, 0], [0, 1]], dtype=complex
    )

    X = np.array(
        [[0, 1], [1, 0]], dtype=complex
    )

    Y = np.array(
        [[0, -1j], [1j, 0]], dtype=complex
    )

    Z = np.array(
        [[1, 0], [0, -1]], dtype=complex
    )

    matrices = {
        "I": I,
        "X": X,
        "Y": Y,
        "Z": Z
    }

    n = int(round(np.log2(H.shape[0])))

    terms = []

    for chars in product("IXYZ", repeat=n):

        label = "".join(chars)

        P = matrices[label[0]]

        for ch in label[1:]:
            P = np.kron(P, matrices[ch])

        coeff = np.trace(
            P.conj().T @ H
        ) / (2 ** n)

        if abs(coeff) > atol:
            terms.append((label, complex(coeff)))

    if not terms:
        raise RuntimeError("Pauli decomposition returned no terms.")

    return SparsePauliOp.from_list(terms).simplify(
        atol=atol
    )

def make_lih_hamiltonian(data):

    H = build_jw_matrix(
        data["h1"],
        data["eri"],
        data["ecore"]
    )

    H_pauli = matrix_to_pauli(H)

    return H, H_pauli

def run_sqd_from_counts(
    counts,
    data,
    samples_per_batch=512,
    num_batches=4,
    max_iterations=10,
    seed=7,
    max_dim=16
):

    from qiskit_addon_sqd.fermion import (
        diagonalize_fermionic_hamiltonian
    )

    rows = []

    for bitstr, count in counts.items():

        bits = bitstr.replace(" ", "")

        if len(bits) != 4:
            raise ValueError(
                f"Expected 4-bit measurement outcome, got {bits}"
            )

        row = np.array(
            [c == "1" for c in bits],
            dtype=bool
        )

        rows.extend([row] * int(count))

    if len(rows) == 0:
        raise ValueError("No measurement samples available.")

    bit_array = np.asarray(rows, dtype=bool)

    result = diagonalize_fermionic_hamiltonian(
        data["h1"],
        data["eri"],
        bit_array,
        samples_per_batch=min(
            int(samples_per_batch),
            len(bit_array)
        ),
        norb=data["norb"],
        nelec=data["nelec"],
        num_batches=int(num_batches),
        max_iterations=int(max_iterations),
        max_dim=max_dim,
        seed=seed
    )

    energy = getattr(result, "energy", None)

    if energy is None and isinstance(result, (tuple, list)):
        energy = result[0]

    if energy is None:
        raise RuntimeError(
            "Could not extract energy from SQD result. "
            f"Result type: {type(result)}"
        )

    return {
        "energy_electronic": float(energy),
        "energy_total": float(energy + data["ecore"]),
        "result": result,
        "num_samples": len(bit_array)
    }
