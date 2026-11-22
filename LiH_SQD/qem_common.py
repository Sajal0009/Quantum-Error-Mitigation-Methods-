"""Shared helpers for the LiH QEM simulation benchmark."""
import numpy as np

def expectation(rho, operator):
    mat = operator.to_matrix() if hasattr(operator, 'to_matrix') else np.asarray(operator)
    return float(np.real(np.trace(mat @ rho)))

def normalize_density(rho, eps=1e-12):
    rho=0.5*(np.asarray(rho,dtype=complex)+np.asarray(rho,dtype=complex).conj().T)
    vals, vecs=np.linalg.eigh(rho)
    vals=np.clip(vals,0,None)
    if vals.sum() < eps: raise ValueError('Density matrix has zero trace after projection.')
    vals/=vals.sum()
    return (vecs*vals)@vecs.conj().T

def sector_projector(n_qubits, n_alpha=1, n_beta=1):
    n=n_qubits//2; dim=1<<n_qubits
    keep=[]
    for state in range(dim):
        na=(state & ((1<<n)-1)).bit_count()
        nb=((state>>n) & ((1<<n)-1)).bit_count()
        keep.append(na==n_alpha and nb==n_beta)
    return np.diag(np.asarray(keep,dtype=float))
