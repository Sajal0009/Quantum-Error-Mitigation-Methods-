import numpy as np
from scipy.linalg import eigh

def number_conserving_pool(n_qubits=4):
    from qiskit.quantum_info import SparsePauliOp
    if n_qubits!=4: raise ValueError('This LiH CAS(2,2) pool is defined for 4 qubits.')
    # Spin-preserving hopping generators are represented by XX+YY combinations.
    labels=[]
    for q0,q1 in ((0,1),(2,3)):
        for p0,p1 in (('X','X'),('Y','Y')):
            chars=['I']*n_qubits
            chars[n_qubits-1-q0]=p0; chars[n_qubits-1-q1]=p1
            labels.append(SparsePauliOp.from_list([(''.join(chars),1.0)]))
    return labels

def subspace_expansion(rho, H, operator_pool=None, cutoff=1e-8):
    rho=np.asarray(rho,dtype=complex); H=np.asarray(H,dtype=complex)
    d=rho.shape[0]
    if operator_pool is None: operator_pool=[np.eye(d)]
    mats=[np.eye(d,dtype=complex)] + [p.to_matrix() if hasattr(p,'to_matrix') else np.asarray(p) for p in operator_pool]
    S=np.empty((len(mats),len(mats)),complex); K=np.empty_like(S)
    for i,A in enumerate(mats):
        for j,B in enumerate(mats):
            S[i,j]=np.trace(rho@A.conj().T@B)
            K[i,j]=np.trace(rho@A.conj().T@H@B)
    S=(S+S.conj().T)/2; K=(K+K.conj().T)/2
    vals,vecs=np.linalg.eigh(S); keep=vals>cutoff
    if not np.any(keep): raise RuntimeError('SE overlap matrix has no supported eigenvectors.')
    W=vecs[:,keep]/np.sqrt(vals[keep])[None,:]
    e=float(np.min(np.linalg.eigvalsh(W.conj().T@K@W)))
    return {'mitigated_value':e,'overlap_rank':int(keep.sum()),'scope':'density-matrix subspace expansion'}
