import numpy as np

def mitigate_probabilities(probabilities, confusion_matrices, clip=True):
    p=np.asarray(probabilities,dtype=float).reshape(-1)
    n=len(confusion_matrices)
    if p.size != 2**n: raise ValueError('Probability vector dimension mismatch.')
    A=np.array([[1.0]])
    for M in confusion_matrices: A=np.kron(M,A)
    corrected=np.linalg.pinv(A)@p
    if clip:
        corrected=np.clip(corrected,0,None)
        if corrected.sum(): corrected/=corrected.sum()
    return corrected

def calibrate_readout(backend, n_qubits, shots=2048):
    from qiskit import QuantumCircuit, transpile
    matrices=[]
    for q in range(n_qubits):
        M=np.zeros((2,2),float)
        for truth in (0,1):
            qc=QuantumCircuit(n_qubits,1)
            if truth: qc.x(q)
            qc.measure(q,0)
            counts=backend.run(transpile(qc,backend),shots=shots).result().get_counts()
            for key,count in counts.items(): M[int(key.replace(' ','')[-1]),truth]=count/shots
        matrices.append(M)
    return matrices
