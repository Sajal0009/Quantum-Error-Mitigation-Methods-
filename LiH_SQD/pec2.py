import numpy as np

def invert_global_depolarizing(rho_noisy, p):
    rho=np.asarray(rho_noisy,dtype=complex); d=rho.shape[0]
    p=float(p)
    if not 0<=p<1: raise ValueError('p must be in [0,1).')
    if p>=1-1e-12: raise ValueError('Global depolarizing channel is not invertible at p=1.')
    rho_est=(rho-p*np.eye(d)/d)/(1-p)
    # Physical projection is reported separately because it introduces bias.
    vals,vecs=np.linalg.eigh((rho_est+rho_est.conj().T)/2)
    vals=np.clip(vals,0,None)
    if vals.sum()==0: raise RuntimeError('Inverted density matrix has no positive support.')
    rho_phys=(vecs*(vals/vals.sum()))@vecs.conj().T
    return {'rho_linear_inverse':rho_est,'rho_physical_projected':rho_phys,
            'gamma_proxy':1/(1-p),'scope':'global depolarizing inverse; not gate-local PEC'}
