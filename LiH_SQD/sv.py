"""Particle-number sector verification for the LiH CAS(2,2) sample space."""
import numpy as np
from qem_common import sector_projector, normalize_density

def symmetry_verify(rho, observable, n_alpha=1, n_beta=1):
    P=sector_projector(int(round(np.log2(rho.shape[0]))),n_alpha,n_beta)
    projected=P@rho@P
    weight=float(np.real(np.trace(projected)))
    if weight<1e-12: raise RuntimeError('No probability mass in requested (Nalpha,Nbeta) sector.')
    corrected=normalize_density(projected/weight)
    O=observable.to_matrix() if hasattr(observable,'to_matrix') else np.asarray(observable)
    return {'mitigated_value':float(np.real(np.trace(O@corrected))),
            'accepted_probability':weight,'rho_verified':corrected}
