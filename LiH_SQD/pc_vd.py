"""Exact density-matrix M=2 virtual distillation (simulator-only)."""
import numpy as np

def virtual_distillation(rho, observable):
    O=observable.to_matrix() if hasattr(observable,'to_matrix') else np.asarray(observable)
    rho=np.asarray(rho,dtype=complex)
    rho2=rho@rho
    purity=float(np.real(np.trace(rho2)))
    if purity<=1e-14: raise ValueError('Density-matrix purity is numerically zero.')
    return {'noisy_value':float(np.real(np.trace(O@rho))),
            'mitigated_value':float(np.real(np.trace(O@rho2))/purity),
            'purity':purity,
            'scope':'exact density-matrix simulation; not a hardware measurement protocol'}
