from __future__ import annotations
import json, os
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector

from benchmark.lih_sqd import build_lih_cas22, make_lih_hamiltonian, run_sqd_from_counts
from benchmark.models import build_noise_model
from qem_common import expectation, normalize_density
from pc_vd import virtual_distillation
from se import subspace_expansion, number_conserving_pool
from sv import symmetry_verify
from pec2 import invert_global_depolarizing
from nrep import NRepMitigator
from zne import fold_global, richardson_extrapolate

SHOTS=2048
SEED=17

def make_ansatz(n_qubits=4):

    qc=QuantumCircuit(n_qubits,name='LiH_CAS22_trial_ansatz')
    
    qc.x(0); qc.x(2)
    
    for q,theta in enumerate((0.24,0.41,0.24,0.41)):
        qc.ry(theta,q)
    qc.cx(0,1); qc.cx(2,3); qc.cx(1,2)
    qc.ry(0.17,0); qc.ry(-0.13,1); qc.ry(0.17,2); qc.ry(-0.13,3)
    return qc

def noisy_density_matrix(circuit, noise_model=None):
    backend=AerSimulator(method='density_matrix',noise_model=noise_model)
    qc=circuit.copy(); qc.save_density_matrix()
    tqc=transpile(qc,backend,optimization_level=0)
    result=backend.run(tqc,shots=1,seed_simulator=SEED).result()
    return np.asarray(result.data(0)['density_matrix'],dtype=complex)

def sampled_counts(circuit, noise_model, shots=SHOTS):
    backend=AerSimulator(noise_model=noise_model)
    qc=circuit.copy(); qc.measure_all()
    tqc=transpile(qc,backend,optimization_level=0)
    return backend.run(tqc,shots=shots,seed_simulator=SEED).result().get_counts()

def main():
    os.makedirs('results',exist_ok=True)
    print('\n=== LiH CAS(2,2): STO-3G / Jordan-Wigner / 4 qubits ===')
    data=build_lih_cas22(bond_length=1.60,basis='sto-3g')
    H,H_pauli=make_lih_hamiltonian(data)
    print(f"RHF converged; active orbitals={data['norb']}; active electrons={sum(data['nelec'])}")
    print(f"CASCI/FCI total-energy reference: {data['energy_fci_total']:.10f} Ha")
    
    sector=[]
    for state in range(H.shape[0]):
        if (state & 0b0011).bit_count()==1 and ((state>>2)&0b0011).bit_count()==1:
            sector.append(state)
    mapped_sector_energy=float(np.linalg.eigvalsh(H[np.ix_(sector,sector)]).min())
    mapping_delta=abs(mapped_sector_energy-data['energy_fci_total'])
    print(f"Mapped JW sector ground energy: {mapped_sector_energy:.10f} Ha")
    print(f"JW vs PySCF FCI absolute difference: {mapping_delta:.3e} Ha")
    if mapping_delta>1e-6:
        raise RuntimeError('Jordan-Wigner Hamiltonian does not reproduce the PySCF CAS(2,2) FCI reference. Stop before benchmarking QEM.')
    print(f"Pauli terms in mapped Hamiltonian: {len(H_pauli)}")

    circuit=make_ansatz(4)
    ideal_rho=noisy_density_matrix(circuit,None)
    ideal_ansatz_energy=expectation(ideal_rho,H_pauli)
    noise=build_noise_model(p1=0.001,p2=0.01)
    rho=noisy_density_matrix(circuit,noise)
    noisy_energy=expectation(rho,H_pauli)
    print(f"Ansatz ideal expectation: {ideal_ansatz_energy:.10f} Ha")
    print(f"Noisy expectation:        {noisy_energy:.10f} Ha")
    print(f"CASCI/FCI reference:      {data['energy_fci_total']:.10f} Ha")

    results={
      'reference_fci_total_hartree':data['energy_fci_total'],
      'mapped_sector_ground_energy_hartree':mapped_sector_energy,
      'jw_fci_absolute_difference_hartree':mapping_delta,
      'ansatz_ideal_hartree':ideal_ansatz_energy,
      'noisy_hartree':noisy_energy,
      'shots':SHOTS,
      'methods':{},
    }

    
    sv=symmetry_verify(rho,H_pauli,1,1)
    results['methods']['symmetry_verification']={k:v for k,v in sv.items() if k!='rho_verified'}

    
    vd=virtual_distillation(rho,H_pauli)
    results['methods']['virtual_distillation']=vd

    
    se=subspace_expansion(rho,H,number_conserving_pool(4))
    results['methods']['subspace_expansion']=se

    
    zne_scales=[1,3,5]; zne_values=[]
    for scale in zne_scales:
        folded=fold_global(circuit,scale)
        rscale=noisy_density_matrix(folded,noise)
        zne_values.append(expectation(rscale,H_pauli))
    zne_zero, zne_coeff=richardson_extrapolate(zne_scales,zne_values,degree=2)
    results['methods']['zne']={'scales':zne_scales,'scaled_energies':zne_values,
                               'mitigated_value':zne_zero,'fit_coefficients':zne_coeff.tolist()}

    
    
    n1=sum(1 for inst in circuit.data if inst.operation.num_qubits==1)
    n2=sum(1 for inst in circuit.data if inst.operation.num_qubits==2)
    p1,p2=0.001,0.01
    p_eff=1.0-(1.0-p1)**n1*(1.0-p2)**n2
    pec=invert_global_depolarizing(rho,p_eff)
    results['methods']['pec_global_depolarizing_surrogate']={
      'linear_inverse_energy':expectation(pec['rho_linear_inverse'],H_pauli),
      'physical_projected_energy':expectation(pec['rho_physical_projected'],H_pauli),
      'gamma_proxy':pec['gamma_proxy'],'p_effective_approx':p_eff,
      'scope':pec['scope']}

    
    
    from mem import mitigate_probabilities
    counts_ro=sampled_counts(circuit,build_noise_model(0,0,0.02),SHOTS)
    p=np.zeros(16)
    for bitstr,count in counts_ro.items(): p[int(bitstr.replace(' ',''),2)]+=count/SHOTS
    M=np.array([[0.98,0.02],[0.02,0.98]])
    mem_p=mitigate_probabilities(p,[M]*4)
    results['methods']['measurement_error_mitigation']={
      'raw_probability_sum':float(p.sum()),'corrected_probability_sum':float(mem_p.sum()),
      'scope':'computational-basis population correction only; not a full Hamiltonian energy estimate'}

    
    
    try:
        nr=NRepMitigator(data['h1'],data['eri'],data['ecore'],shots=SHOTS,seed=SEED)
        nr_result=nr.run_from_density_matrix(rho,add_sampling_noise=True)
        results['methods']['n_representability']={
          'noisy_value':nr_result['noisy_value'],
          'mitigated_value':nr_result['mitigated_value'],
          'scope':nr_result['scope']}
    except Exception as exc:
        results['methods']['n_representability']={'status':'FAILED','error':repr(exc)}

    
    sqd_counts=sampled_counts(circuit,noise,SHOTS)
    try:
        sqd=run_sqd_from_counts(sqd_counts,data,samples_per_batch=512,num_batches=4,
                                max_iterations=10,seed=SEED,max_dim=16)
        results['sqd']={k:v for k,v in sqd.items() if k!='result'}
        print(f"SQD total energy: {sqd['energy_total']:.10f} Ha")
    except Exception as exc:
        results['sqd']={'status':'FAILED','error':repr(exc)}
        print('SQD stage failed; see results/lih_qem_results.json for the exception.')
        print(repr(exc))

    out='results/lih_qem_results.json'
    with open(out,'w') as f: json.dump(results,f,indent=2,default=lambda x:float(x))
    print('\n=== Mitigation summary ===')
    for name,entry in results['methods'].items():
        print(f"{name:38s}: {entry.get('mitigated_value',entry.get('physical_projected_energy',entry.get('status','see details')))}")
    print(f'\nSaved: {out}')
    print('\nImportant: this is a simulation development baseline. MEM and N-representability are not yet full energy-level QEM implementations; PEC is a global-channel surrogate; VD and SE use exact simulated density matrices.')

if __name__=='__main__':
    main()
