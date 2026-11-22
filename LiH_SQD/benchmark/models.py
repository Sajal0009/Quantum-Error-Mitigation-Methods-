from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

def build_noise_model(p1=0.001, p2=0.01, readout_p=0.0):
    if not (0<=p1<1 and 0<=p2<1 and 0<=readout_p<0.5):
        raise ValueError('Noise probabilities out of range.')
    nm=NoiseModel()
    if p1:
        err1=depolarizing_error(float(p1),1)
        nm.add_all_qubit_quantum_error(err1,['x','sx','ry','rz','rx','h','u','u3'])
    if p2:
        err2=depolarizing_error(float(p2),2)
        nm.add_all_qubit_quantum_error(err2,['cx'])
    if readout_p:
        ro=ReadoutError([[1-readout_p,readout_p],[readout_p,1-readout_p]])
        nm.add_all_qubit_readout_error(ro)
    return nm

def build_named_noise_models():

    from qiskit_aer.noise import amplitude_damping_error, phase_damping_error, pauli_error
    out={
      'depolarizing':build_noise_model(0.001,0.01),
      'readout':build_noise_model(0,0,0.02),
    }
    amp=amplitude_damping_error(0.01)
    phase=phase_damping_error(0.01)
    nm_amp=NoiseModel(); nm_amp.add_all_qubit_quantum_error(amp,['x','sx','ry'])
    nm_phase=NoiseModel(); nm_phase.add_all_qubit_quantum_error(phase,['x','sx','ry'])
    out['amplitude_damping']=nm_amp; out['phase_damping']=nm_phase
    bitflip=pauli_error([('X',0.005),('I',0.995)])
    nm_flip=NoiseModel(); nm_flip.add_all_qubit_quantum_error(bitflip,['x','sx','ry'])
    out['bit_flip']=nm_flip
    return out
