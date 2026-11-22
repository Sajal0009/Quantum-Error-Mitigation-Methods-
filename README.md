
This repository contains the code for my B.Tech project, Quantum Error Mitigation in Noisy Quantum Circuits.
Project Overview
The project focuses on implementing and comparing seven quantum error mitigation (QEM) methods:
- Zero-noise extrapolation (ZNE)
- Probabilistic error cancellation (PEC)
- Measurement-error mitigation (MEM)
- Symmetry verification (SV)
- Virtual distillation (VD)
- Subspace expansion (SE)
- Fermionic N-representability (NRep)
These methods are studied using different noise models and quantum circuits. The main H₂ study uses a four-qubit, two-electron, particle-number-conserving fermionic hardware-efficient ansatz (HEA).
Studies Included
- Individual-noise benchmarking: Comparison of the mitigation methods under different noise channels and circuit depths.
- Combined-noise simulations: Testing individual methods and selected combinations under combined noise.
- IBM Quantum hardware experiments: Running selected measurement-based mitigation methods for the H₂ circuit on real quantum hardware.
- LiH CAS(2,2): Applying the QEM workflow to a lithium hydride active-space example.
The hardware experiments use methods that can be implemented with measurements on the QPU. Some of the implemented methods require simulator features, such as access to a density matrix, and are therefore not directly used in the hardware workflow.
Reference Energy
For the circuit benchmarks, the error is generally calculated relative to the noiseless expectation value of the same fixed-parameter circuit. The circuit parameters are not reoptimized through a VQE loop, so this reference should not be confused with the molecular full configuration interaction (FCI) ground-state energy.
Running the Code
The code was developed and run on qBraid using a Qiskit environment. The notebooks can be opened in Jupyter, and the Python scripts can be run from the repository directory.

To run the project on a personal computer, install Python and the required packages and libraries beforehand. The environment should use compatible versions of Qiskit, Qiskit Aer, and the other scientific Python libraries used by the code. Additional dependencies may be required for specific workflows, such as the LiH active-space calculation.

For the IBM Quantum hardware experiments, configure your own IBM Quantum account and ensure that you have access to the backend specified in the relevant script or notebook. Account credentials and API tokens are not included in this repository.
Repository Contents
The repository includes the Python implementations of the QEM methods, shared utility and noise-model code, experiment scripts, and Jupyter notebooks for the simulation and hardware studies. The relevant scripts and notebooks contain the configuration options and output locations for their respective experiments.
