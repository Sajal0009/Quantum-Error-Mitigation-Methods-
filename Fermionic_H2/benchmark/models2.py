'Noise models used in the QEM benchmarks.'

from itertools import product as cartesian_product
import numpy as np

from qiskit_aer.noise import (
    NoiseModel, ReadoutError, depolarizing_error, amplitude_damping_error,
    phase_damping_error, thermal_relaxation_error, pauli_error, reset_error,
    coherent_unitary_error,
)

SINGLE_QUBIT_GATES = ["x", "sx", "h", "rx", "ry", "u", "u3"]
TWO_QUBIT_GATES = ["cx", "cz", "ecr", "iswap", "swap"]

NOISE_TYPES = [
    "depolarizing", "thermal", "thermal_relaxation",
    "amplitude_damping", "phase_damping", "generalized_amplitude_damping",
    "bit_flip", "phase_flip", "bit_phase_flip", "pauli",
    "two_qubit_depolarizing", "two_qubit_pauli", "correlated_pauli",
    "readout", "reset", "coherent", "coherent_overrotation",
    "gate_dependent", "idle_thermal", "zz_crosstalk", "crosstalk_zz",
    "combined", "ibm_calibrated",
]

def available_noise_types():
    return NOISE_TYPES.copy()

def _prob(x, name):
    x = float(x)
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {x}")
    return x

def _add_1q(nm, error, gates):
    if gates:
        nm.add_all_qubit_quantum_error(error, list(gates))

def _add_2q(nm, error, gates):
    if gates:
        nm.add_all_qubit_quantum_error(error, list(gates))

def _thermal(t1, t2, duration):
    if t1 <= 0 or t2 <= 0 or duration < 0:
        raise ValueError("T1/T2 must be positive and duration nonnegative.")
    if t2 > 2 * t1:
        raise ValueError("Physical relaxation requires T2 <= 2*T1.")
    return thermal_relaxation_error(t1, t2, duration)

def _pauli_1q(px, py, pz):
    px, py, pz = (_prob(px, "px"), _prob(py, "py"), _prob(pz, "pz"))
    if px + py + pz > 1:
        raise ValueError("px + py + pz cannot exceed 1.")
    return pauli_error([
        ("I", 1-px-py-pz), ("X", px), ("Y", py), ("Z", pz)
    ])

def _independent_2q_pauli(px, py, pz):
    one = [
        ("I", 1-px-py-pz), ("X", px), ("Y", py), ("Z", pz)
    ]
    terms = [(a+b, pa*pb) for (a,pa),(b,pb) in cartesian_product(one, repeat=2)]
    return pauli_error(terms)

def _coherent_x(theta):
    c, s = np.cos(theta/2), np.sin(theta/2)
    return coherent_unitary_error(np.array(
        [[c, -1j*s], [-1j*s, c]], dtype=complex
    ))

def _coherent_axis(theta, axis):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    I = np.eye(2, dtype=complex)
    X = np.array([[0,1],[1,0]], dtype=complex)
    Y = np.array([[0,-1j],[1j,0]], dtype=complex)
    Z = np.array([[1,0],[0,-1]], dtype=complex)
    P = axis[0]*X + axis[1]*Y + axis[2]*Z
    U = np.cos(theta/2)*I - 1j*np.sin(theta/2)*P
    return coherent_unitary_error(U)

def _zz_error(theta):
    # exp(-i theta Z⊗Z / 2), a coherent two-qubit ZZ interaction error.
    U = np.diag([
        np.exp(-1j*theta/2), np.exp(1j*theta/2),
        np.exp(1j*theta/2), np.exp(-1j*theta/2)
    ])
    return coherent_unitary_error(U)

def build_noise_model(
    noise_type,
    p=0.01, p1=None, p2=None, gamma=None,
    px=None, py=None, pz=None,
    p2_xx=None, p2_yy=None, p2_zz=None,
    t1=40000, t2=60000, gate_time_1q=50, gate_time_2q=300,
    p01=0.02, p10=0.02, p_ro=None,
    reset_p0=0.01, reset_p1=0.01,
    epsilon_1q=0.01, epsilon_2q=0.01,
    theta_x=None, theta_y=0.0, theta_z=0.0,
    excited_state_population=0.0,
    idle_time=1000, crosstalk_angle=0.01,
    gate_parameters=None, correlated_probabilities=None,
    backend=None,
    basis_gates=None, single_qubit_gates=None, two_qubit_gates=None,
):
    'Build one selected noise model. '
    name = str(noise_type).lower().strip()
    aliases = {"ibm_calibrated":"combined", "thermal_relaxation":"thermal",
               "coherent_overrotation":"coherent",
               "crosstalk_zz":"zz_crosstalk"}
    name = aliases.get(name, name)
    if name not in NOISE_TYPES:
        raise ValueError(f"Unknown noise type {noise_type!r}; available: {NOISE_TYPES}")

    sq = list(single_qubit_gates or SINGLE_QUBIT_GATES)
    tq = list(two_qubit_gates or TWO_QUBIT_GATES)
    nm = NoiseModel(basis_gates=basis_gates) if basis_gates else NoiseModel()

    if name == "depolarizing":
        a = _prob(p if p1 is None else p1, "p1")
        b = _prob(p if p2 is None else p2, "p2")
        _add_1q(nm, depolarizing_error(a, 1), sq)
        _add_2q(nm, depolarizing_error(b, 2), tq)

    elif name in ("amplitude_damping", "phase_damping",
                  "generalized_amplitude_damping", "bit_flip",
                  "phase_flip", "bit_phase_flip", "pauli"):
        g = _prob(p if gamma is None else gamma, "gamma")
        if name == "amplitude_damping":
            e1 = amplitude_damping_error(g)
        elif name == "phase_damping":
            e1 = phase_damping_error(g)
        elif name == "generalized_amplitude_damping":
            q = _prob(excited_state_population, "excited_state_population")
            # Current Qiskit Aer API: amplitude_damping_error also supports
            # non-zero equilibrium excited-state population.
            e1 = amplitude_damping_error(g, excited_state_population=q)
        elif name == "bit_flip":
            e1 = pauli_error([("X",g),("I",1-g)])
        elif name == "phase_flip":
            e1 = pauli_error([("Z",g),("I",1-g)])
        elif name == "bit_phase_flip":
            e1 = pauli_error([("Y",g),("I",1-g)])
        else:
            e1 = _pauli_1q(
                0.25*p if px is None else px,
                0.25*p if py is None else py,
                0.50*p if pz is None else pz
            )
        # Independent local channel on each qubit after a 2-qubit gate.
        e2 = e1.tensor(e1)
        _add_1q(nm, e1, sq)
        _add_2q(nm, e2, tq)

    elif name == "thermal":
        e1 = _thermal(t1, t2, gate_time_1q)
        e2local = _thermal(t1, t2, gate_time_2q)
        _add_1q(nm, e1, sq)
        _add_2q(nm, e2local.tensor(e2local), tq)

    elif name == "two_qubit_depolarizing":
        _add_2q(nm, depolarizing_error(_prob(p,"p"), 2), tq)

    elif name in ("two_qubit_pauli", "correlated_pauli"):
        if name == "two_qubit_pauli":
            x = 0.01 if px is None else _prob(px,"px")
            y = 0.01 if py is None else _prob(py,"py")
            z = 0.01 if pz is None else _prob(pz,"pz")
            if x+y+z > 1:
                raise ValueError("px+py+pz must not exceed 1.")
            e2 = _independent_2q_pauli(x,y,z)
        else:
            probs = correlated_probabilities or {
                "XX":0.002, "YY":0.002, "ZZ":0.004
            }
            terms = {"II":1.0}
            for label, value in probs.items():
                value = _prob(value, label)
                if len(label) != 2 or any(c not in "IXYZ" for c in label):
                    raise ValueError(f"Invalid two-qubit Pauli label: {label}")
                terms[label] = value
            total = sum(v for k,v in terms.items() if k != "II")
            if total > 1:
                raise ValueError("Correlated Pauli probabilities sum to > 1.")
            terms["II"] = 1-total
            e2 = pauli_error(list(terms.items()))
        _add_2q(nm, e2, tq)

    elif name == "readout":
        a = _prob(p01, "p01")
        b = _prob(p10, "p10")
        if p_ro is not None:
            a = b = _prob(p_ro, "p_ro")
        nm.add_all_qubit_readout_error(ReadoutError([
            [1-a, a], [b, 1-b]
        ]))

    elif name == "reset":
        _add_1q(nm, reset_error(
            _prob(reset_p0,"reset_p0"), _prob(reset_p1,"reset_p1")
        ), ["reset"])

    elif name == "coherent":
        tx = epsilon_1q if theta_x is None else theta_x
        _add_1q(nm, _coherent_axis(tx, [1,0,0]), sq)
        if theta_y:
            _add_1q(nm, _coherent_axis(theta_y, [0,1,0]), sq)
        if theta_z:
            _add_1q(nm, _coherent_axis(theta_z, [0,0,1]), sq)
        _add_2q(nm, _zz_error(epsilon_2q), tq)

    elif name == "gate_dependent":
        # Distinct depolarizing probabilities by instruction name.
        defaults = {
            "x":0.0005, "sx":0.001, "h":0.001, "rx":0.001,
            "ry":0.001, "u":0.001, "u3":0.001,
            "cx":0.01, "cz":0.01, "ecr":0.012, "iswap":0.015, "swap":0.02
        }
        if gate_parameters:
            defaults.update(gate_parameters)
        for gate in sq:
            if gate in defaults:
                _add_1q(nm, depolarizing_error(_prob(defaults[gate],gate),1), [gate])
        for gate in tq:
            if gate in defaults:
                _add_2q(nm, depolarizing_error(_prob(defaults[gate],gate),2), [gate])

    elif name == "idle_thermal":
        # Acts only on explicit id instructions. Insert id/delay operations to
        # represent idle periods; this is not automatic scheduler-aware decoherence.
        e = _thermal(t1, t2, idle_time)
        _add_1q(nm, e, ["id"])

    elif name == "zz_crosstalk":
        # Simplified pairwise coherent ZZ error on the active two-qubit gate.
        # This is not a full spectator/nonlocal crosstalk model.
        _add_2q(nm, _zz_error(crosstalk_angle), tq)

    elif name == "combined":
        # IBM-calibration-derived model. Do not substitute guessed global
        # probabilities: the backend supplies per-qubit and per-gate data.
        if backend is None:
            raise ValueError(
                "noise_type='combined' now requires an IBM-compatible backend "
                "object. Obtain one with QiskitRuntimeService().backend(...), "
                "then call build_noise_model('combined', backend=backend). "
                "No backend-independent IBM error ratio exists."
            )
        return build_ibm_calibrated_noise_model(backend)

    return nm

def build_ibm_calibrated_noise_model(
    backend,
    gate_error=True,
    readout_error=True,
    thermal_relaxation=True,
    temperature=0,
    gate_lengths=None,
    gate_length_units="ns",
):
    'Build an approximate Aer noise model from IBM superconducting backend calibration properties. '
    if backend is None:
        raise ValueError("A valid backend object is required.")

    try:
        from qiskit_aer.noise import NoiseModel
        model = NoiseModel.from_backend(
            backend,
            gate_error=gate_error,
            readout_error=readout_error,
            thermal_relaxation=thermal_relaxation,
            temperature=temperature,
            gate_lengths=gate_lengths,
            gate_length_units=gate_length_units,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not construct an IBM-calibrated Aer noise model from this "
            "backend. Check that it exposes compatible calibration properties "
            "and that your qiskit-aer version supports NoiseModel.from_backend()."
        ) from exc

    return model

IBM_REFERENCE_BACKEND = "ibm_kingston"

def get_ibm_kingston_backend(service=None):
    'Return the IBM Quantum backend object for ibm_kingston. '
    if service is None:
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
        except ImportError as exc:
            raise ImportError(
                "qiskit-ibm-runtime is required. Install it and configure "
                "your IBM Quantum account before requesting ibm_kingston."
            ) from exc
        service = QiskitRuntimeService()

    try:
        return service.backend(IBM_REFERENCE_BACKEND)
    except Exception as exc:
        raise RuntimeError(
            f"Could not access IBM backend {IBM_REFERENCE_BACKEND!r}. "
            "Check that it is available to your account and that your "
            "IBM Quantum credentials are valid."
        ) from exc

def build_ibm_kingston_noise_model(
    service=None,
    gate_error=True,
    readout_error=True,
    thermal_relaxation=True,
    temperature=0,
):
    'Build the combined Aer noise model from ibm_kingston calibrations. '
    backend = get_ibm_kingston_backend(service=service)
    return build_ibm_calibrated_noise_model(
        backend,
        gate_error=gate_error,
        readout_error=readout_error,
        thermal_relaxation=thermal_relaxation,
        temperature=temperature,
    )

def print_noise_model(noise_model):
    print("========== NOISE MODEL ==========")
    print(noise_model)
    print("=================================")
