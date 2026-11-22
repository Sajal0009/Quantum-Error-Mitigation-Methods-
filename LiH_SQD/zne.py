import numpy as np

def richardson_extrapolate(scales, values, degree=None):

    x = np.asarray(scales, dtype=float)
    y = np.asarray(values, dtype=float)

    if len(x) != len(y):
        raise ValueError(
            "Scales and values must have equal lengths."
        )

    if len(x) < 2:
        raise ValueError(
            "At least two data points are required."
        )

    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Scales and values must be finite.")

    if len(np.unique(x)) != len(x):
        raise ValueError(
            "Actual noise scales must be unique."
        )

    if np.any(x <= 0):
        raise ValueError("Noise scales must be positive.")

    if degree is None:
        degree = len(x) - 1

    degree = min(int(degree), len(x) - 1)

    if degree < 1:
        raise ValueError("Polynomial degree must be at least 1.")

    coeff = np.polyfit(x, y, degree)

    zero_noise_value = float(
        np.polyval(coeff, 0.0)
    )

    return zero_noise_value, coeff

def fold_global(circuit, scale_factor):

    scale_factor = int(scale_factor)

    if scale_factor < 1 or scale_factor % 2 == 0:
        raise ValueError(
            "Folding requires positive odd integer factors: 1, 3, 5, ..."
        )

    # Create a new circuit with the same registers.
    from qiskit import QuantumCircuit

    out = QuantumCircuit(
        *circuit.qregs,
        *circuit.cregs,
        name=f"{circuit.name}_fold{scale_factor}"
    )

    out.global_phase = circuit.global_phase

    repetitions = (scale_factor - 1) // 2

    for inst in circuit.data:

        op = inst.operation

        # Barriers do not represent physical gates.
        if op.name == "barrier":
            out.append(
                op,
                inst.qubits,
                inst.clbits
            )
            continue

        # Reject non-unitary instructions.
        if op.name in ("measure", "reset"):
            raise ValueError(
                f"Cannot fold non-unitary instruction: {op.name}"
            )

        # Qiskit 2.x compatibility:
        # Do not use op.is_unitary().
        # Check whether the operation has a valid inverse.
        try:
            inverse_op = op.inverse()
        except Exception as exc:
            raise ValueError(
                f"Instruction '{op.name}' cannot be folded. "
                "Only invertible unitary gates are supported."
            ) from exc

        # U (U† U)^k
        for _ in range(repetitions):

            out.append(
                op,
                inst.qubits,
                inst.clbits
            )

            out.append(
                inverse_op,
                inst.qubits,
                inst.clbits
            )

        # Final U
        out.append(
            op,
            inst.qubits,
            inst.clbits
        )

    return out

def validate_scale_factors(scale_factors):

    scales = [int(s) for s in scale_factors]

    if not scales:
        raise ValueError("No ZNE scale factors supplied.")

    if any(s < 1 or s % 2 == 0 for s in scales):
        raise ValueError(
            "Use positive odd folding factors such as [1, 3, 5]."
        )

    if len(set(scales)) != len(scales):
        raise ValueError(
            "ZNE scale factors must be unique."
        )

    return scales
