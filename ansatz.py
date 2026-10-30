"""Trainable circuits (ansatz) and their expressibility (Sim et al. 2019).

    ansatz = build_ansatz("real_amplitudes", n_qubits=4, reps=2)
"""
import os
import numpy as np
import pandas as pd
from qiskit.circuit.library import real_amplitudes, efficient_su2
from qiskit.quantum_info import Statevector

ANSATZ_ENTANGLEMENT = "linear"   # same connectivity as the feature map (Qiskit default is reverse_linear)
CANDIDATE_REPS = [1, 2, 3]       # beyond 3, expressibility stops improving (see table)


def build_ansatz(name, n_qubits, reps):
    """'real_amplitudes': RY layers + CNOT chain (real-valued states).
       'efficient_su2'  : RY+RZ layers + CNOT chain (complex states)."""
    if name == "real_amplitudes":
        return real_amplitudes(n_qubits, reps=reps, entanglement=ANSATZ_ENTANGLEMENT)
    if name == "efficient_su2":
        return efficient_su2(n_qubits, reps=reps, entanglement=ANSATZ_ENTANGLEMENT)
    raise ValueError("ansatz must be 'real_amplitudes' or 'efficient_su2'")


def effective_parameters(name, qc):
    """EfficientSU2 ends with RZ gates right before measurement; they can't change the
    measured probabilities, so those n parameters have no effect."""
    return qc.num_parameters - (qc.num_qubits if name == "efficient_su2" else 0)


def expressibility(qc, n_pairs=5000, bins=75, seed=42):
    """KL divergence between the circuit's state-overlap distribution and that of random
    (Haar) states. Lower = more expressive. Settings follow Sim et al. (2019):
    parameters uniform in [0, 2pi], 5000 pairs, 75 histogram bins."""
    rng = np.random.default_rng(seed)
    dim = 2 ** qc.num_qubits
    overlaps = np.empty(n_pairs)
    for i in range(n_pairs):
        a = Statevector(qc.assign_parameters(rng.uniform(0, 2 * np.pi, qc.num_parameters))).data
        b = Statevector(qc.assign_parameters(rng.uniform(0, 2 * np.pi, qc.num_parameters))).data
        overlaps[i] = abs(np.vdot(a, b)) ** 2
    edges = np.linspace(0, 1, bins + 1)
    p, _ = np.histogram(overlaps, bins=edges)
    p = p / p.sum()
    # Haar overlap distribution, integrated over each bin: P(F) = (dim-1)(1-F)^(dim-2)
    q = (1 - edges[:-1]) ** (dim - 1) - (1 - edges[1:]) ** (dim - 1)
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


if __name__ == "__main__":
    # Characterise every candidate (and reps 4-5, to show where expressibility levels off).
    # 5000 pairs per cell, as in Sim et al.; takes several minutes.
    rows = []
    for n in [4, 8]:
        for name in ["real_amplitudes", "efficient_su2"]:
            for reps in [1, 2, 3, 4, 5]:
                qc = build_ansatz(name, n, reps)
                rows.append({"qubits": n, "ansatz": name, "reps": reps,
                             "parameters": qc.num_parameters,
                             "effective_parameters": effective_parameters(name, qc),
                             "cnots": qc.num_nonlocal_gates(), "depth": qc.depth(),
                             "expressibility_kl": round(expressibility(qc), 4)})
                print(rows[-1])
    os.makedirs("results", exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv("results/ansatz_expressibility.csv", index=False)
    print(table.pivot_table(index=["qubits", "ansatz"], columns="reps",
                            values="expressibility_kl"))
    print(build_ansatz("real_amplitudes", 4, 2).decompose().draw(fold=120))