"""VQC = feature map + ansatz + readout + cross-entropy loss.

    vqc = build_vqc("zz", "real_amplitudes", n_qubits=4, reps=2, n_classes=2, optimizer=...)
"""
import os
import numpy as np
import pandas as pd
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.primitives import QMLSampler

from feature_maps import build_feature_map, ANGLE_MAX
from ansatz import build_ansatz


def readout(n_qubits, n_classes):
    """How a measured bitstring becomes a class.

    Qiskit's default (x % n_classes) reads qubit 0 for binary problems. With linear
    entanglement, qubit 0 only ever CONTROLS CNOTs, so information from later qubits
    (features) cannot reach it (see the sensitivity check below). The LAST qubits sit at the
    end of the CNOT chain and depend on every feature, so we read those instead.
      binary     : class = value of qubit n-1                      (1 qubit read)
      8 classes  : class = value of qubits n-3..n-1 as an integer  (3 qubits read, one-to-one)
    Reading few qubits = a local cost, which keeps gradients trainable on shallow
    circuits (Cerezo et al., Nat. Commun. 2021)."""
    n_bits = int(np.ceil(np.log2(n_classes)))
    if 2 ** n_bits != n_classes:
        raise ValueError("n_classes must be a power of 2 (2 or 8 here)")
    if n_qubits < n_bits:
        raise ValueError(f"{n_classes} classes need at least {n_bits} qubits, got {n_qubits}")
    shift = n_qubits - n_bits

    def interpret(x):
        # Qiskit is little-endian: bit q of the integer x is the value of qubit q.
        return x >> shift   # keeps only the top n_bits qubits (n-n_bits .. n-1)

    return interpret, n_classes


def build_circuit(fmap_name, ansatz_name, n_qubits, reps):
    fmap = build_feature_map(fmap_name, n_qubits)
    ansatz = build_ansatz(ansatz_name, n_qubits, reps)
    return fmap, ansatz


def build_vqc(fmap_name, ansatz_name, n_qubits, reps, n_classes,
              optimizer=None, initial_point=None, callback=None):
    fmap, ansatz = build_circuit(fmap_name, ansatz_name, n_qubits, reps)
    interpret, output_shape = readout(n_qubits, n_classes)
    return VQC(
        feature_map=fmap,
        ansatz=ansatz,
        loss="cross_entropy",      # log-loss: maximum likelihood, smooth gradients
        optimizer=optimizer,       # decided in the next step
        initial_point=initial_point,
        callback=callback,
        sampler=QMLSampler(),      # shots=None -> exact probabilities, no shot noise
        interpret=interpret,
        output_shape=output_shape,
    )


def readout_sensitivity(fmap_name, reps, n_qubits=4, n_draws=30, eps=1e-3, seed=42):
    """Mean |dP(qubit q = 1)/dx_f| over random inputs and weights.
    Row q = readout qubit, column f = feature. A zero means feature f cannot influence
    what qubit q reports, whatever the trained weights are."""
    rng = np.random.default_rng(seed)
    fmap, ansatz = build_circuit(fmap_name, "real_amplitudes", n_qubits, reps)
    qc = fmap.compose(ansatz)
    idx = np.arange(2 ** n_qubits)

    def p_one(x, theta):
        probs = Statevector(qc.assign_parameters(list(x) + list(theta))).probabilities()
        return np.array([probs[((idx >> q) & 1) == 1].sum() for q in range(n_qubits)])

    sens = np.zeros((n_qubits, n_qubits))
    for _ in range(n_draws):
        theta = rng.uniform(0, 2 * np.pi, ansatz.num_parameters)
        x = rng.uniform(0, ANGLE_MAX[fmap_name], n_qubits)
        base = p_one(x, theta)
        for f in range(n_qubits):
            x2 = x.copy()
            x2[f] += eps
            sens[:, f] += np.abs(p_one(x2, theta) - base) / eps
    return sens / n_draws


if __name__ == "__main__":
    from qml_data import load_scenario, to_qubits

    # (a) Readout mapping: which bits decide the class (4 qubits, little-endian).
    interp, _ = readout(4, 2)
    print("binary, 4 qubits: bitstring (q3 q2 q1 q0) -> class")
    for x in [0b0000, 0b0001, 0b0111, 0b1000, 0b1110]:
        print(f"  {x:04b} -> {interp(x)}")
    interp8, _ = readout(4, 8)
    print("8 classes, 4 qubits:", {f"{x:04b}": interp8(x) for x in [0b0001, 0b0010, 0b1000, 0b1111]})

    # (b) Evidence for reading the LAST qubit: which features each qubit can depend on.
    rows = []
    for fmap_name in ["ry", "zz"]:
        for reps in [1, 2, 3]:
            s = readout_sensitivity(fmap_name, reps)
            print(f"\n{fmap_name.upper()} + RealAmplitudes reps={reps}  (rows q0..q3, cols x0..x3)")
            print(np.round(s, 3))
            for q in range(4):
                rows.append({"feature_map": fmap_name, "reps": reps, "readout_qubit": q,
                             **{f"x{f}": round(s[q, f], 4) for f in range(4)}})
    os.makedirs("results", exist_ok=True)
    pd.DataFrame(rows).to_csv("results/readout_sensitivity.csv", index=False)

    # (c) One forward pass on real data with random (untrained) weights.
    #     Probabilities must sum to 1. ln(n_classes) is the loss of a model that guesses
    #     uniformly; random weights usually do worse (confident but random), and training
    #     must bring the loss below ln(n_classes) to have learned anything.
    rng = np.random.default_rng(0)
    for scenario, n_classes in [("Recon", 2), ("multiclass", 8)]:
        for fmap_name in ["ry", "zz"]:
            train, test, classes = load_scenario(scenario, n_train_per_class=50, seed=0)
            Xtr, Xte, ytr, yte, info = to_qubits(train, test, n_qubits=4,
                                                 angle_max=ANGLE_MAX[fmap_name])
            vqc = build_vqc(fmap_name, "real_amplitudes", 4, 2, n_classes)
            w = rng.uniform(0, 2 * np.pi, vqc.neural_network.num_weights)
            probs = vqc.neural_network.forward(Xtr, w)
            ce = -np.mean(np.log(probs[np.arange(len(ytr)), ytr] + 1e-12))
            print(f"{scenario:<10} {fmap_name}: probs shape={probs.shape}  "
                  f"sum to 1: {np.allclose(probs.sum(axis=1), 1)}  "
                  f"untrained cross-entropy={ce:.3f}  (uniform guess: ln {n_classes} = {np.log(n_classes):.3f})")