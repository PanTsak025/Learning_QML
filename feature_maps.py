"""Feature maps (data encoding circuits) compared in the experiments.

    fmap = build_feature_map("zz", n_qubits=4)
    X_tr, X_te, ... = to_qubits(train, test, n_qubits=4, angle_max=ANGLE_MAX["zz"])
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector, state_fidelity

# Each feature is mapped onto half the period of its encoding gate, so that distinct values
# give distinct single-qubit states and the two ends of the range are orthogonal:
#   RY(x) has period 2*pi -> x in [0, pi]   : |0> ... |1>
#   P(2x) has period pi   -> x in [0, pi/2] : |+> ... |->   (ZZ's single-qubit part, after H)
ANGLE_MAX = {"ry": np.pi, "zz": np.pi / 2}

ZZ_REPS = 2               # depth of the original ZZ map (Havlicek et al., Nature 2019)
ZZ_ENTANGLEMENT = "linear"  # neighbour pairs only: CNOTs grow linearly with qubits, hardware-friendly


def build_feature_map(name, n_qubits):
    """'ry': one RY rotation per qubit, no entanglement (product-state baseline).
       'zz': Havlicek ZZ feature map, entangling, linear connectivity."""
    if name == "ry":
        x = ParameterVector("x", n_qubits)
        qc = QuantumCircuit(n_qubits, name="RY-encoding")
        for q in range(n_qubits):
            qc.ry(x[q], q)
        return qc
    if name == "zz":
        return zz_feature_map(n_qubits, reps=ZZ_REPS, entanglement=ZZ_ENTANGLEMENT,
                              parameter_prefix="x")
    raise ValueError("feature map must be 'ry' or 'zz'")


if __name__ == "__main__":
    # 1. What the circuits look like, and what they cost as qubits grow (report in the paper).
    for name in ["ry", "zz"]:
        print(f"\n=== {name.upper()}  data range [0, {ANGLE_MAX[name]:.4f}]")
        print(build_feature_map(name, 4).draw(fold=120))
        for n in [2, 4, 6, 8]:
            qc = build_feature_map(name, n)
            print(f"  {n} qubits: depth={qc.depth():>2}  CNOTs={qc.num_nonlocal_gates():>2}  "
                  f"gates={dict(qc.count_ops())}")

    # 2. Why the angle ranges differ: overlap between the two ends of the data range
    #    (0 = perfectly distinguishable, 1 = identical states).
    def overlap(name, a, b, n=2):
        qc = build_feature_map(name, n)
        state = lambda v: Statevector(qc.assign_parameters([v] * n))
        return state_fidelity(state(a), state(b))

    print("\nOverlap of the range ends:")
    print("  RY  0 vs pi   :", round(overlap("ry", 0, np.pi), 4))
    one = QuantumCircuit(1); one.h(0)
    for v in [0, np.pi / 2, np.pi]:
        c = one.copy(); c.p(2 * v, 0)
        print(f"  ZZ single-qubit part, x={v:.4f}: state {np.round(Statevector(c).data, 3)}")
    print("  full ZZ 0 vs pi/2:", round(overlap("zz", 0, np.pi / 2), 4))