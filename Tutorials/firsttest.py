import pennylane as qp
qml = qp
from pennylane import numpy as np

dev = qp.device("default.qubit", wires=2)
@qp.qnode(dev)
def my_first_quantum_function(theta):
    qp.RX(theta, wires=0)
    qp.PauliY(wires=1)
    qp.Hadamard(wires=0)
    qp.Hadamard(wires=1)
    return qp.state()

print(my_first_quantum_function(np.pi/4))