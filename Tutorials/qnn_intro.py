from qiskit_machine_learning.utils import algorithm_globals
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.circuit import ParameterVector

from matplotlib import pyplot as plt

algorithm_globals.random_seed = 42

params1 = [Parameter("input1"), Parameter("weight1")]
qc1 = QuantumCircuit(1)      # create a quantum circuit with 1 qubit
qc1.h(0)                    # apply a Hadamard gate to the qubit
qc1.ry(params1[0], 0)        # apply a rotation around the y-axis by an angle specified by the first parameter
qc1.rx(params1[1], 0)         # apply a rotation around the x-axis by an angle specified by the second parameter
qc1.draw("mpl", style="clifford")             

observable1 = SparsePauliOp.from_list([("Y"*qc1.num_qubits, 1)])  

estimator = Estimator()
estimator_qnn = EstimatorQNN(
    circuit=qc1,
    observables=observable1,
    input_params=[params1[0]],
    weight_params=[params1[1]],
    estimator=estimator,
)

inputs2 = ParameterVector("input", 2)
weights2 = ParameterVector("weight", 4)
print(f"input parameters: {[str(item) for item in inputs2.params]}")
print(f"weight parameters: {[str(item) for item in weights2.params]}")

qc2 = QuantumCircuit(2)
qc2.ry(inputs2[0], 0)
qc2.ry(inputs2[1], 1)
qc2.cx(0, 1)
qc2.ry(weights2[0], 0)
qc2.ry(weights2[1], 1)
qc2.cx(0, 1)
qc2.ry(weights2[2], 0)
qc2.ry(weights2[3], 1)

qc2.draw("mpl", style="clifford")
plt.show()

sampler = Sampler()
sampler_qnn = SamplerQNN(
    circuit=qc2,
    input_params=inputs2,
    weight_params=weights2,
    sampler=sampler
    )



