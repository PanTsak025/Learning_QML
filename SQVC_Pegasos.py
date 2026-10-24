from sklearn.datasets import make_blobs
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from qiskit.circuit.library import z_feature_map
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.algorithms import PegasosQSVC


features, labels = make_blobs(n_samples=20, n_features=2, centers=2, random_state=3, shuffle=True)
features = MinMaxScaler(feature_range=(0,np.pi)).fit_transform(features)
train_features, test_features, train_labels, test_labels = train_test_split(
    features,labels, train_size=15, shuffle=False
)

num_qubits = 2      # no. features = no. qubits
tau = 100             # steps during training
C = 1000             # regularization parameter

sampler = Sampler()
fidelity = ComputeUncompute(sampler=sampler)
algorithm_globals.random_seed = 12345

feature_map = z_feature_map(feature_dimension=num_qubits, reps=1)
qkernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)

pegasos_qsvc = PegasosQSVC(quantum_kernel=qkernel, C=C, num_steps=tau)
pegasos_qsvc.fit(train_features,train_labels)
pegasos_score = pegasos_qsvc.score(test_features, test_labels)
print(f"PegasosQSVC Classification test score: {pegasos_score}")

grid_step = 0.2
margin = 0.2
grid_x, grid_y = np.meshgrid(
    np.arange(-margin, np.pi+margin,grid_step), np.arange(-margin,np.pi+margin,grid_step)
)

meshgrid_features = np.column_stack((grid_x.ravel(), grid_y.ravel()))
meshgrid_colors = pegasos_qsvc.predict(meshgrid_features)


import matplotlib.pyplot as plt

plt.figure(figsize=(5, 5))
meshgrid_colors = meshgrid_colors.reshape(grid_x.shape)
plt.pcolormesh(grid_x, grid_y, meshgrid_colors, cmap="RdBu", shading="auto")

plt.scatter(
    train_features[:, 0][train_labels == 0],
    train_features[:, 1][train_labels == 0],
    marker="s",
    facecolors="w",
    edgecolors="r",
    label="A train",
)
plt.scatter(
    train_features[:, 0][train_labels == 1],
    train_features[:, 1][train_labels == 1],
    marker="o",
    facecolors="w",
    edgecolors="b",
    label="B train",
)

plt.scatter(
    test_features[:, 0][test_labels == 0],
    test_features[:, 1][test_labels == 0],
    marker="s",
    facecolors="r",
    edgecolors="r",
    label="A test",
)
plt.scatter(
    test_features[:, 0][test_labels == 1],
    test_features[:, 1][test_labels == 1],
    marker="o",
    facecolors="b",
    edgecolors="b",
    label="B test",
)

plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
plt.title("Pegasos Classification")
plt.show()

