from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import seaborn as sns
import time
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from qiskit_machine_learning.utils import algorithm_globals
from sklearn.svm import SVC

from qiskit.circuit.library import zz_feature_map, real_amplitudes, efficient_su2
from qiskit_machine_learning.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.algorithms.classifiers import VQC

from sklearn.decomposition import PCA


iris_data = load_iris()
#print(iris_data.DESCR) 4 features, 3 classes, 150 samples (50 each)
features = iris_data.data
labels = iris_data.target

features = PCA(n_components=2).fit_transform(features)

features = MinMaxScaler().fit_transform(features)

algorithm_globals.random_seed = 123

plt.rcParams['figure.figsize'] = [6, 6]
plt.figure()
sns.scatterplot(x=features[:, 0], y=features[:, 1], hue=labels, palette="tab10")
plt.title("Iris after PCA (2 components)")

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2,
                             random_state=algorithm_globals.random_seed)

svc = SVC()
svc.fit(train_features, train_labels)

train_score_c2 = svc.score(train_features, train_labels)
test_score_c2 = svc.score(test_features, test_labels)

print(f"Classical SVC on the training set: {train_score_c2:.4f}")
print(f"Classical SVC on the test set: {test_score_c2:.4f}")

num_features = features.shape[1]
feature_map = zz_feature_map(feature_dimension=num_features, reps=1)
ansatz = real_amplitudes(num_qubits=num_features, reps=3)

optimizer = COBYLA(maxiter=20)

sampler = Sampler(seed=algorithm_globals.random_seed)  # fixed shot noise -> repeatable runs

objective_func_vals = []
plt.rcParams['figure.figsize'] = [12, 6]

def callback_graph(weights, obj_func_eval):
    # Only record values here; the graph is drawn once after training
    objective_func_vals.append(obj_func_eval)

vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    sampler=sampler,
    callback=callback_graph,
)

objective_func_vals = []
plt.rcParams['figure.figsize'] = [12, 6]

start = time.time()
vqc.fit(train_features, train_labels)
elapsed = time.time() - start

print(f"Training time: {elapsed:.2f} seconds")

plt.figure()
plt.title("Objective function value against iteration")
plt.xlabel("Iteration")
plt.ylabel("Objective function value")
plt.plot(range(len(objective_func_vals)), objective_func_vals)

train_score_q2 = vqc.score(train_features, train_labels)
test_score_q2 = vqc.score(test_features, test_labels)

print(f"Quantum VQC on the training set: {train_score_q2:.4f}")
print(f"Quantum VQC on the test set: {test_score_q2:.4f}")

ansatz = efficient_su2(num_qubits=num_features, reps=3)
optimizer = COBYLA(maxiter=100)

vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    sampler=sampler,
    callback=callback_graph,
)

objective_func_vals = []

start = time.time()
vqc.fit(train_features, train_labels)
elapsed = time.time() - start

print(f"Training time: {elapsed:.2f} seconds")
plt.figure()
plt.title("Objective function value against iteration")
plt.xlabel("Iteration")
plt.ylabel("Objective function value")
plt.plot(range(len(objective_func_vals)), objective_func_vals)

train_score_q2_efficient = vqc.score(train_features, train_labels)
test_score_q2_efficient = vqc.score(test_features, test_labels)

print(f"Quantum VQC (efficient SU2) on the training set: {train_score_q2_efficient:.4f}")
print(f"Quantum VQC (efficient SU2) on the test set: {test_score_q2_efficient:.4f}")

print(f"Model comparison:")
print(f"Classical SVC - Training: {train_score_c2:.4f}, Test: {test_score_c2:.4f}")
print(f"Quantum VQC - Training: {train_score_q2:.4f}, Test: {test_score_q2:.4f}")
print(f"Quantum VQC (efficient SU2) - Training: {train_score_q2_efficient:.4f}, Test: {test_score_q2_efficient:.4f}")

plt.show()
