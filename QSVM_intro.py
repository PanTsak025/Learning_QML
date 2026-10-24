from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.datasets import ad_hoc_data
import numpy as np
import matplotlib.pyplot as plt

from qiskit.circuit.library import zz_feature_map
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel

from sklearn.svm import SVC
from qiskit_machine_learning.algorithms import QSVC


algorithm_globals.random_seed = 12345
adhoc_dimension = 2
train_features, train_labels, test_features, test_labels, adhoc_total = ad_hoc_data(
    training_size=20,
    test_size=5,
    n=adhoc_dimension,
    gap=0.3,
    plot_data=False,
    one_hot=False,
    include_sample_total=True,
)

def plot_features(ax,features,labels,class_label,marker,face,edge,label):
    ax.scatter(
        features[labels == class_label, 0],
        features[labels == class_label, 1],
        marker=marker,
        facecolors=face,
        edgecolors=edge,
        label=label,
    )

def plot_dataset(train_features, train_labels, test_features, test_labels, adhoc_total):
    plt.figure(figsize=(5, 5))
    plt.ylim(0,2*np.pi)
    plt.xlim(0,2*np.pi)
    plt.imshow(
        np.asmatrix(adhoc_total).T,
        interpolation="nearest",
        origin="lower",
        cmap="RdBu",
        extent=[0, 2 * np.pi, 0, 2 * np.pi],
    )

plot_features(plt, train_features, train_labels, 0, "o", "white", "black", "Class 0 (train)")
plot_features(plt, train_features, train_labels, 1, "o", "black", "white", "Class 1 (train)")
plot_features(plt, test_features, test_labels, 0, "s", "white", "black", "Class 0 (test)")
plot_features(plt, test_features, test_labels, 1, "s", "black", "white", "Class 1 (test)")
plt.title("Ad Hoc Data Set")                                                                 # preparation of data is complete.

adhoc_feature_map = zz_feature_map(feature_dimension=adhoc_dimension, reps=2, entanglement="linear")
sampler = Sampler()
fidelity = ComputeUncompute(sampler=sampler)
adhoc_kernel = FidelityQuantumKernel(feature_map=adhoc_feature_map, fidelity=fidelity)              #kernel is defined here.

adhoc_svc = SVC(kernel=adhoc_kernel.evaluate)                                               #first way of using the kernel for SVC classification
adhoc_svc.fit(train_features, train_labels)
adhoc_score_callable_function = adhoc_svc.score(test_features, test_labels)
print(f"Ad Hoc SVC score (callable function): {adhoc_score_callable_function}")

adhoc_matrix_train = adhoc_kernel.evaluate(x_vec=train_features)
adhoc_matrix_test = adhoc_kernel.evaluate(x_vec=test_features, y_vec=train_features)
fig,axs = plt.subplots(1,2,figsize=(10,5))
axs[0].imshow(np.asmatrix(adhoc_matrix_train),interpolation="nearest",origin="upper",cmap="Blues")
axs[0].set_title("Ad Hoc Kernel Matrix (Train)")
axs[1].imshow(np.asmatrix(adhoc_matrix_test),interpolation="nearest",origin="upper",cmap="Reds")
axs[1].set_title("Ad Hoc Kernel Matrix (Test)")                                                       #setup the matrices for the second way
plt.show()

adhoc_svc = SVC(kernel="precomputed")                                                                 #second way of using the kernel for SVC classification
adhoc_svc.fit(adhoc_matrix_train, train_labels)
adhoc_score_precomputed_matrix = adhoc_svc.score(adhoc_matrix_test, test_labels)
print(f"Ad Hoc SVC score (precomputed matrix): {adhoc_score_precomputed_matrix}")          #second way of using kernel for SVC classification.

qsvc = QSVC(quantum_kernel=adhoc_kernel)
qsvc.fit(train_features, train_labels)
qsvc_score = qsvc.score(test_features, test_labels)
print(f"Ad Hoc QSVC score: {qsvc_score}")                                         # QSVC classification using the same kernel.

print(f"Classsification Model:                              | Accuracy Score")
print(f"----------------------------------------------------|----------------")
print(f"Ad Hoc SVC score (callable function):               | {adhoc_score_callable_function}")
print(f"Ad Hoc SVC score (precomputed matrix):              | {adhoc_score_precomputed_matrix}")
print(f"Ad Hoc QSVC score:                                  | {qsvc_score}")