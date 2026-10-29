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
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score

algorithm_globals.random_seed = 12345
adhoc_dimension = 2
train_features, train_labels, test_features, test_labels, adhoc_total = ad_hoc_data(
    training_size=25,
    test_size=0,
    n=adhoc_dimension,
    gap=0.6,
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

plt.show()
adhoc_feature_map = zz_feature_map(feature_dimension=adhoc_dimension, reps=2, entanglement="linear")
sampler = Sampler()
fidelity = ComputeUncompute(sampler=sampler)
adhoc_kernel = FidelityQuantumKernel(feature_map=adhoc_feature_map, fidelity=fidelity)              #kernel is defined here.

adhoc_matrix_train = adhoc_kernel.evaluate(x_vec=train_features)
adhoc_matrix_test = adhoc_kernel.evaluate(x_vec=test_features, y_vec=train_features)
fig,axs = plt.subplots(1,1,figsize=(10,5))
axs.imshow(np.asmatrix(adhoc_matrix_train),interpolation="nearest",origin="upper",cmap="Blues")
axs.set_title("Ad Hoc Kernel Matrix (Train)")                                                      #setup the matrices for the second way
plt.show()

adhoc_spectral = SpectralClustering(2,affinity="precomputed")
cluster_labels = adhoc_spectral.fit_predict(adhoc_matrix_train)
cluster_score = normalized_mutual_info_score(cluster_labels, train_labels)
print(f"Ad Hoc Spectral Clustering score: {cluster_score}")
