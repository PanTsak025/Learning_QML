from qiskit import QuantumCircuit
from matplotlib import pyplot as plt          
# x = 4                                          # Basis Encoding
# y = 8
# z = 5

# x_bits = [int(b) for b in format(x, "04b")]   #4 is converted into 4-bit binary list [0,1,0,1]
# y_bits = [int(b) for b in format(y, "04b")]   # same for 8
# z_bits = [int(b) for b in format(z, "04b")]   # same for 5

# all_bits = x_bits + y_bits + z_bits             #combine 3 x 4 = 12 bit array list

# qc = QuantumCircuit(12)                      # create 12 qubit circuit

# for idx,bit in enumerate(all_bits):          # add x-gate wherever there is a 1.
#     if bit ==1:
#         qc.x(idx)

# qc.draw("mpl")
# plt.show()

import math

# desired_state = [                             # Amplitude Encoding
#     1 / math.sqrt(105) * 4,
#     1 / math.sqrt(105) * 8,
#     1 / math.sqrt(105) * 5,
#     1 / math.sqrt(105) * 0,
# ]

# qc = QuantumCircuit(2)
# qc.initialize(desired_state, [0,1])

# qc.decompose(reps=5).draw(output="mpl")
# plt.show()

from qiskit.quantum_info import Statevector
from math import pi
import numpy as np
from qiskit.visualization.bloch import Bloch
from qiskit.visualization.state_visualization import _bloch_multivector_data

# qc = QuantumCircuit(1)                                               Angle Encoding
# state1 = Statevector.from_instruction(qc)
# qc.ry(pi /2,0)
# state2 = Statevector.from_instruction(qc)
# states = state1, state2

# def plot_Nstates(states, axis, plot_trace_points=True):
#     """This function plots N states to 1 Bloch sphere"""
#     bloch_vecs = [_bloch_multivector_data(s)[0] for s in states]

#     if axis is None:
#         bloch_plot = Bloch()
#     else:
#         bloch_plot = Bloch(axes=axis)

#     bloch_plot.add_vectors(bloch_vecs)

#     if len(states) > 1:

#         def rgba_map(x, num):
#             g = (0.95 - 0.05) / (num - 1)
#             i = 0.95 - g * num
#             y = g * x + i
#             return (0.0, y, 0.0, 0.7)

#         num = len(states)
#         bloch_plot.vector_color = [rgba_map(x, num) for x in range(1, num + 1)]

#     bloch_plot.vector_width = 3
#     bloch_plot.vector_style = "simple"

#     if plot_trace_points:

#         def trace_points(bloch_vec1, bloch_vec2):
#             # bloch_vec = (x,y,z)
#             n_points = 15
#             thetas = np.arccos([bloch_vec1[2], bloch_vec2[2]])
#             phis = np.arctan2(
#                 [bloch_vec1[1], bloch_vec2[1]], [bloch_vec1[0], bloch_vec2[0]]
#             )
#             if phis[1] < 0:
#                 phis[1] = phis[1] + 2 * pi
#             angles0 = np.linspace(phis[0], phis[1], n_points)
#             angles1 = np.linspace(thetas[0], thetas[1], n_points)

#             xp = np.cos(angles0) * np.sin(angles1)
#             yp = np.sin(angles0) * np.sin(angles1)
#             zp = np.cos(angles1)
#             pnts = [xp, yp, zp]
#             bloch_plot.add_points(pnts)
#             bloch_plot.point_color = "k"
#             bloch_plot.point_size = [4] * len(bloch_plot.points)
#             bloch_plot.point_marker = ["o"]

#         for i in range(len(bloch_vecs) - 1):
#             trace_points(bloch_vecs[i], bloch_vecs[i + 1])

#     bloch_plot.sphere_alpha = 0.05
#     bloch_plot.frame_alpha = 0.15
#     bloch_plot.figsize = [4, 4]

#     bloch_plot.render()


# plot_Nstates(states, axis=None, plot_trace_points=True)
# plt.show()

# qc = QuantumCircuit(1)                                              Phase Encoding
# qc.h(0)
# state1 = Statevector.from_instruction(qc)
# qc.p(pi/2,0)
# state2 = Statevector.from_instruction(qc)

# states = state1, state2

# qc.draw("mpl", scale=1)
# plt.show()

# qc = QuantumCircuit(1)                                            Dense Angle Encoding
# state1 = Statevector.from_instruction(qc)
# qc.ry(3*pi/8,0)
# state2 = Statevector.from_instruction(qc)
# qc.rz(7*pi/4,0)
# state3 = Statevector.from_instruction(qc)
# states = state1, state2, state3

from qiskit.circuit.library import efficient_su2

# x = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2]                         EfficientSU2
# circuit = efficient_su2(num_qubits=3, reps=1, insert_barriers=True)
# encode = circuit.assign_parameters(x)
# encode.decompose().draw(output="mpl")
# plt.show()

from qiskit.circuit.library import z_feature_map

# z_feature_map = z_feature_map(feature_dimension=2, reps=3)                    Z Feature Map
# z_feature_map = z_feature_map.assign_parameters([(1/2)*pi/2,(1/2)*pi/3])
# z_feature_map.decompose().draw("mpl")
# plt.show()

from qiskit.circuit.library import zz_feature_map

# feature_dim=2                                                              ZZ Feature Map
# zz_feature_map = zz_feature_map(
#     feature_dimension=feature_dim, entanglement="linear", reps=1
# )
# zz_feature_map.decompose(reps=1).draw("mpl",scale=1)
# plt.show()

from qiskit.circuit.library import pauli_feature_map

feature_dim=3
pfmap = pauli_feature_map(
    feature_dimension=feature_dim, entanglement="linear", reps=1, paulis=["Y","XX"]
)

pfmap.decompose().draw("mpl", scale=1.5)
plt.show()