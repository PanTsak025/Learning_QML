import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

small_data = [
    [-0.194, 0.114, -0.006, 0.301, -0.359, -0.088, -0.156, 0.342, -0.016, 0.143, 1],
    [-0.1, 0.002, 0.244, 0.127, -0.064, -0.086, 0.072, 0.043, -0.053, 0.02, -1],
]

train_data = [small_data[0][:-1], small_data[1][:-1]]
