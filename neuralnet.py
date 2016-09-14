import pandas
import numpy as np

# Forward pass of a regular layer in a neural network.
# x and w have to be multiplyable numpy matrices.
def nodes_forward(x, w):
	return x.dot(w)

# Forward pass of a relu layer in a neural network.
def relu_forward(x):
	return np.maximum(x, 0)
