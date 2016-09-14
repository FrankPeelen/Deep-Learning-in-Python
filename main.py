

def main():
	# Take the CIFAR-10 Data, and create an algorithm that can label images from the test set with as low as possible error.


	# Download, load, and prepare the CIFAR-10 data.

	# Download the CIFAR-10 data.
	# DONE

	## Load the CIFAR-10 data into C++.

	"""
	For each file:
	Input = 30730000 bytes
	Desired output = y(10.000,1) & x(10.000,3072)
	"""

	# Fix and correct missing data.

	#  Preprocess data --> Normalize + zero-mean

	"""
	Fit the data neatly into:
	x_train 	& 	y_train
	x_val 	& 	y_val
	x_test 	& 	y_test
	sets for training, validation, and testing.
	"""


	# Build a neural network to serve as the algorithm.

	# Assign the NN a loss function, so it can determine it's error.

	# Create an optimization algorithm, so the NN can reduce it's error and learn.

	# Tweak & tune the NN and it's hyperparameters on a validation set.

	# Test it's performance on the test set for a final result.
