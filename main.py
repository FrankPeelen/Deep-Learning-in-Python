import pandas
import numpy as np
import cPickle as pickle

def main():
	# Download, load, and prepare the CIFAR-10 data.

	# Download the CIFAR-10 data.
	# DONE. Located in /home/peel/Downloads/cifar10py/

	## Load the CIFAR-10 data into Python.

	# Unpickle the test data, and put them into x_test & y_test
	unpickled_test_data = unpickle('/home/peel/Downloads/cifar10py/test_batch')
	x_test = unpickled_test_data['data']
	y_test = unpickled_test_data['labels']
	y_test = np.asarray(y_test)

	# Unpickle all 5 data files to be used for training.
	unpickled_data_1 = unpickle('/home/peel/Downloads/cifar10py/data_batch_1')
	unpickled_data_2 = unpickle('/home/peel/Downloads/cifar10py/data_batch_2')
	unpickled_data_3 = unpickle('/home/peel/Downloads/cifar10py/data_batch_3')
	unpickled_data_4 = unpickle('/home/peel/Downloads/cifar10py/data_batch_4')
	unpickled_data_5 = unpickle('/home/peel/Downloads/cifar10py/data_batch_5')
	# Put the unpickled data into x_train & y_train
	x_train = np.append(unpickled_data_1['data'], unpickled_data_2['data'], axis=0)
	x_train = np.append(x_train, unpickled_data_3['data'], axis=0)
	x_train = np.append(x_train, unpickled_data_4['data'], axis=0)
	x_train = np.append(x_train, unpickled_data_5['data'], axis=0)
	y_train = np.append(unpickled_data_1['labels'], unpickled_data_2['labels'])
	y_train = np.append(y_train, unpickled_data_3['labels'])
	y_train = np.append(y_train, unpickled_data_4['labels'])
	y_train = np.append(y_train, unpickled_data_5['labels'])

	# Optional test to check if everything went correctly.
	# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

	# Fix and correct missing data.
	# print(x_train.min(), x_train.max())
	# print(y_train.min(), y_train.max())
	# print(x_test.min(), x_test.max())
	# print(y_test.min(), y_test.max())
	# The data looks fine. No crazy outliers. All values are inbetween 0 and 255.

	# Preprocess the data --> Make the features have zero-mean.
	mean = x_train.mean(axis=0)
	x_train = x_train - mean
	x_test = x_test - mean

	# Build a neural network to serve as the algorithm.

	# Assign the NN a loss function, so it can determine it's error.

	# Create an optimization algorithm, so the NN can reduce it's error and learn.

	# Tweak & tune the NN and it's hyperparameters on a validation set.

	# Test it's performance on the test set for a final result.

# Unpickle a file.
# Returns a dictionairy with the file's contents.
def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo)
    fo.close()
    return dict

main()
