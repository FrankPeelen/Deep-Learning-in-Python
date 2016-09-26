import pandas
import numpy as np
import cPickle as pickle
from neuralnet import *

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

	"""
	Test to see if the data looks like we'd expect.
	print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
	Result: ((50000, 3072), (50000,), (10000, 3072), (10000,))
	"""
	m = x_train.shape[0] # Number of training examples.
	n = x_train.shape[1] # Number of features

	"""
	Fix and correct missing data, if present.
	print(x_train.min(), x_train.max())
	print(y_train.min(), y_train.max())
	print(x_test.min(), x_test.max())
	print(y_test.min(), y_test.max())
	Result:
	(0, 255)
	(0, 9)
	(0, 255)
	(0, 9)
	The data looks fine. No crazy outliers. All values are inbetween 0 and 255, and all labels between 0 and 9.
	"""
	num_classes = y_train.max() + 1 # Number of classes, as long as the first class has the value 0, and not 1.

	# Preprocess the data --> Make the features have zero-mean.
	mean = x_train.mean(axis=0)
	x_train = x_train - mean
	x_test = x_test - mean

	"""
	See if the data has zero-mean now.
	print(x_train.min(), x_train.max())
	print(x_test.min(), x_test.max())
	Result:
	(-140.26882000000001, 155.00494)
	(-140.26882000000001, 155.00494)
	"""

	# Build a neural network to serve as the algorithm.

	"""
	Decide on the neural net's structure:
	I will start simplish with:
	500 Nodes -> ReLu -> 10 Nodes -> SVM Classifier
	"""
	layer1_size = 500
	layer2_size = 10	

	# Initialize net's weights
	init_const = 0.01 # Constant used for initializing the network's weights
	w1 = init_const * np.random.randn(n + 1,layer1_size)
	w2 = init_const * np.random.randn(layer1_size + 1,layer2_size)

	# Create a training loop with a certain batch size and chose for how many epochs you want to train
	batch_size_const = 100 # I chose a batch size of 100.
	num_epochs = 5 # I chose to go through all training examples 5 times

	while (num_epochs > 0):
		examples_left = m - batch_size_const # Minus batch_size, because we have to be able to pick a new batch without running out of examples.
		while (examples_left >= 0):
			print(examples_left, m - examples_left - batch_size_const, m - examples_left)
			batch = x_train[m - examples_left - batch_size_const : m - examples_left, :]
			examples_left = -1#-batch_size_const
		num_epochs += -1


	# Create a loss function, so that the NN can determine it's error

	# Choose a optimization algorithm


	# Tweak & tune the NN and it's hyperparameters on a validation set.

	# Test it's performance on the test set for a final result.
	scores = predict(x_test, [w1, w2])
	print(accuracy(scores, y_test))


# Unpickle a file.
# Returns a dictionairy with the file's contents.
def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo)
    fo.close()
    return dict

def svm_loss(scores, y):
	correct_class_score = scores[np.arange(m), y]

	for i in xrange(num_classes):
		i_eqs_y = i == y
		scores_images_of_class_i = i_eqs_y * scores

  	
	"""
  	for j in xrange(num_classes):
    	w = j != y
    	margin = w * (scores[:, j] - correct_class_score + 1) # note delta = 1
    	pos_margin = margin > 0
    	loss += np.sum(pos_margin * margin)
    	dW[:,j] += np.sum((pos_margin[0] * X * w[0]), axis=0)
    	new = np.zeros((500,10))
    	new[np.arange(num_train), y] += 1
    	dW += np.transpose(pos_margin[0] * X * w[0]).dot(new) * -1
    """
    

  	# Right now the loss is a sum over all training examples, but we want it
  	# to be an average instead so we divide by num_train.
  	loss /= m

  	# Add regularization to the loss.
  	loss += 0.5 * reg * np.sum(W * W)


	np.maximum()
	return np.maximum(+ 1, 0)

# Calculates the scores given input x into a neural network consisting of layers with weights w,
# and ReLu layers after each layer, except for the final layer.
# w has to be a list weights.
def predict(x, w):
	for i in xrange(len(w) - 1):
		x = np.insert(x, 0, 1, axis=1) # Adding the bias
		x = nodes_forward(x,w[i])
		x = relu_forward(x)
	x = np.insert(x, 0, 1, axis=1) # Adding the bias
	return nodes_forward(x,w[len(w) - 1])

# Calculates the accuracy of scores measured against labels y.
def accuracy(scores, y):
	predictions = np.argmax(scores, axis=1)
	num_correct_preds = np.sum(y == predictions)
	return num_correct_preds / (y.shape[0] * 1.0)

main()
