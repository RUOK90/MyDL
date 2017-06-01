import numpy
import theano

def categorical_cross_entropy(output, y) : 
	return theano.tensor.nnet.categorical_crossentropy(output, y).mean()

def binary_cross_entropy(output, y) : 
	return theano.tensor.nnet.binary_crossentropy(output, y).mean()
