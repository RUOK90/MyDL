import numpy
import theano

def softmax(input) : 
	return theano.tensor.nnet.softmax(input)

def sigmoid(input) : 
	return theano.tensor.nnet.sigmoid(input)

def tanh(input) : 
	return theano.tensor.tanh(input)

def ReLU(input) : 
	return theano.tensor.nnet.relu(input)
