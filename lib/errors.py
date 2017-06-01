import numpy
import theano

def single_label_mis_rate(output, y, threshold) : 
	output_prediction = theano.tensor.argmax(output, axis = 1)
	y_answer = theano.tensor.argmax(y, axis = 1) 
	return theano.tensor.neq(output_prediction, y_answer).mean() * 100.

def multi_label_mis_rate(output, y, threshold) : 
	return theano.tensor.neq(output > threshold, y).mean() * 100.

