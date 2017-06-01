import numpy
import theano

def L2(reg_param, reg_lambda) :
	return (reg_param ** 2).sum() * reg_lambda
