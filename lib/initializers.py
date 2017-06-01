import numpy
import theano

def zero(n_in, n_out) : 
	if n_in == 0 : 
		return numpy.zeros((n_out), dtype = theano.config.floatX)
	else : 
		return numpy.zeros((n_in, n_out), dtype = theano.config.floatX)

def uniform(n_in, n_out, low, high) : 
	if n_in == 0 : 
		return numpy.random.uniform(low, high, (n_out)).astype(theano.config.floatX)
	else : 
		return numpy.random.uniform(low, high, (n_in, n_out)).astype(theano.config.floatX)

def normal(n_in, n_out, mean, s_dev) : 
	if n_in == 0 : 
		return numpy.random.normal(mean, s_dev, (n_out)).astype(theano.config.floatX)
	else : 
		return numpy.random.normal(mean, s_dev, (n_in, n_out)).astype(theano.config.floatX) 

def orthogonal(n_dim, mean, s_dev) : 
	m = numpy.random.normal(mean, s_dev, (n_dim, n_dim))
	u, s, v = numpy.linalg.svd(m)
	return u.astype(theano.config.floatX)
