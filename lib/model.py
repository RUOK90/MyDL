import numpy
import theano

class NN(object) : 
	def __init__(self) : 
		self.params = []
		self.output = None
		self.reg = 0.
		self.cost = 0.
		self.err = 0.
		self.x = theano.tensor.matrix()
		self.y = theano.tensor.imatrix()
		self.dropout_switch = theano.shared(numpy.asarray(-1, dtype = theano.config.floatX))

	def layer(self, results, reg_weight_func = None, reg_weight_lambda = None, reg_bias_func = None, reg_bias_lambda = None) :
		self.output = results[0]

		for weight in results[1] : 
			self.params.append(weight)
			if reg_weight_func != None : 
				self.reg += reg_weight_func(weight, reg_weight_lambda)

		for bias in results[2] : 
			self.params.append(bias)
			if reg_bias_func != None : 
				self.reg += reg_bias_func(bias, reg_bias_lambda)

	def activation(self, result) : 
		self.output = result

	def train_cost(self, cost_func) : 
		self.cost += cost_func(self.output, self.y)
	
	def test_err(self, err_func, threshold) : 
		self.err = err_func(self.output, self.y, threshold)

	def get_output(self) : 
		if self.output == None : 
			return self.x
		else : 
			return self.output
	
	def get_dropout_switch(self) : 
		return self.dropout_switch

	def get_params(self) : 
		return self.params
	
	def do_masking(self) : 
		return False

	def train_setting(self, do_valid, opt_func) :
		train_cost_func, train_update_func, grads_shared = opt_func(self.cost + self.reg, self.params, self.x, None, self.y)
		test_err_func = theano.function([self.x, self.y], self.err)

		if do_valid : 
			valid_err_func = theano.function([self.x, self.y], self.err)
		else : 
			valid_err_func = None
			
		return train_cost_func, train_update_func, valid_err_func, test_err_func, grads_shared

class RNN(NN) : 
	def __init__(self) :
		super(RNN, self).__init__()
		self.x = theano.tensor.imatrix()
		self.mask = theano.tensor.matrix()

	def get_mask(self) : 
		return self.mask

	def do_masking(self) : 
		return True

	def train_setting(self, do_valid, opt_func) :
		train_cost_func, train_update_func, grads_shared = opt_func(self.cost + self.reg, self.params, self.x, self.mask, self.y)
		test_err_func = theano.function([self.x, self.mask, self.y], self.err)

		if do_valid : 
			valid_err_func = theano.function([self.x, self.mask, self.y], self.err)
		else : 
			valid_err_func = None
			
		return train_cost_func, train_update_func, valid_err_func, test_err_func, grads_shared
