import numpy
import theano

def FeedForward(input, init_W, init_b) : 
	W = theano.shared(value = init_W)
	b = theano.shared(value = init_b)
	return theano.tensor.dot(input, W) + b, [W], [b]

def Embedding(input, init_W) : 
	W = theano.shared(value = init_W)
	return W[input], [W], []

def max_over_time(input) : 
	return input.max(axis = 0), [], []

def mean_over_time(input, mask) : 
	if input.ndim == 3 : 
		mask = mask.dimshuffle(1, 0, 'x')
	else :
		mask = mask.dimshuffle(0, 'x')

	return (input * mask).sum(axis = 0) / mask.sum(axis = 0), [], []

def min_over_time(input) : 
	return input.min(axis = 0), [], []

def dropout_layer(input, dropout_switch, theano_random, rate) :
	result = theano.tensor.switch(dropout_switch, 
			input * theano_random.binomial(input.shape, p = rate, n = 1, dtype = theano.config.floatX), input * rate)
	return result, [], []

def LSTM(input, mask, do_dimshuffle, init_Wf, init_Wi, init_Wc, init_Wo, init_Uf, init_Ui, init_Uc, init_Uo,
		init_bf, init_bi, init_bc, init_bo) : 

	Wf = theano.shared(value = init_Wf)
	Wi = theano.shared(value = init_Wi)
	Wc = theano.shared(value = init_Wc)
	Wo = theano.shared(value = init_Wo)

	Uf = theano.shared(value = init_Uf)
	Ui = theano.shared(value = init_Ui)
	Uc = theano.shared(value = init_Uc)
	Uo = theano.shared(value = init_Uo)

	bf = theano.shared(value = init_bf)
	bi = theano.shared(value = init_bi)
	bc = theano.shared(value = init_bc)
	bo = theano.shared(value = init_bo)

	W = theano.tensor.concatenate([Wf, Wi, Wc, Wo], axis = 1)
	U = theano.tensor.concatenate([Uf, Ui, Uc, Uo], axis = 1)
	b = theano.tensor.concatenate([bf, bi, bc, bo], axis = 0)

	if input.ndim == 3 :
		if do_dimshuffle : 
			input = input.dimshuffle(1, 0, 2)
		mask = mask.dimshuffle(1, 0, 'x')
		n_samples = input.shape[1]
	else:
		mask = mask.dimshuffle(0, 'x')
		n_samples = 1

	n_steps = input.shape[0]
	dim = W.shape[0]

	def slice(M, n, dim) :
		if M.ndim == 2 :
			return M[:, n*dim : (n+1)*dim]
		else : 
			return M[n*dim : (n+1)*dim]

	def step(Wx_t_b, m_t, h_tm1, c_tm1) :
		Wx_t_Uh_tm1_b = Wx_t_b + theano.tensor.dot(h_tm1, U)

		f_t = theano.tensor.nnet.sigmoid(slice(Wx_t_Uh_tm1_b, 0, dim))
		i_t = theano.tensor.nnet.sigmoid(slice(Wx_t_Uh_tm1_b, 1, dim))
		cc_t = theano.tensor.tanh(slice(Wx_t_Uh_tm1_b, 2, dim))
		o_t = theano.tensor.nnet.sigmoid(slice(Wx_t_Uh_tm1_b, 3, dim))

		c_t = f_t * c_tm1 + i_t * cc_t
		c_t = m_t * c_t + (1. - m_t) * c_tm1

		h_t = o_t * theano.tensor.tanh(c_t)
		h_t = m_t * h_t + (1. - m_t) * h_tm1

		return h_t, c_t

	Wx_b = theano.tensor.dot(input, W) + b
	zero = numpy.asarray(0, dtype = theano.config.floatX)
	h = theano.tensor.alloc(zero, n_samples, dim)
	c = theano.tensor.alloc(zero, n_samples, dim)
	results, updates = theano.scan(fn = step, sequences = [Wx_b, mask], outputs_info = [h, c], n_steps = n_steps)
	
	return results[0], [Wf, Wi, Wc, Wo, Uf, Ui, Uc, Uo], [bf, bi, bc, bo]

def GRU(input, mask, do_dimshuffle, init_Wz, init_Wr, init_Wh, init_Uz, init_Ur, init_Uh, init_bz, init_br, init_bh) : 

	Wz = theano.shared(value = init_Wz)
	Wr = theano.shared(value = init_Wr)
	Wh = theano.shared(value = init_Wh)

	Uz = theano.shared(value = init_Uz)
	Ur = theano.shared(value = init_Ur)
	Uh = theano.shared(value = init_Uh)

	bz = theano.shared(value = init_bz)
	br = theano.shared(value = init_br)
	bh = theano.shared(value = init_bh)

	W = theano.tensor.concatenate([Wz, Wr, Wh], axis = 1)
	Uzr = theano.tensor.concatenate([Uz, Ur], axis = 1)
	b = theano.tensor.concatenate([bz, br, bh], axis = 0)

	if input.ndim == 3 :
		if do_dimshuffle : 
			input = input.dimshuffle(1, 0, 2)
		mask = mask.dimshuffle(1, 0, 'x')
		n_samples = input.shape[1]
	else:
		mask = mask.dimshuffle(0, 'x')
		n_samples = 1

	n_steps = input.shape[0]
	dim = W.shape[0]

	def slice(M, n, dim) :
		if M.ndim == 2 :
			return M[:, n*dim : (n+1)*dim]
		else : 
			return M[n*dim : (n+1)*dim]

	def step(Wx_t_b, m_t, h_tm1) :
		Uzr_h_tm1 = theano.tensor.dot(h_tm1, Uzr)

		z_t = theano.tensor.nnet.sigmoid(slice(Wx_t_b, 0, dim) + slice(Uzr_h_tm1, 0, dim))
		r_t = theano.tensor.nnet.sigmoid(slice(Wx_t_b, 1, dim) + slice(Uzr_h_tm1, 1, dim))
		ch_t = theano.tensor.tanh(slice(Wx_t_b, 2, dim) + theano.tensor.dot(r_t * h_tm1, Uh))

		h_t = (1. - z_t) * h_tm1 + z_t * ch_t
		h_t = m_t * h_t + (1. - m_t) * h_tm1

		return h_t

	Wx_b = theano.tensor.dot(input, W) + b
	zero = numpy.asarray(0, dtype = theano.config.floatX)
	h = theano.tensor.alloc(zero, n_samples, dim)
	results, updates = theano.scan(fn = step, sequences = [Wx_b, mask], outputs_info = [h], n_steps = n_steps)
	
	return results, [Wz, Wr, Wh, Uz, Ur, Uh], [bz, br, bh]

def Vanilla_RNN(input, mask, do_dimshuffle, init_W, init_U, init_b) : 

	W = theano.shared(value = init_W)
	U = theano.shared(value = init_U)
	b = theano.shared(value = init_b)

	if input.ndim == 3 :
		if do_dimshuffle : 
			input = input.dimshuffle(1, 0, 2)
		mask = mask.dimshuffle(1, 0, 'x')
		n_samples = input.shape[1]
	else:
		mask = mask.dimshuffle(0, 'x')
		n_samples = 1

	n_steps = input.shape[0]
	dim = W.shape[0]

	def step(Wx_t_b, m_t, h_tm1) :
		h_t = theano.tensor.tanh(Wx_t_b + theano.tensor.dot(h_tm1, U))
		h_t = m_t * h_t + (1. - m_t) * h_tm1

		return h_t

	Wx_b = theano.tensor.dot(input, W) + b
	zero = numpy.asarray(0, dtype = theano.config.floatX)
	h = theano.tensor.alloc(zero, n_samples, dim)
	results, updates = theano.scan(fn = step, sequences = [Wx_b, mask], outputs_info = [h], n_steps = n_steps)
	
	return results, [W, U], [b]
