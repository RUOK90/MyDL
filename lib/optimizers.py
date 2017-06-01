import numpy
import theano

from theano import config
import theano.tensor as tensor

def basic(cost, params, x, mask, y) : 
	grads = theano.grad(cost, params)
	grads_shared = [theano.shared(param.get_value() * 0.) for param in params]
	
	grads_to_grads_shared = [(grad_shared, grad) for grad_shared, grad in zip(grads_shared, grads)]
	if mask == None : 
		cost_func = theano.function([x, y], cost, updates = grads_to_grads_shared)
	else : 
		cost_func = theano.function([x, mask, y], cost, updates = grads_to_grads_shared)

	lr = theano.tensor.scalar()
	params_update = [(param, param - lr * grad) for param, grad in zip(params, grads_shared)]

	update_func = theano.function([lr], [], updates = params_update)

	return cost_func, update_func, grads_shared

def numpy_floatX(data):
	return numpy.asarray(data, dtype=config.floatX)

def adadelta(cost, params, x, mask, y):
	"""
	An adaptive learning rate optimizer

	Parameters
	----------
	lr : Theano SharedVariable
		Initial learning rate
	tpramas: Theano SharedVariable
		Model parameters
	grads: Theano variable
		Gradients of cost w.r.t to parameres
	x: Theano variable
		Model inputs
	mask: Theano variable
		Sequence mask
	y: Theano variable
		Targets
	cost: Theano variable
		Objective fucntion to minimize

	Notes
	-----
	For more information, see [ADADELTA]_.

	.. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
	   Rate Method*, arXiv:1212.5701.
	"""

	grads = theano.grad(cost, params)
	zipped_grads = [theano.shared(param.get_value() * 0.) for param in params]
	running_up2 = [theano.shared(param.get_value() * 0.) for param in params]
	running_grads2 = [theano.shared(param.get_value() * 0.) for param in params]

	zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
	rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]

	f_grad_shared = theano.function([x, mask, y], cost, updates=zgup + rg2up, name='adadelta_f_grad_shared')

	updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
		for zg, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)]
	ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2)) for ru2, ud in zip(running_up2, updir)]
	param_up = [(p, p + ud) for p, ud in zip(params, updir)]

	lr = theano.tensor.scalar()
	f_update = theano.function([lr], [], updates=ru2up + param_up, on_unused_input='ignore', name='adadelta_f_update')

	return f_grad_shared, f_update, None


def rmsprop(lr, tparams, grads, x, mask, y, cost):
	"""
	A variant of  SGD that scales the step size by running average of the
	recent step norms.

	Parameters
	----------
	lr : Theano SharedVariable
		Initial learning rate
	tpramas: Theano SharedVariable
		Model parameters
	grads: Theano variable
		Gradients of cost w.r.t to parameres
	x: Theano variable
		Model inputs
	mask: Theano variable
		Sequence mask
	y: Theano variable
		Targets
	cost: Theano variable
		Objective fucntion to minimize

	Notes
	-----
	For more information, see [Hint2014]_.

	.. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
	   lecture 6a,
	   http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
	"""

	zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
								  name='%s_grad' % k)
					for k, p in tparams.items()]
	running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
								   name='%s_rgrad' % k)
					 for k, p in tparams.items()]
	running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
									name='%s_rgrad2' % k)
					  for k, p in tparams.items()]

	zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
	rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
	rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
			 for rg2, g in zip(running_grads2, grads)]

	f_grad_shared = theano.function([x, mask, y], cost,
									updates=zgup + rgup + rg2up,
									name='rmsprop_f_grad_shared')

	updir = [theano.shared(p.get_value() * numpy_floatX(0.),
						   name='%s_updir' % k)
			 for k, p in tparams.items()]
	updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
				 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
											running_grads2)]
	param_up = [(p, p + udn[1])
				for p, udn in zip(tparams.values(), updir_new)]
	f_update = theano.function([lr], [], updates=updir_new + param_up,
							   on_unused_input='ignore',
							   name='rmsprop_f_update')

	return f_grad_shared, f_update
