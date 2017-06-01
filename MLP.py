import numpy
import theano

from lib.model import *
from lib.layers import *
from lib.initializers import *
from lib.activation import *
from lib.dataset import *
from lib.costs import *
from lib.errors import *
from lib.optimizers import *
from lib.regularizers import *
from lib.train import *

numpy.random.seed(1234)

def Train(n_epochs, from_dataset, batch_size, do_shuffle, lr, save_to) :
	print 'loading data'
	dataset = load_MNIST(from_dataset)

	print 'building the model'
	MLP = NN()
	low = -numpy.sqrt(6. / (784 + 500)) 
	high = numpy.sqrt(6. / (784 + 500))
	MLP.layer(FeedForward(MLP.get_output(), uniform(784, 500, low, high), zero(0, 500)), L2, 0.0001, None, None)
	MLP.activation(tanh(MLP.get_output()))
	MLP.layer(FeedForward(MLP.get_output(), zero(500, 10), zero(0, 10)), L2, 0.0001, None, None)
	MLP.activation(softmax(MLP.get_output()))

	print 'setting train options'
	MLP.train_cost(categorical_cross_entropy)
	MLP.test_err(single_label_mis_rate, None)
	train_cost_func, train_update_func, valid_err_func, test_err_func, grads_shared = MLP.train_setting(False, basic)
	
	print 'training the model'
	train(MLP, n_epochs, dataset, batch_size, do_shuffle, None, None, lr, save_to,
			train_cost_func, train_update_func, valid_err_func, test_err_func, grads_shared)

if __name__ == '__main__' : 
	Train(10, 'data/mnist.pkl', 20, True, 0.01, 'model/MLP_model.pkl')
