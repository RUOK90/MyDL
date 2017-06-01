import numpy
import theano
import theano.sandbox.rng_mrg

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
theano_random = theano.sandbox.rng_mrg.MRG_RandomStreams(1234)

def Train(n_epochs, from_dataset, batch_size, do_shuffle, lr, save_to) :

	maxlen = None
	pad_dir = 'right'
	dic_size = 10001 # dic_size is the number of words + 1 (padding)

	print 'loading data'
	dataset = load_IMDB(from_dataset)

	print 'building the model'
	V_RNN = RNN()
	V_RNN.layer(Embedding(V_RNN.get_output(), normal(dic_size, 128, 0, 0.01)), None, None, None, None)
	V_RNN.layer(Vanilla_RNN(V_RNN.get_output(), V_RNN.get_mask(), True,
				orthogonal(128, 0, 0.01), orthogonal(128, 0, 0.01), zero(0, 128)), None, None, None, None)
	V_RNN.layer(mean_over_time(V_RNN.get_output(), V_RNN.get_mask()))
	V_RNN.layer(dropout_layer(V_RNN.get_output(), V_RNN.get_dropout_switch(), theano_random, 0.5))
	V_RNN.layer(FeedForward(V_RNN.get_output(), normal(128, 2, 0, 0.01), zero(0, 2)), None, None, None, None)
	V_RNN.activation(softmax(V_RNN.get_output()))

	print 'setting train options'
	V_RNN.train_cost(categorical_cross_entropy)
	V_RNN.test_err(single_label_mis_rate, None)
	train_cost_func, train_update_func, valid_err_func, test_err_func, grads_shared = V_RNN.train_setting(False, adadelta)
	
	print 'training the model'
	train(V_RNN, n_epochs, dataset, batch_size, do_shuffle, maxlen, pad_dir, lr, save_to,
			train_cost_func, train_update_func, valid_err_func, test_err_func, grads_shared)

if __name__ == '__main__' : 
	Train(100, 'data/imdb.pkl', 20, True, 0.0001, 'model/RNN_model.pkl')
