import numpy
import theano
import time
import six.moves.cPickle as pickle

from preprocess import *

def train(model, n_epochs, dataset, batch_size, do_shuffle, maxlen, pad_dir, lr, save_to, 
		train_cost_func, train_update_func, valid_err_func, test_err_func, grads_shared) : 

	train_set_x = dataset[0]
	train_set_y = dataset[1]
	test_set_x = dataset[4]
	test_set_y = dataset[5]

	n_train = len(train_set_x)
	n_test = len(test_set_x)
	train_idxs = numpy.arange(n_train)
	test_idxs = numpy.arange(n_test)

	epoch_test_err = 0
	epoch_best_test_err = numpy.inf 
	epoch_prev_test_err = 0
	epoch = 0

	if not do_shuffle : 
		proc_train_set_x, train_masks, proc_train_set_y = get_processed_samples(train_set_x, train_set_y, train_idxs, 
				batch_size, model.do_masking(), maxlen, pad_dir)
		n_train_batches = len(proc_train_set_x)
		train_batch_idxs = numpy.arange(n_train_batches)

	proc_test_set_x, test_masks, proc_test_set_y = get_processed_samples(test_set_x, test_set_y, test_idxs, 
			batch_size, model.do_masking(), maxlen, pad_dir)
	n_test_batches = len(proc_test_set_x)
	test_batch_idxs = numpy.arange(n_test_batches)

	train_start_time = time.time()
	for epoch in range(n_epochs) : 
		epoch_start_time = time.time()

		if do_shuffle : 
			numpy.random.shuffle(train_idxs)
			proc_train_set_x, train_masks, proc_train_set_y = get_processed_samples(train_set_x, train_set_y, train_idxs, 
					batch_size, model.do_masking(), maxlen, pad_dir)
			n_train_batches = len(proc_train_set_x)
			train_batch_idxs = numpy.arange(n_train_batches)
			numpy.random.shuffle(train_batch_idxs)

		epoch_train_costs = []
		model.get_dropout_switch().set_value(1.)
		for i in range(n_train_batches) : 
			batch_start_time = time.time()
			idx = train_batch_idxs[i]
			if model.do_masking() : 
				batch_train_costs = train_cost_func(proc_train_set_x[idx], train_masks[idx], proc_train_set_y[idx])
			else : 
				batch_train_costs = train_cost_func(proc_train_set_x[idx], proc_train_set_y[idx])
			batch_train_cost = numpy.mean(batch_train_costs)
			epoch_train_costs.append(batch_train_cost)
			# print [grad.get_value() for grad in grads_shared]
			train_update_func(lr)

			batch_end_time = time.time()
			'''
			print 'batch %d/%d, train cost %f, %f secs' % \
				(i+1, n_train_batches, batch_train_cost, batch_end_time - batch_start_time) 
			'''

		epoch_test_errs = []
		model.get_dropout_switch().set_value(0.)
		for i in range(n_test_batches) : 
			idx = test_batch_idxs[i]
			if model.do_masking() : 
				batch_test_errs = test_err_func(proc_test_set_x[idx], test_masks[idx], proc_test_set_y[idx])
			else : 
				batch_test_errs = test_err_func(proc_test_set_x[idx], proc_test_set_y[idx])
			batch_test_err = numpy.mean(batch_test_errs)
			epoch_test_errs.append(batch_test_err)
		
		epoch_train_cost = numpy.mean(epoch_train_costs)
		epoch_test_err = numpy.mean(epoch_test_errs)

		# something about validation
		# if valid_err_func != None : 
 
		if epoch_test_err < epoch_best_test_err : 
			epoch_best_test_err = epoch_test_err
			if save_to != None : 
				# consider HDF5
				with open(save_to, 'wb') as f:
					pickle.dump(model, f)
		epoch_prev_test_err = epoch_test_err	

		epoch_end_time = time.time()
		print 'epoch %d/%d, train cost %f, test error %f%%, test error of the best model %f%%, %f secs' % \
			(epoch+1, n_epochs, epoch_train_cost, epoch_test_err, epoch_best_test_err, epoch_end_time - epoch_start_time)

	train_end_time = time.time()
	print 'training is completed with the best test error %f%%, %d epochs, %f secs' % \
		(epoch_best_test_err, epoch+1, train_end_time - train_start_time)
