import numpy
import theano

def get_batch_idxs(n_samples, batch_size) : 
	n_batches = n_samples / batch_size
	batch_idxs = [(i * batch_size, (i+1) * batch_size) for i in range(n_batches)]

	if n_samples % batch_size != 0 : 
		batch_idxs.append((n_batches * batch_size, n_samples))

	return batch_idxs

def get_sorted_by_len_idxs(xs, idxs) : 
	return sorted(idxs, key = lambda idx : len(xs[idx]))

def get_sorted_by_len_samples(xs, ys, idxs, maxlen) : 
	sorted_idxs = get_sorted_by_len_idxs(xs, idxs)
	sorted_xs = []
	sorted_ys = []

	for idx in sorted_idxs : 
		if maxlen != None and len(xs[idx]) > maxlen :  
			break
		sorted_xs.append(xs[idx])
		sorted_ys.append(ys[idx])
	
	return sorted_xs, sorted_ys

def get_batch_samples(xs, ys, batch_size) : 
	n_samples = len(xs)
	n_batches = n_samples / batch_size
	batch_xs = [[x for x in xs[i * batch_size : (i+1) * batch_size]] for i in range(n_batches)]
	batch_ys = [[y for y in ys[i * batch_size : (i+1) * batch_size]] for i in range(n_batches)]

	if n_samples % batch_size != 0 : 
		batch_xs.append([x for x in xs[n_batches * batch_size : n_samples]])
		batch_ys.append([y for y in ys[n_batches * batch_size : n_samples]])

	return batch_xs, batch_ys

def get_padded_samples(xs, pad_dir) : 
	padded_xs = []
	masks = []

	for batch in xs :
		lens = [len(x) for x in batch]
		maxlen = numpy.max(lens)
		n_samples = len(batch)
		padded_batch = [[0 for i in range(maxlen)] for j in range(n_samples)]
		batch_masks = [[0 for i in range(maxlen)] for j in range(n_samples)]
		
		for idx, x in enumerate(batch):
			if pad_dir == 'right' : 
				padded_batch[idx][ : lens[idx]] = x
				x_1s = [1 for i in range(len(x))]
				batch_masks[idx][ : lens[idx]] = x_1s
			elif pad_dir == 'left' : 
				padded_batch[idx][maxlen - lens[idx] : ] = x
				x_1s = [1 for i in range(len(x))]
				batch_masks[idx][maxlen - lens[idx] : ] = x_1s
		
		padded_xs.append(padded_batch)
		masks.append(batch_masks)

	return padded_xs, masks

def get_one_hot_samples(xs, dic_size) : 
	one_hot_xs = []
	for batch in xs : 
		one_hot_batch = []
		for x in batch : 
			one_hot_x = []
			for w in x : 
				one_hot_w = [0 for i in range(dic_size)]
				one_hot_w[w] = 1
				one_hot_x.append(one_hot_w)
			one_hot_batch.append(one_hot_x)
		one_hot_xs.append(one_hot_batch)

	return one_hot_xs

def get_processed_samples(xs, ys, idxs, batch_size, do_masking, maxlen, pad_dir) : 
	sorted_xs, sorted_ys = get_sorted_by_len_samples(xs, ys, idxs, maxlen)
	batch_xs, batch_ys = get_batch_samples(sorted_xs, sorted_ys, batch_size)
	proc_set_ys = [numpy.asarray(batch, dtype = 'int32') for batch in batch_ys]

	if do_masking :
		padded_xs, masks = get_padded_samples(batch_xs, pad_dir)
		proc_set_xs = [numpy.asarray(batch, dtype = 'int32') for batch in padded_xs]
		masks = [numpy.asarray(batch, dtype = theano.config.floatX) for batch in masks]
	else :
		proc_set_xs = [numpy.asarray(batch, dtype = theano.config.floatX) for batch in batch_xs]
		masks = None

	return proc_set_xs, masks, proc_set_ys










