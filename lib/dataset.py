import numpy
import theano
import gzip
import six.moves.cPickle as pickle

def load_MNIST(path) :
	f = open(path, 'rb')
	train_set, valid_set, test_set = pickle.load(f)
	f.close()

	new_train_labels = []
	for label in train_set[1] : 
		candidate_label = [0 for i in range(10)]
		candidate_label[label] = 1
		new_train_labels.append(candidate_label)
		
	new_test_labels = []
	for label in test_set[1] : 
		candidate_label = [0 for i in range(10)]
		candidate_label[label] = 1
		new_test_labels.append(candidate_label)

	train_set_x = train_set[0]
	train_set_y = new_train_labels
	valid_set_x = valid_set[0]
	valid_set_y = valid_set[1]
	test_set_x = test_set[0]
	test_set_y = new_test_labels

	return train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y

def load_IMDB(path) :
	f = open(path, 'rb')
	train_set = pickle.load(f)
	test_set = pickle.load(f)
	f.close()

	maxlen = 100

	new_train_set_x = []
	new_train_set_y = []
	for x, y in zip(train_set[0], train_set[1]):
		if len(x) <= maxlen:
			new_train_set_x.append(x)
			new_train_set_y.append(y)
	
	train_set_x = new_train_set_x
	train_set_y = new_train_set_y

	new_test_set_x = []
	new_test_set_y = []
	for x, y in zip(test_set[0], test_set[1]):
		if len(x) <= maxlen:
			new_test_set_x.append(x)
			new_test_set_y.append(y)

	test_set_x = new_test_set_x
	test_set_y = new_test_set_y

	dic_size = 10000

	new_train_set_x = []
	new_train_set_y = []
	for x, y in zip(train_set_x, train_set_y) : 
		add = True
		for w in x : 
			if w > dic_size :
				add = False 
				break
		if add : 
			new_train_set_x.append(x)
			new_train_set_y.append(y)

	train_set_x = new_train_set_x
	train_set_y = new_train_set_y

	new_test_set_x = []
	new_test_set_y = []
	for x, y in zip(test_set_x, test_set_y) : 
		add = True
		for w in x : 
			if w > dic_size :
				add = False 
				break
		if add : 
			new_test_set_x.append(x)
			new_test_set_y.append(y)

	test_set_x = new_test_set_x
	test_set_y = new_test_set_y

	new_train_set_y = []
	for y in train_set_y : 
		new_y = [0 for i in range(2)]
		new_y[y] = 1
		new_train_set_y.append(new_y)
		
	new_test_set_y = []
	for y in test_set_y : 
		new_y = [0 for i in range(2)]
		new_y[y] = 1
		new_test_set_y.append(new_y)

	train_set_y = new_train_set_y
	test_set_y = new_test_set_y

	return train_set_x, train_set_y, None, None, test_set_x, test_set_y

def load_XOR() : 
	train_set_x = [[1, 1], [1, 0], [0, 1], [0, 0]]
	train_set_y = [[0], [1], [1], [0]]

	test_set_x = [[1, 1], [1, 0], [0, 1], [0, 0]]
	test_set_y = [[0], [1], [1], [0]]

	return train_set_x, train_set_y, None, None, test_set_x, test_set_y

def load_HEART(path) : 
	f = file(path, 'r')

	train_set_x = []
	train_set_y = []


