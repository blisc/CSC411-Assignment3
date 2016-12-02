import os
import tensorflow as tf
import numpy as np
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from deeplearningmodels.imagenet_utils import preprocess_input
from deeplearningmodels.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', 'data/', 'Directory for storing data')
flags.DEFINE_string('checkpoint_dir', 'results/valid/', 'Directory for storing results')
flags.DEFINE_string('checkpoint_dir_1', 'results/train/', 'Directory for storing results')

def linear(input, output_dim, wd, scope = None, bias=0.0, input_shape=None):
	with tf.variable_scope(scope or 'linear'):
		if input_shape is None:
			weights = tf.get_variable("weights", [input.get_shape()[1], output_dim], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
		else:
			weights = tf.get_variable("weights", [input_shape, output_dim], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
		bias = tf.get_variable("biases", [output_dim], initializer=tf.constant_initializer(bias))
		
		variable_summaries(weights, (scope or 'linear')+'_weights')
		if wd is not None:
			weight_decay = tf.mul(tf.nn.l2_loss(weights), wd, name='weight_loss')
			tf.add_to_collection('losses', weight_decay)
		
		return tf.matmul(input, weights) + bias
		
def conv2d(input, kernel_size, wd, scope = None, bias=0.0, stride = [1, 1, 1, 1]):
	"""
	returns a convolution layers
	input (tensor): data going into the convolution layer
	kernel_size (array): 4-D array specifying size of kernel eg [x_filter_size, y_filter_size, input_channels, output_channels]
	scope (str): name of layer
	bias (float): initial value of bias terms
	Stride length should also be a parameter here
	"""
	with tf.variable_scope(scope or 'conv'):
		weights = tf.get_variable("weights", kernel_size, initializer=tf.contrib.layers.xavier_initializer(uniform=False))
		bias = tf.get_variable("biases", [kernel_size[3]], initializer=tf.constant_initializer(bias))
		conv = tf.nn.conv2d(input, weights, strides=stride, padding='SAME')
		
		variable_summaries(weights, (scope or 'conv2d')+'_weights')
		if wd is not None:
			weight_decay = tf.mul(tf.nn.l2_loss(weights), wd, name='weight_loss')
			tf.add_to_collection('losses', weight_decay)
		
		return tf.nn.bias_add(conv, bias)

def max_pool(input, kernel_size=[1, 2, 2, 1], scope = None, bias=0.0):
	"""
	Maybe stride length should also be a parameter :/
	"""
	return tf.nn.max_pool(input, kernel_size, strides=[1, 2, 2, 1], padding='SAME')
		
def optimizer(loss, learningRate, var_list):
	optimizer = tf.train.AdamOptimizer(learningRate).minimize(loss, var_list = var_list)
	return optimizer

def variable_summaries(var, name):
	"""Attach a lot of summaries to a Tensor."""
	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
		tf.scalar_summary('mean/' + name, mean)
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
		tf.scalar_summary('stddev/' + name, stddev)
		tf.scalar_summary('max/' + name, tf.reduce_max(var))
		tf.scalar_summary('min/' + name, tf.reduce_min(var))
		tf.histogram_summary(name, var)

def resize(arr):
	test = np.array([imresize(arr[0],(224,224),interp="lanczos")])
	for i,img in enumerate(arr):
		if i==0:
			continue
		test = np.concatenate((test,[imresize(img,(224,224),interp="lanczos")]))
	test = test.astype(np.float32)
		
class DataDistribution:
	def __init__(self, dataset, preprocessing):
		if preprocessing:
			# dataset["inputs_train"] = resize(dataset["inputs_train"])
			# dataset["inputs_val"]	= resize(dataset["inputs_val"])
			dataset["inputs_train"] = preprocess_input(dataset["inputs_train"])
			dataset["inputs_val"] = preprocess_input(dataset["inputs_val"])
		self.train_images = dataset["inputs_train"]
		self.train_labels = dataset["targets_train"]
		self.val_images = dataset["inputs_val"]
		self.val_labels = dataset["targets_val"]
		assert self.train_images.shape[0] == self.train_labels.shape[0], (
					'images.shape: %s labels.shape: %s' % (self.train_images.shape, self.train_labels.shape))

		self._num_examples = self.train_images.shape[0]
		self._index_in_epoch = 0
		self._num_examples = len(self.train_images)
		self._epochs_completed = 0
		
		#shuffle data
		perm = np.arange(self._num_examples)
		np.random.shuffle(perm)
		self.train_images = self.train_images[perm]
		self.train_labels = self.train_labels[perm]
		
		print( self.train_images.shape)
		print( self.train_labels.shape)
		print( self.val_images.shape)
		print( self.val_labels.shape)
	
	def sample(self, numSamples):
		"""
		Copied from tensorflow's mnist exmaple
		"""
		start = self._index_in_epoch
		self._index_in_epoch += numSamples
		if self._index_in_epoch > self._num_examples:
			# Finished epoch
			self._epochs_completed += 1
			# Shuffle the data
			perm = np.arange(self._num_examples)
			np.random.shuffle(perm)
			self.train_images = self.train_images[perm]
			self.train_labels = self.train_labels[perm]
			# Start next epoch
			start = 0
			self._index_in_epoch = numSamples
			assert numSamples <= self._num_examples
		end = self._index_in_epoch
		return self.train_images[start:end], self.train_labels[start:end]
		
class Model():
	def __init__(self, batchSize=128, trainingIterations = 54000, learningRate=0.0003, weight_decay = 0.0, kernel_1=[5, 5, 3, 64],kernel_2 = [5, 5, 64, 128],linear_hidden_size = 1024,keep_prob=0.5):
		self.batchSize = batchSize
		self.input_size = [None, 128, 128, 3]
		self.kernel_1 = kernel_1
		self.kernel_2 = kernel_2
		self.linear_hidden_size = linear_hidden_size
		self.output_labels = 8
		self.trainingIterations = trainingIterations
		self.learningRate = learningRate
		self.weight_decay = weight_decay
		self.preprocessing = False
	
	def createModelCNN4(self):
		print "creating model CNN4"
		self.input = tf.placeholder(tf.float32, self.input_size)
	
		self.conv1 = tf.nn.relu(conv2d(self.input, self.kernel_1, self.weight_decay, scope="conv1", stride=[1,2,2,1]))
		self.pool1 = max_pool(self.conv1, scope="max_pool1")
		
		self.conv2 = tf.nn.relu(conv2d(self.pool1, self.kernel_2, self.weight_decay, scope="conv2"))
		self.pool2 = max_pool(self.conv2, scope="max_pool2")
		
		# Need to define dimensionality of pool2 array
		# cut 128*128 by 2 and by 2 again * self.kernel_2[3]
		print(self.pool2.get_shape())
		reshape_size = int(self.pool2.get_shape()[1]) * int(self.pool2.get_shape()[2]) * int(self.pool2.get_shape()[3])
		self.pool2_flat = tf.reshape(self.pool2, [-1,reshape_size])
		self.fc1 = tf.nn.relu(linear(self.pool2_flat, self.linear_hidden_size, self.weight_decay, scope="fc1"))
		
		self.keep_prob = tf.placeholder(tf.float32)
		self.fc1_drop = tf.nn.dropout(self.fc1, self.keep_prob)
				
		self.output = linear(self.fc1_drop, self.output_labels, self.weight_decay, scope="softmax")
		
		self.softmax = tf.nn.softmax(self.output)
		
		self.labels = tf.placeholder(tf.float32, [None, self.output_labels])
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(self.output, self.labels)
		cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
		tf.add_to_collection('losses', cross_entropy_mean)
		
		self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
		tf.scalar_summary(self.loss.op.name, self.loss)

		self.correct_prediction = tf.equal(tf.argmax(self.labels, 1), tf.argmax(self.softmax, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
		tf.scalar_summary("Validation Accuracy", self.accuracy)
		
		self.optimizer = optimizer(self.loss, self.learningRate, tf.trainable_variables())
		
	def createModelFC1(self):
		print "creating model FC1"
		self.input_size = [None,512]
		self.linear_hidden_size = 1024
		self.input = tf.placeholder(tf.float32, self.input_size)
	
		self.fc1 = tf.nn.relu(linear(self.input, self.linear_hidden_size, self.weight_decay, scope="fc1"))
		self.fc2 = tf.nn.relu(linear(self.fc1, self.linear_hidden_size, self.weight_decay, scope="fc2"))
		self.fc3 = tf.nn.relu(linear(self.fc2, self.linear_hidden_size, self.weight_decay, scope="fc3"))
		
		self.output = linear(self.fc3_drop, self.output_labels, self.weight_decay, scope="softmax")
		
		self.softmax = tf.nn.softmax(self.output)
		
		self.labels = tf.placeholder(tf.float32, [None, self.output_labels])
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(self.output, self.labels)
		cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
		tf.add_to_collection('losses', cross_entropy_mean)
		
		self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
		tf.scalar_summary(self.loss.op.name, self.loss)

		self.correct_prediction = tf.equal(tf.argmax(self.labels, 1), tf.argmax(self.softmax, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
		tf.scalar_summary("Validation Accuracy", self.accuracy)
		
		self.optimizer = optimizer(self.loss, self.learningRate, tf.trainable_variables())
		
	def createModelFC2(self):
		print "creating model FC2"
		self.input_size = [None,512]
		self.input = tf.placeholder(tf.float32, self.input_size)
		self.keep_prob = tf.placeholder(tf.float32)
	
		self.fc1 = tf.nn.relu(linear(self.input, self.linear_hidden_size, self.weight_decay, scope="fc1"))
		self.fc1_drop = tf.nn.dropout(self.fc1, self.keep_prob)
		self.fc2 = tf.nn.relu(linear(self.fc1_drop, self.linear_hidden_size, self.weight_decay, scope="fc2"))
		self.fc2_drop = tf.nn.dropout(self.fc2, self.keep_prob)
		self.fc3 = tf.nn.relu(linear(self.fc2_drop, self.linear_hidden_size, self.weight_decay, scope="fc3"))
		self.fc3_drop = tf.nn.dropout(self.fc3, self.keep_prob)
		
		self.output = linear(self.fc3_drop, self.output_labels, self.weight_decay, scope="softmax")
		
		self.softmax = tf.nn.softmax(self.output)
		
		self.labels = tf.placeholder(tf.float32, [None, self.output_labels])
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(self.output, self.labels)
		cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
		tf.add_to_collection('losses', cross_entropy_mean)
		
		self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
		tf.scalar_summary(self.loss.op.name, self.loss)

		self.correct_prediction = tf.equal(tf.argmax(self.labels, 1), tf.argmax(self.softmax, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
		tf.scalar_summary("Validation Accuracy", self.accuracy)
		
		self.optimizer = optimizer(self.loss, self.learningRate, tf.trainable_variables())
		
	def createModelVGGTop(self):
		print "creating model VGGTop"
		self.input_size = [None,7*7*512]
		self.input = tf.placeholder(tf.float32, self.input_size)
		self.keep_prob = tf.placeholder(tf.float32)
				
		self.fc1 = tf.nn.relu(linear(self.input, self.linear_hidden_size, self.weight_decay, scope="fc1"))
		self.fc1_drop = tf.nn.dropout(self.fc1, self.keep_prob)
		self.fc2 = tf.nn.relu(linear(self.fc1_drop, self.linear_hidden_size, self.weight_decay, scope="fc2"))
		self.fc2_drop = tf.nn.dropout(self.fc2, self.keep_prob)
		self.fc3 = tf.nn.relu(linear(self.fc2_drop, self.output_labels, self.weight_decay, scope="fc3"))
		self.fc3_drop = tf.nn.dropout(self.fc3, self.keep_prob)
		
		self.output = linear(self.fc3_drop, self.output_labels, self.weight_decay, scope="softmax")
		
		self.softmax = tf.nn.softmax(self.output)
		
		self.labels = tf.placeholder(tf.float32, [None, self.output_labels])
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(self.output, self.labels)
		cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
		tf.add_to_collection('losses', cross_entropy_mean)
		
		self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
		tf.scalar_summary(self.loss.op.name, self.loss)

		self.correct_prediction = tf.equal(tf.argmax(self.labels, 1), tf.argmax(self.softmax, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
		tf.scalar_summary("Validation Accuracy", self.accuracy)
		
		self.optimizer = optimizer(self.loss, self.learningRate, tf.trainable_variables())
		
	def createModelCNN1(self):
		print "creating model CNN1"
		self.input = tf.placeholder(tf.float32, self.input_size)
	
		self.conv1 = tf.nn.relu(conv2d(self.input, self.kernel_1, self.weight_decay, scope="conv1", stride=[1,2,2,1]))
		self.pool1 = max_pool(self.conv1, scope="max_pool1")
		
		self.conv2 = tf.nn.relu(conv2d(self.pool1, self.kernel_2, self.weight_decay, scope="conv2"))
		self.pool2 = max_pool(self.conv2, scope="max_pool2")
		
		# Need to define dimensionality of pool2 array
		# cut 128*128 by 2 and by 2 again * self.kernel_2[3]
		print(self.pool2.get_shape())
		reshape_size = int(self.pool2.get_shape()[1]) * int(self.pool2.get_shape()[2]) * int(self.pool2.get_shape()[3])
		self.pool2_flat = tf.reshape(self.pool2, [-1,reshape_size])
		self.fc1 = tf.nn.relu(linear(self.pool2_flat, self.linear_hidden_size, self.weight_decay, scope="fc1"))
		
		self.fc2 = tf.nn.relu(linear(self.fc1, self.linear_hidden_size, self.weight_decay, scope="fc2"))
		
		self.output = linear(self.fc2, self.output_labels, self.weight_decay, scope="softmax")
		
		self.softmax = tf.nn.softmax(self.output)
		
		self.labels = tf.placeholder(tf.float32, [None, self.output_labels])
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(self.output, self.labels)
		cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
		tf.add_to_collection('losses', cross_entropy_mean)
		
		self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
		tf.scalar_summary(self.loss.op.name, self.loss)

		self.correct_prediction = tf.equal(tf.argmax(self.labels, 1), tf.argmax(self.softmax, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
		tf.scalar_summary("Validation Accuracy", self.accuracy)
		
		self.optimizer = optimizer(self.loss, self.learningRate, tf.trainable_variables())

	def createModelCNN2(self):
		print "creating model CNN2"
		self.input = tf.placeholder(tf.float32, self.input_size)
	
		self.conv1 = tf.nn.relu(conv2d(self.input, [10, 10, 3, 64], self.weight_decay, scope="conv1", stride=[1,2,2,1]))
		self.pool1 = max_pool(self.conv1, scope="max_pool1", kernel_size=[1,3,3,1])
		
		self.conv2 = tf.nn.relu(conv2d(self.pool1, [5, 5, 64, 192], self.weight_decay, scope="conv2"))
		self.pool2 = max_pool(self.conv2, scope="max_pool2")
		
		self.conv3 = tf.nn.relu(conv2d(self.pool2, [3, 3, 192, 256], self.weight_decay, scope="conv3"))
		self.conv4 = tf.nn.relu(conv2d(self.conv3, [3, 3, 256, 128], self.weight_decay, scope="conv4"))
		self.conv5 = tf.nn.relu(conv2d(self.conv4, [3, 3, 128, 128], self.weight_decay, scope="conv5"))
		self.pool5 = max_pool(self.conv5, scope="max_pool5")
		
		# Need to define dimensionality of pool2 array
		# cut 128*128 by 2 and by 2 again * self.kernel_2[3]
		reshape_size = 8*8*128
		self.pool5_flat = tf.reshape(self.pool5, [-1,reshape_size])
		self.fc1 = tf.nn.relu(linear(self.pool5_flat, self.linear_hidden_size, self.weight_decay, scope="fc1"))
		
		self.fc2 = tf.nn.relu(linear(self.fc1, self.linear_hidden_size, self.weight_decay, scope="fc2"))
		
		self.output = linear(self.fc2, self.output_labels, self.weight_decay, scope="softmax")
		
		self.softmax = tf.nn.softmax(self.output)
		
		self.labels = tf.placeholder(tf.float32, [None, self.output_labels])
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(self.output, self.labels)
		cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
		tf.add_to_collection('losses', cross_entropy_mean)
		
		self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
		tf.scalar_summary(self.loss.op.name, self.loss)
		
		self.correct_prediction = tf.equal(tf.argmax(self.labels, 1), tf.argmax(self.softmax, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
		tf.scalar_summary("Validation Accuracy", self.accuracy)


		self.optimizer = optimizer(self.loss, self.learningRate, tf.trainable_variables())
		
	def createModelCNN3(self):
		print "creating model CNN3"
		with tf.variable_scope("CNN3"):
			self.input = tf.placeholder(tf.float32, self.input_size)
		
			self.conv1 = tf.nn.relu(conv2d(self.input, [10, 10, 3, 64], self.weight_decay, scope="conv1", stride=[1,2,2,1]))
			self.pool1 = max_pool(self.conv1, scope="max_pool1")
			
			self.conv2 = tf.nn.relu(conv2d(self.pool1, [5, 5, 64, 192], self.weight_decay, scope="conv2"))
			self.pool2 = max_pool(self.conv2, scope="max_pool2")
			
			self.conv3 = tf.nn.relu(conv2d(self.pool2, [3, 3, 192, 256], self.weight_decay, scope="conv3"))
			self.conv4 = tf.nn.relu(conv2d(self.conv3, [3, 3, 256, 128], self.weight_decay, scope="conv4"))
			self.conv5 = tf.nn.relu(conv2d(self.conv4, [3, 3, 128, 128], self.weight_decay, scope="conv5"))
			self.pool5 = max_pool(self.conv5, scope="max_pool5")
			
			#print(self.pool5.get_shape())		
			
			# Need to define dimensionality of pool2 array
			# cut 128*128 by 2 and by 2 again * self.kernel_2[3]
			reshape_size = 8*8*128
			self.pool5_flat = tf.reshape(self.pool5, [-1,reshape_size])
			self.fc1 = tf.nn.relu(linear(self.pool5_flat, self.linear_hidden_size, self.weight_decay, scope="fc1"))
			
			self.fc2 = tf.nn.relu(linear(self.fc1, self.linear_hidden_size, self.weight_decay, scope="fc2"))
			
			self.output = linear(self.fc2, self.output_labels, self.weight_decay, scope="softmax")
			
			self.softmax = tf.nn.softmax(self.output)
			
		self.labels = tf.placeholder(tf.float32, [None, self.output_labels])
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(self.output, self.labels)
		cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
		tf.add_to_collection('losses', cross_entropy_mean)
		
		self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
		tf.scalar_summary(self.loss.op.name, self.loss)
		
		self.correct_prediction = tf.equal(tf.argmax(self.labels, 1), tf.argmax(self.softmax, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
		tf.scalar_summary("Validation Accuracy", self.accuracy)


		self.optimizer = optimizer(self.loss, self.learningRate, tf.trainable_variables())
		
	def VGG16Wrapper(self):
		self.preprocessing = True
		print "creating model VGG16"
		with tf.variable_scope("VGG16"):
			self.input = tf.placeholder(tf.float32, self.input_size)
			
			self.VGG = VGG16(include_top=False, input_tensor=self.input)
			
			# print(self.VGG.output.get_shape())
			# print(self.VGG.output.get_shape()[1])
			# print(self.VGG.output.get_shape()[2])
			# print(self.VGG.output.get_shape()[3])
			reshape_size = int(self.VGG.output.get_shape()[1]) * int(self.VGG.output.get_shape()[2]) * int(self.VGG.output.get_shape()[3])
			self.VGG_flat = tf.reshape(self.VGG.output, [-1,reshape_size])
			
			self.fc1 = tf.nn.relu(linear(self.VGG_flat, self.linear_hidden_size*4, self.weight_decay, scope="train/fc1"))
			self.fc2 = tf.nn.relu(linear(self.fc1, self.linear_hidden_size, self.weight_decay, scope="train/fc2"))
			self.output = linear(self.fc2, self.output_labels, self.weight_decay, scope="train/softmax")
			self.softmax = tf.nn.softmax(self.output)
			
		self.labels = tf.placeholder(tf.float32, [None, self.output_labels])
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(self.output, self.labels)
		cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
		tf.add_to_collection('losses', cross_entropy_mean)
		
		self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
		tf.scalar_summary(self.loss.op.name, self.loss)
		
		self.correct_prediction = tf.equal(tf.argmax(self.labels, 1), tf.argmax(self.softmax, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
		tf.scalar_summary("Validation Accuracy", self.accuracy)

		self.train_vars = [v for v in tf.trainable_variables() if v.name.startswith('VGG16/train/')]
		self.optimizer = optimizer(self.loss, self.learningRate, self.train_vars)
		print("All training vars")
		for v in self.train_vars:
			print(v.name)
			
	def VGG16Wrapper_2(self):
		self.preprocessing = True
		print "creating model VGG16_2"
		with tf.variable_scope("VGG16_2"):
			self.input = tf.placeholder(tf.float32, self.input_size)
			
			self.VGG = VGG16(include_top=False, input_tensor=self.input)
			
			# print(self.VGG.output.get_shape())
			# print(self.VGG.output.get_shape()[1])
			# print(self.VGG.output.get_shape()[2])
			# print(self.VGG.output.get_shape()[3])
			reshape_size = int(self.VGG.output.get_shape()[1]) * int(self.VGG.output.get_shape()[2]) * int(self.VGG.output.get_shape()[3])
			self.VGG_flat = tf.reshape(self.VGG.output, [-1,reshape_size])
			
			self.fc1 = tf.nn.relu(linear(self.VGG_flat, self.linear_hidden_size*4, self.weight_decay, scope="train/fc1"))
			self.fc2 = tf.nn.relu(linear(self.fc1, self.linear_hidden_size, self.weight_decay, scope="train/fc2"))
			self.output = linear(self.fc2, self.output_labels, self.weight_decay, scope="train/softmax")
			self.softmax = tf.nn.softmax(self.output)
			
		self.labels = tf.placeholder(tf.float32, [None, self.output_labels])
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(self.output, self.labels)
		cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
		tf.add_to_collection('losses', cross_entropy_mean)
		
		self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
		tf.scalar_summary(self.loss.op.name, self.loss)
		
		self.correct_prediction = tf.equal(tf.argmax(self.labels, 1), tf.argmax(self.softmax, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
		tf.scalar_summary("Validation Accuracy", self.accuracy)

		# self.train_vars = [v for v in tf.trainable_variables() if v.name.startswith('VGG16/train/')]
		self.optimizer = optimizer(self.loss, self.learningRate, tf.trainable_variables())
		# print("All training vars")
		# for v in self.train_vars:
			# print(v.name)
			
	def VGG16Wrapper_3(self):
		self.input_size = [None, 224, 224, 3]
		self.preprocessing = True
		print "creating model VGG16_3"
		with tf.variable_scope("VGG16"):
			self.input = tf.placeholder(tf.float32, self.input_size)
			
			self.VGG = VGG16(include_top=True, input_tensor=self.input)
			
			# print(self.VGG.output.get_shape())
			# print(self.VGG.output.get_shape()[1])
			# print(self.VGG.output.get_shape()[2])
			# print(self.VGG.output.get_shape()[3])
			#reshape_size = int(self.VGG.output.get_shape()[1]) * int(self.VGG.output.get_shape()[2]) * int(self.VGG.output.get_shape()[3])
			#self.VGG_flat = tf.reshape(self.VGG.output, [-1,reshape_size])
			
			self.fc1 = tf.nn.relu(linear(self.VGG.output, self.output_labels, self.weight_decay, scope="train/fc1"))
			self.output = linear(self.fc1, self.output_labels, self.weight_decay, scope="train/softmax")
			self.softmax = tf.nn.softmax(self.output)
			
		self.labels = tf.placeholder(tf.float32, [None, self.output_labels])
		self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(self.output, self.labels)
		self.cross_entropy_mean = tf.reduce_mean(self.cross_entropy, name='cross_entropy')
		tf.add_to_collection('losses', self.cross_entropy_mean)
		
		self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
		tf.scalar_summary(self.loss.op.name, self.loss)
		
		self.correct_prediction = tf.equal(tf.argmax(self.labels, 1), tf.argmax(self.softmax, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
		tf.scalar_summary("Validation Accuracy", self.accuracy)

		self.train_vars = [v for v in tf.trainable_variables() if v.name.startswith('VGG16/train/')]
		self.optimizer = optimizer(self.loss, self.learningRate, self.train_vars)
		print("All training vars")
		for v in self.train_vars:
			print(v.name)
			
	def VGG16Wrapper_4(self):
		self.input_size = [None, 224, 224, 3]
		self.preprocessing = True
		print "creating model VGG16"
		with tf.variable_scope("VGG16"):
			self.input = tf.placeholder(tf.float32, self.input_size)
			self.keep_prob = tf.placeholder(tf.float32)
			
			self.VGG = VGG16(include_top=False, input_tensor=self.input)
			
			# print(self.VGG.output.get_shape())
			# print(self.VGG.output.get_shape()[1])
			# print(self.VGG.output.get_shape()[2])
			# print(self.VGG.output.get_shape()[3])
			reshape_size = int(self.VGG.output.get_shape()[1]) * int(self.VGG.output.get_shape()[2]) * int(self.VGG.output.get_shape()[3])
			self.VGG_flat = tf.reshape(self.VGG.output, [-1,reshape_size])
	
		
			self.fc1 = tf.nn.relu(linear(self.VGG_flat, self.linear_hidden_size*4, self.weight_decay, scope="train/fc1"))
			self.fc1_drop = tf.nn.dropout(self.fc1, self.keep_prob)
			self.fc2 = tf.nn.relu(linear(self.fc1_drop, self.linear_hidden_size, self.weight_decay, scope="train/fc2"))
			self.fc2_drop = tf.nn.dropout(self.fc2, self.keep_prob)
			self.output = linear(self.fc2_drop, self.output_labels, self.weight_decay, scope="train/softmax")
			self.softmax = tf.nn.softmax(self.output)
			
		self.labels = tf.placeholder(tf.float32, [None, self.output_labels])
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(self.output, self.labels)
		cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
		tf.add_to_collection('losses', cross_entropy_mean)
		
		self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
		tf.scalar_summary(self.loss.op.name, self.loss)
		
		self.correct_prediction = tf.equal(tf.argmax(self.labels, 1), tf.argmax(self.softmax, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
		tf.scalar_summary("Validation Accuracy", self.accuracy)

		self.train_vars = [v for v in tf.trainable_variables() if v.name.startswith('VGG16/train/')]
		self.optimizer = optimizer(self.loss, self.learningRate, self.train_vars)
		print("All training vars")
		for v in self.train_vars:
			print(v.name)
	def VGG16_test(self):
		self.input_size = [None, 7*7*512]
		self.input = tf.placeholder(tf.float32, self.input_size)
		model = VGG16(include_top=True, my_input=self.input)
		return model
	
	def VGG16_extract(self):
		self.input_size = [None, 224, 224, 3]
		self.input = tf.placeholder(tf.float32, self.input_size)
		model = VGG16(include_top=False,input_tensor=self.input)
		self.output = model.output
		
		return model
	
	def extract_outputlayer(self, name, dataset, labels, type="train"):
		dataset = preprocess_input(dataset)
		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)) as sess:
			tf.initialize_all_variables().run()
			data_in = dataset
			results = sess.run(self.output, feed_dict={self.input:data_in})
			print results.shape
			print "Saving ", name
			if type == 'train':
				np.savez_compressed(name, inputs_train=results, targets_train=labels)
			elif type == 'test':
				np.savez_compressed(name, inputs_test=results)

	
	def VGG16_vanilla(self):
		self.input_size = [None, 224, 224, 3]
		self.input = tf.placeholder(tf.float32, self.input_size)
		model = VGG16(input_tensor=self.input)
		return model
		
		
	def train(self, dataset, restore=False, dropout=1.):
		self.dataDistribution = DataDistribution(dataset, self.preprocessing)
		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)) as sess:
			checkpoint_dir = 'results/VGGTop_Hidden_{}_WD_{}/valid/'.format(self.linear_hidden_size, self.weight_decay)
			checkpoint_dir_1 = 'results/VGGTop_Hidden_{}_WD_{}/train/'.format(self.linear_hidden_size, self.weight_decay)
			summary = tf.merge_all_summaries()
			summary_writer = tf.train.SummaryWriter(checkpoint_dir, sess.graph)
			summary_writer_1 = tf.train.SummaryWriter(checkpoint_dir_1, sess.graph)
			tf.initialize_all_variables().run()
			saver = tf.train.Saver(max_to_keep=3)
			lastestLoss = -1
			start_step = 0
			if restore:
				saver.restore(sess,restore)
				start_step = int(restore.split("-")[1])
				print("Restored :{}".format(restore))
			# print("All vars")
			# for v in tf.trainable_variables():
				# print(v.name)
			for i in range(start_step,self.trainingIterations):
				data_in, data_labels = self.dataDistribution.sample(self.batchSize)
				_, lastestLoss = sess.run([self.optimizer,self.loss], feed_dict={self.input:data_in, self.labels:data_labels, self.keep_prob:dropout})
				
				if i % 10 == 0:
					print("Running step: {}".format(i))
					print("Loss results: {}".format(lastestLoss))
					train_sum = sess.run(summary, feed_dict={self.input:data_in, self.labels:data_labels, self.keep_prob:1.})
					valid_sum = sess.run(summary, feed_dict={self.input:self.dataDistribution.val_images, self.labels:self.dataDistribution.val_labels, self.keep_prob:1.})
					summary_writer_1.add_summary(train_sum, i)
					summary_writer.add_summary(valid_sum, i)
					summary_writer.flush()
					summary_writer_1.flush()
				
				if i % 540 == 0:
					checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint')
					saver.save(sess, checkpoint_file, global_step=i)
					
	def train_datagen(self, dataset, restore=False):
		self.dataDistribution = DataDistribution(dataset, self.preprocessing)
		self.gen = ImageDataGenerator(
					featurewise_center=True,
					featurewise_std_normalization=True,
					rotation_range=20,
					width_shift_range=0.2,
					height_shift_range=0.2,
					shear_range=0.2,
					zoom_range=0.2,
					horizontal_flip=True)
		self.gen.fit(self.dataDistribution.train_images)
		
		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)) as sess:
			summary = tf.merge_all_summaries()
			summary_writer = tf.train.SummaryWriter(FLAGS.checkpoint_dir, sess.graph)
			summary_writer_1 = tf.train.SummaryWriter(FLAGS.checkpoint_dir_1, sess.graph)
			tf.initialize_all_variables().run()
			saver = tf.train.Saver(max_to_keep=3)
			lastestLoss = -1
			start_step = 0
			if restore:
				saver.restore(sess,restore)
				start_step = int(restore.split("-")[1])
				print("Restored :{}".format(restore))
			# print("All vars")
			# for v in tf.trainable_variables():
				# print(v.name)
			for i in range(start_step,self.trainingIterations):
				batches = 0
				data_in = []
				data_labels = []
				for X_batch, Y_batch in self.gen.flow(self.dataDistribution.train_images, self.dataDistribution.train_labels, batch_size=self.batchSize):
					_, lastestLoss = sess.run([self.optimizer,self.loss], feed_dict={self.input:X_batch, self.labels:Y_batch, self.keep_prob:0.5})
					
					print("Running step: {}.{}".format(i,batches))
					print("Loss results: {}".format(lastestLoss))
					data_in = X_batch
					data_labels = Y_batch
					batches += 1
					if batches >= len(self.dataDistribution.train_images) / self.batchSize:
						break
						
				train_sum = sess.run(summary, feed_dict={self.input:data_in, self.labels:data_labels, self.keep_prob:1.})
				valid_sum = sess.run(summary, feed_dict={self.input:self.dataDistribution.val_images, self.labels:self.dataDistribution.val_labels,self.keep_prob:1.})
				summary_writer_1.add_summary(train_sum, i)
				summary_writer.add_summary(valid_sum, i)
				summary_writer.flush()
				summary_writer_1.flush()
				
				if i % 10 == 0:
					checkpoint_file = os.path.join(FLAGS.checkpoint_dir, 'checkpoint')
					saver.save(sess, checkpoint_file, global_step=i)
					
	def inference(self, restoreFile, testData):
		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)) as sess:
			tf.initialize_all_variables().run()
			saver = tf.train.Saver()
			saver.restore(sess,restoreFile)
			print "successfully restored from {}".format(restoreFile)
			data_in = testData
			results = sess.run(self.softmax, feed_dict={self.input:data_in, self.keep_prob:1.0})
			labels = np.argmax(results, axis=1)
			np.savetxt("./results.txt", labels, delimiter='\n')
