import os
import tensorflow as tf
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', 'data/', 'Directory for storing data')
flags.DEFINE_string('checkpoint_dir', 'results/', 'Directory for storing results')

def linear(input, output_dim, wd, scope = None, bias=0.0):
	with tf.variable_scope(scope or 'linear'):
		weights = tf.get_variable("weights", [input.get_shape()[1], output_dim], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
		bias = tf.get_variable("biases", [output_dim], initializer=tf.constant_initializer(bias))
		
		variable_summaries(weights, (scope or 'linear')+'_weights')
		if wd is not None:
			weight_decay = tf.mul(tf.nn.l2_loss(weights), wd, name='weight_loss')
			tf.add_to_collection('losses', weight_decay)
		
		return tf.matmul(input, weights) + bias
		
def conv2d(input, kernel_size, wd, scope = None, bias=0.0):
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
		conv = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='SAME')
		
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

class DataDistribution:
	def __init__(self, dataset):
		self.train_images = dataset["inputs_train"]
		self.train_labels = dataset["targets_train"]
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
	def __init__(self, dataset, batchSize=128, trainingIterations = 1000000, learningRate=0.0003, weight_decay = 0.0):
		self.batchSize = batchSize
		self.input_size = [None, 128, 128, 3]
		self.kernel_1 = [5, 5, 3, 64]
		self.kernel_2 = [5, 5, 64, 128]
		self.linear_hidden_size = 1024
		self.output_labels = 8
		self.trainingIterations = trainingIterations
		self.learningRate = learningRate
		self.weight_decay = weight_decay
		self.dataDistribution = DataDistribution(dataset)
		
		self.createModel()
		self.loss()
		self.train_step()
	
	def createModel(self):
		self.input = tf.placeholder(tf.float32, self.input_size)
	
		self.conv1 = tf.nn.relu(conv2d(self.input, self.kernel_1, self.weight_decay, scope="conv1"))
		self.pool1 = max_pool(self.conv1, scope="max_pool1")
		
		self.conv2 = tf.nn.relu(conv2d(self.pool1, self.kernel_2, self.weight_decay, scope="conv2"))
		self.pool2 = max_pool(self.conv2, scope="max_pool1")
		
		# Need to define dimensionality of pool2 array
		# cut 128*128 by 2 and by 2 again * self.kernel_2[3]
		reshape_size = 32*32*128
		self.pool2_flat = tf.reshape(self.pool2, [-1,reshape_size])
		self.fc1 = tf.nn.relu(linear(self.pool2_flat, self.linear_hidden_size, self.weight_decay, scope="fc1"))
		
		self.fc2 = tf.nn.relu(linear(self.fc1, self.linear_hidden_size, self.weight_decay, scope="fc2"))
		
		self.output = linear(self.fc2, self.output_labels, self.weight_decay, scope="softmax")
		
	def loss(self):
		self.labels = tf.placeholder(tf.float32, [None, self.output_labels])
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(self.output, self.labels)
		cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
		tf.add_to_collection('losses', cross_entropy_mean)
		
		self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
		tf.scalar_summary(self.loss.op.name, self.loss)

	def train_step(self):
		self.optimizer = optimizer(self.loss, self.learningRate, tf.trainable_variables())
		
	def train(self, restore=False):
		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)) as sess:
			summary = tf.merge_all_summaries()
			summary_writer = tf.train.SummaryWriter(FLAGS.checkpoint_dir, sess.graph)
			tf.initialize_all_variables().run()
			saver = tf.train.Saver(max_to_keep=2)
			lastestLoss = -1
			if restore:
				saver.restore(sess,restore)
			#print("All vars")
			#for v in tf.trainable_variables():
			#	print(v.name)
			for i in range(self.trainingIterations):
				data_in, data_labels = self.dataDistribution.sample(self.batchSize)
				_, lastestLoss = sess.run([self.optimizer,self.loss], feed_dict={self.input:data_in, self.labels:data_labels})
				
				print("Running step: {}".format(i))
				print("Loss results: {}".format(lastestLoss))
				summary_str = sess.run(summary, feed_dict={self.input:data_in, self.labels:data_labels})
				summary_writer.add_summary(summary_str, i)
				summary_writer.flush()
				
				if i % 25 == 0 or (i+1) == self.trainingIterations:
					checkpoint_file = os.path.join(FLAGS.checkpoint_dir, 'checkpoint')
					saver.save(sess, checkpoint_file, global_step=i)