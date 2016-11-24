import os, csv, sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
plt.ion()


#=========================================================================
# Class for processing images and their labels
#=========================================================================

class DataTable:
	def __init__(self, imgPath, targetPath, dataType):
		# imgPath: path to image directory
		# targetPath: path to csv file containing labels
		self.images = imgPath + '/'
		self.targets = targetPath
		# Make sure there are only jpegs in this directory
		self.files = np.array(os.listdir(self.images))

		# Type of data set being processed
		assert dataType == 'train' or dataType == 'test'
		self._data = dataType

		# [num_samples, height, width, channels]
		self._inputs = np.array([])
		# [num_samples, 1]
		self._labels = np.array([])

	def processImages(self):
		# We only have labels for the training set
		if self._data == 'train':
			self._readLabels()

		# Hard code set sizes
		if self._data == 'train':
			subsets = [range(1,1001), range(1001,2001), range(2001,3001), range(3001,4001), \
					   range(4001,5001), range(5001,6001), range(6001,7001)]
		elif self._data == 'test':
			subsets = [range(1,self.files.shape[0]+1)]
		else:
			print("Data type not supported.")
			sys.exit(1)

		subsets = np.array(subsets)
		for subset in subsets:
			start = np.min(subset)
			end = np.max(subset)
			# "-1" for indexing
			self._processImagesSubset(images = subset-1)
			if self._data == 'train':
				name = 'train_' + str(start) + '_' + str(end)
			elif self._data == 'test':
				name = 'test_' + str(start) + '_' + str(end)
			else:
				print("Data type not supported.")
				sys.exit(1)

			print name
			if self._data == 'train':
				labels = self._labels[subset]
				self._saveData(name, labels)
			elif self._data == 'test':
				self._saveData(name)
			else:
				print("Data type not supported.")
				sys.exit(1)

	def _readLabels(self):
		# Reads labels file for the training data
		# [ [number, label], [number, label], ... , [number, label] ]
		try:
			self._labels = np.genfromtxt(self.targets, delimiter=',')
			# Delete the third column and first row (useless)
			self._labels = np.delete(self._labels, 2, axis=1)
		except:
			print("Incorrect format or file.")
			sys.exit(1)

	def _processImagesSubset(self, images):
		# Data shape: [num_samples, height, width, channels]
		print("Processing images...")
		tmp = list()
		files = self.files[images]
		i = 1
		for f in files:
			assert f.endswith(".jpg")
			if i % 100 == 0:
				print '.',
				sys.stdout.flush()
			t = imread(self.images + f)
			tmp.append(t)
			i += 1

		self._inputs = np.array(tmp)
		# Normalize
		self._inputs = np.divide(self._inputs, 255.0)

		assert len(self._inputs.shape) == 4

		print("")
		print("Done processing image subset.")

	def _saveData(self, name, labels=np.array([])):
		if self._data == 'train':
			assert self._inputs.shape[0] != 0 and labels.shape[0] != 0
			assert self._inputs.shape[0] == labels.shape[0]

		if self._data == 'train':
			np.savez_compressed(name, inputs_train=self._inputs, targets_train=labels)
		elif self._data == 'test':
			np.savez_compressed(name, inputs_test=self._inputs)
		else:
			print("Data type not supported.")
			sys.exit(1)


#=========================================================================
# Helper functions
#=========================================================================

def ShowImage(img, number=0, gray=0):
	# img is a 128x128x3 numpy array
	plt.figure(number)
	plt.clf()
	if gray == 0:
		plt.imshow(img)
	else:
		plt.imshow(img, cmap=plt.get_cmap('gray'))
	plt.draw()
	plt.show()
	#raw_input('Press Enter.')


def Plot2D(data, labels, colors, number):
	#assert len(np.unique(labels[:,1])) == len(colors)
	assert data.shape[0] == len(labels)
	assert data.shape[1] == 2

	colorList = np.array([colors[int(i-1)] for i in labels[:,1]])
	print colorList.shape
	plt.figure(number)
	plt.clf()
	plt.scatter(data[:,0], data[:,1], c=colorList)
	plt.show()
	
def LoadData(datafile, dataType):
	assert datafile.endswith('.npz')
	d = np.load(datafile)
	data = dict()
	if dataType == 'train':
		data['inputs_train'] = d['inputs_train']
		data['targets_train'] = d['targets_train']
	elif dataType == 'test':
		data['inputs_test'] = d['inputs_test']
	else:
		print("Incorrect data type: {train / test}")
		sys.exit(1)
		
	return data


if __name__ == '__main__':
	# Run this file in order to get the data files (only needs to be run once
	# for the set of data.

	dataType = 'test'
	if dataType == 'train':
		pathToImages = 'Data/train/'
		labelsFile = 'Data/train.csv'
		data = DataTable(pathToImages, labelsFile, 'train')
		data.processImages()
	elif dataType == 'test':
		pathToImages = 'Data/val/'
		labelsFile = ''
		data = DataTable(pathToImages, labelsFile, 'test')
		data.processImages()



