import os, csv, sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread


#=========================================================================
# Class for processing images and their labels
#=========================================================================

class DataTable:
	def __init__(self, imgPath, targetPath):
		# imgPath: path to image directory
		# targetPath: path to csv file containing labels
		self.images = imgPath + '/'
		self.targets = targetPath
		self._numImagesProcessed = None

		# [num_samples, height, width, channels]
		self._inputs = np.array([])
		# [num_samples, all features]
		self._flatInputs = np.array([])
		# [num_samples, 1]
		self._labels = np.array([])

		# Dictionary containing all the data
		self._dataTable = dict()

	def getDataTable(self):
		if self._labels.shape[0] == 0:
			print("Did not read labels.")
			sys.exit(1)
		if self._inputs.shape[0] == 0 or self._flatInputs.shape[0] == 0:
			print("Did not read trainig samples.")
			sys.exit(1)

		if self._numImagesProcessed:
			self._labels = self._labels[0:self._numImagesProcessed,:]

		print(self._inputs.shape)
		print(self._flatInputs.shape)
		print(self._labels.shape)

		assert self._inputs.shape[0] == self._flatInputs.shape[0] == self._labels.shape[0]
		self._dataTable['inputs'] = self._inputs
		self._dataTable['flat_inputs'] = self._flatInputs
		self._dataTable['labels'] = self._labels
		return self._dataTable

	def readLabels(self):
		# Reads labels file for the training data
		# [ [number, label], [number, label], ... , [number, label] ]
		try:
			self._labels = np.genfromtxt(self.targets, delimiter=',')
			# Delete the third column and first row (useless)
			self._labels = np.delete(self._labels, 2, axis=1)
			self._labels = np.delete(self._labels, 0, axis=0)
		except:
			print("Incorrect format or file.")
			sys.exit(1)

	def processImages(self, numImages = None):
		# Reads in the training images. Set "numImages" if you only want
		# to process the first N images in the directory.
		# Data shape: [num_samples, height, width, channels]
		print("Processing images...")
		self._numImagesProcessed = numImages
		tmp = list()
		i = 0
		for filename in os.listdir(self.images):
			if numImages:
				i += 1
				if i == numImages+1:
					break
			if filename.endswith(".jpg"):
				t = imread(self.images + filename)
				tmp.append(t)
		self._inputs = np.array(tmp)
		# Normalize
		self._inputs = np.divide(self._inputs, 255.0)

		assert len(self._inputs.shape) == 4
		numFeatures = self._inputs.shape[1] * self._inputs.shape[2] * self._inputs.shape[3]
		self._flatInputs = self._inputs.reshape(self._inputs.shape[0], numFeatures)
		print("Done processing images.")


#=========================================================================
# Helper functions
#=========================================================================

def ShowImage(img, number=0):
	# img is a 128x128x3 numpy array
	plt.figure(number)
	plt.clf()
	plt.imshow(img)
	plt.draw()
	plt.show()
	raw_input('Press Enter.')


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


