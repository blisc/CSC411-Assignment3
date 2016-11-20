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
		self._inputs = np.array([])
		self._labels = np.array([])

	def getInputs(self):
		if self._inputs.shape[0] == 0:
			print("Did not read trainig samples.")
			sys.exit(1)
		return self._inputs

	def getLabels(self):
		if self._labels.shape[0] == 0:
			print("Did not read labels.")
			sys.exit(1)
		return self._labels

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
