import os
import numpy as np
import matplotlib.pyplot as plt
from util import DataTable, ShowImage, Plot2D
from preprocess import DoPCA


def printSomeLabels(labels, number):
	# labels shape: [num_samples, 1]
	# number: number of labels to print
	i = 0
	for label in labels:
		i += 1
		if i == number+1:
			break
		print label


def dispSomeImages(images, number):
	# images shape: [num_samples, height, width, channels]
	# number: number of images to display
	i = 0
	for image in images:
		i += 1
		if i == number+1:
			break
		ShowImage(image, i)


def loadData(dataDir):
	dataDir += '/'
	trainData = None
	trainLabels = None
	testData = None
	for f in os.listdir(dataDir):
		print f
		if f.endswith(".npz"):
			data = np.load(dataDir + f)

			# Training set
			if f.startswith('train'):
				if trainData == None:
					trainData = data['inputs_train']
				else:
					trainData = np.vstack((trainData, data['inputs_train']))

				if trainLabels == None:
					trainLabels = data['targets_train']
				else:
					trainLabels = np.vstack((trainLabels, data['targets_train']))

			# Test set
			elif f.startswith('test'):
				if testData == None:
					testData = data['inputs_test']
				else:
					testData = np.vstack((testData, data['inputs_test']))

			# Unsupported data type
			else:
				sys.exit(1)

	print trainData.shape
	print trainLabels.shape
	print testData.shape
	return trainData, trainLabels, testData


def main(debug=0):

	dataDir = 'Data/NPZ_data/'
	loadData(dataDir)

#	# Preprocess
#	lowDimInputs = DoPCA(flatInputs, numComponents=2)
#	print(lowDimInputs.shape)
#
#	colors = ['r', 'y', 'b', 'g', 'm', 'c', 'k', 'w']
#	Plot2D(lowDimInputs, labels, colors, 2)


if __name__ == '__main__':
	main(debug=0)


