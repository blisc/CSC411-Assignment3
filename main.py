from util import DataTable, ShowImage, Plot2D
from preprocess import DoPCA
import matplotlib.pyplot as plt


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


def main(debug=0):

#
#	# Some debugging / sanity check
#	if debug:
#		printSomeLabels(labels, 10)
#		dispSomeImages(inputs, 1)
#
#	# Preprocess
#	lowDimInputs = DoPCA(flatInputs, numComponents=2)
#	print(lowDimInputs.shape)
#
#	colors = ['r', 'y', 'b', 'g', 'm', 'c', 'k', 'w']
#	Plot2D(lowDimInputs, labels, colors, 2)


if __name__ == '__main__':
	main(debug=0)


