import os, csv, sys
import numpy as np
from skimage.io import imread

class DataTable:
	def __init__(self, imgPath, targetPath):
		# imgPath is path to image directory
		# targetPath is csv file containing labels
		self.images = imgPath + '/'
		self.targets = targetPath
		self.data = None
		self.labels = None

	def readLabels(self):
		# Reads labels file for the training data
		# [ [number, label], [number, label], ... , [number, label] ]
		try:
			self.labels = np.genfromtxt(self.targets, delimiter=',')
			# Delete the third column and first row (useless)
			self.labels = np.delete(self.labels, 2, axis=1)
			self.labels = np.delete(self.labels, 0, axis=0)
		except:
			print("Incorrect format or file.")
			sys.exit(1)

	def processImages(self):
		# Reads training images
		# Data shape: [num samples, height, width, channels]
		print("Processing images...")
		tmp = list()
		for filename in os.listdir(self.images):
			if filename.endswith(".jpg"):
				t = imread(self.images + filename)
				tmp.append(t)
		self.data = np.array(tmp)
		print("Done processing images.")

