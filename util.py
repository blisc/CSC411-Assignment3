import os
from skimage.io import imread

class DataTable:
	def __init__(self, imgPath, targetPath):
		# imgPath is path to image directory
		# targetPath is csv file containing labels
		self.images = imgPath + '/'
		self.targets = targetPath

	def readTarget(self):
		pass

	def processImages(self):
		print("Processing images...")
		for filename in os.listdir(self.images):
			if filename.endswith(".jpg"):
				t = imread(self.images + filename)
		print("Done processing images.")
