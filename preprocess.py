import numpy as np
from sklearn.decomposition import PCA

def DoPCA(inputs, numComponents=10):
	# Performs PCA and transforms data into numComponents dimension space
	# inputs shape: [1, numFeatures]

	pca = PCA(n_components=numComponents)
	lowDimInputs = pca.fit_transform(inputs)
	print(pca.explained_variance_ratio_)

	return lowDimInputs


