import numpy as np
from sklearn.decomposition import PCA
from collections import defaultdict
from util import LoadData


def DoPCA(inputs, numComponents=10):
	# Performs PCA and transforms data into numComponents dimension space
	# inputs shape: [1, numFeatures]

	pca = PCA(n_components=numComponents)
	lowDimInputs = pca.fit_transform(inputs)
	print(pca.explained_variance_ratio_)

	return lowDimInputs


def GetDataDistribution(dataDir, listOfFiles):
	# Returns numpy array of labels for all training data

	allLabels = np.empty(shape=(0,0))
	for datafile in listOfFiles:
		print 'Reading ', datafile, '...'
		datafiledir = dataDir + '/' + datafile
		data = LoadData(datafiledir, 'train')
		labels = data['targets_train']
		assert labels.shape[1] == 2
		labels = labels[:,1]
		if allLabels.shape == (0,0):
			allLabels = labels
		else:
			allLabels = np.hstack((allLabels,labels))

	np.savetxt('labels.out', allLabels.T, delimiter=',')
	numUnique = len(np.unique(allLabels))
	return allLabels, numUnique


