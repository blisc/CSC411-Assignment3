from __future__ import division

import numpy as np
import scipy.io as scio

from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.cross_validation import KFold

from skimage.feature import hog

from preprocess import StandardizeData, DoIncrementalPCA, TransformData, LoadAllTrainData, DoPCA
from util import LoadData, ShowImage


def DoSVM_Gist(gist_data_file):
	data = scio.loadmat(gist_data_file)['gist']
	print data.shape


def DoSVMFull(dataDir, listOfFiles):
	scaler = StandardScaler()
	pca, inputs_transform, var, allLabels = DoPCA(dataDir, listOfFiles, scaler=scaler, numComponents=500)
	print "Variance explained:", np.sum(var)

	numFolds=10
	kf = KFold(inputs_transform.shape[0], numFolds, shuffle=True)

	print "Training..."
	clf = SVC(class_weight='balanced', decision_function_shape='ovr', kernel='rbf')
	for train_idx, valid_idx in kf:
		inputs = inputs_transform[train_idx,:]
		targets = allLabels[train_idx]
		val_inputs = inputs_transform[valid_idx,:]
		val_targets = allLabels[valid_idx]
		clf.fit(inputs, targets)
		predictions = clf.predict(val_inputs)
		assert len(val_targets) == len(predictions)
		print accuracy_score(val_targets, predictions)


def DoSVM_new(dataDir, listOfFiles):
	numClasses = 8
	classes = range(1,numClasses+1)

	# Scale data
	#scaler = StandardizeData(dataDir, listOfFiles)

	#for i in range(0, len(listOfFiles)):
	for i in range(0, 1):
		print listOfFiles[i]
		datafiledir = dataDir + '/' + listOfFiles[i]
		data = LoadData(datafiledir, 'train')
		inputs = data['inputs_train']
		targets = data['targets_train']
		numSamples = inputs.shape[0]
		numFeatures = np.prod(inputs.shape[1:])
		targets = targets[:,1]

		inputs = color.rgb2gray(inputs)
	ShowImage(inputs[0], gray=1)

		# Transform data
		#inputs = inputs.reshape(numSamples, numFeatures)


def DoSVM(dataDir, listOfFiles, numEpochs=30):
	numClasses = 8
	classes = range(1,numClasses+1)

	# Scale data
	scaler = StandardizeData(dataDir, listOfFiles)

	# Do PCA
	ipca, var = DoIncrementalPCA(dataDir, listOfFiles, scaler, numComponents=100)
	print "Variance explained: ", np.sum(var)

	# Train model with SGD classifier (with hinge loss = SVM)
	# eta0 is learning rate
	clf = SGDClassifier(loss='hinge', eta0=0.001, learning_rate='constant')
	print "Training model..."
	holdoutPercent = 0.2
	validationAccuracy = np.zeros((numEpochs,))
	trainAccuracy = np.zeros((numEpochs,))
	rbf_feature = RBFSampler(gamma=1, n_components=500)
	for k in range(0, numEpochs):
		valSetIndices = list()
		trainSetIndices = list()
		print "========= EPOCH %d =========" % k
		for i in range(0, len(listOfFiles)):
			print listOfFiles[i]
			datafiledir = dataDir + '/' + listOfFiles[i]
			data = LoadData(datafiledir, 'train')
			inputs = data['inputs_train']
			targets = data['targets_train']
			numSamples = inputs.shape[0]
			numFeatures = np.prod(inputs.shape[1:])
			assert numSamples == targets.shape[0]
			targets = targets[:,1]
	
			# Transform data
			inputs = inputs.reshape(numSamples, numFeatures)
			inputs = TransformData(inputs, ipca, scaler)
	
			# Get validation and training set
			holdoutAmount = int(holdoutPercent * numSamples)
			rnd_idx = np.arange(inputs.shape[0])
			np.random.shuffle(rnd_idx)
			numTrain = numSamples - holdoutAmount
			trainSet = inputs[rnd_idx[0:numTrain]]
			targetSet = targets[rnd_idx[0:numTrain]]

			trainSetIdx = rnd_idx[0:numTrain]
			trainSetIndices.append(trainSetIdx)
			valSetIdx = rnd_idx[numTrain:]
			valSetIndices.append(valSetIdx)

			# Use RBF kernel
			trainSet = rbf_feature.fit_transform(trainSet)
			clf = clf.partial_fit(trainSet, targetSet, classes=classes)

		# Temp hack to predict on training set to get train accuracy
		print "Predicting on training set..."
		train_acc = PredictOnValidation(clf, dataDir, listOfFiles, trainSetIndices, \
										numClasses, scaler, ipca, rbf_feature)
		print "Predicting on validation set..."
		val_acc = PredictOnValidation(clf, dataDir, listOfFiles, valSetIndices, \
									numClasses, scaler, ipca, rbf_feature)
		trainAccuracy[k] = train_acc
		validationAccuracy[k] = val_acc
		print 'Train accuracy: ', train_acc
		print 'Validation accuracy: ', val_acc

	np.savez('trainAcc', train=trainAccuracy)
	np.savez('validAcc', val=validationAccuracy)


def PredictOnValidation(clf, dataDir, listOfFiles, valSetIndices, numClasses, \
						scaler, ipca, rbf_feature, predict = 1):
	assert isinstance(clf, SGDClassifier)
	assert len(valSetIndices) == len(listOfFiles)

	# Predict on set
	classes = range(1,numClasses+1)
	predictions = np.empty((0,))
	trueLabels = np.empty((0,))
	for i in range(0, len(listOfFiles)):
		valSetIdx = valSetIndices[i]
		print listOfFiles[i]
		datafiledir = dataDir + '/' + listOfFiles[i]
		data = LoadData(datafiledir, 'train')
		inputs = data['inputs_train']
		targets = data['targets_train']
		numSamples = inputs.shape[0]
		numFeatures = np.prod(inputs.shape[1:])
		assert numSamples == targets.shape[0]
		targets = targets[:,1]

		# Transform data and get validation set
		inputs = inputs.reshape(numSamples, numFeatures)
		inputs = TransformData(inputs, ipca, scaler)
		inputs = inputs[valSetIdx,:]
		targets = targets[valSetIdx]
		inputs = rbf_feature.fit_transform(inputs)

		# Get predictions
		if not predict:
			valPredictions = clf.decision_function(inputs)
			targets = label_binarize(targets, classes=classes)
		else:
			valPredictions = clf.predict(inputs)

		if predictions.shape[0] == 0:
			predictions = valPredictions
		else:
			if not predict:
				predictions = np.vstack((predictions, valPredictions))
			else:
				predictions = np.hstack((predictions, valPredictions))
		if trueLabels.shape[0] == 0:
			trueLabels = targets
		else:
			if not predict:
				trueLabels = np.vstack((trueLabels, targets))
			else:
				trueLabels = np.hstack((trueLabels, targets))

	print trueLabels.shape, predictions.shape
	assert len(trueLabels) == len(predictions)
	total = len(trueLabels)
	correct = np.sum((predictions == trueLabels).astype(int))
	i=0
	for i in range(0, len(predictions)):
		if predictions[i] == trueLabels[i]:
			print predictions[i]
	return correct / total


def DoKNN(dataDir, listOfFiles):
	numClasses = 8
	classes = range(1,numClasses+1)

	# Scale data
	scaler = StandardizeData(dataDir, listOfFiles)

	# Do PCA
	ipca, var = DoIncrementalPCA(dataDir, listOfFiles, scaler, numComponents=100)
	print "Variance explained: ", np.sum(var)
	for i in range(0, len(listOfFiles)):
		print listOfFiles[i]
		datafiledir = dataDir + '/' + listOfFiles[i]
		data = LoadData(datafiledir, 'train')
		inputs = data['inputs_train']
		targets = data['targets_train']
		numSamples = inputs.shape[0]
		numFeatures = np.prod(inputs.shape[1:])
		assert numSamples == targets.shape[0]
		targets = targets[:,1]

		# Transform data
		inputs = inputs.reshape(numSamples, numFeatures)
		inputs = TransformData(inputs, ipca, scaler)


if __name__ == '__main__':
	dataDir = 'Data/NPZ_data'
	listOfTrainingSetFiles = ['train_1_1000.npz', 'train_1001_2000.npz', \
							  'train_2001_3000.npz', 'train_3001_4000.npz', \
							  'train_4001_5000.npz', 'train_5001_6000.npz', \
							  'train_6001_7000.npz']

	#DoSVM(dataDir, listOfTrainingSetFiles, numEpochs=15)
	#DoSVMFull(dataDir, listOfTrainingSetFiles)
	#DoSVM_new(dataDir, listOfTrainingSetFiles)

	DoSVM_gist('Data/gist_data.mat')



