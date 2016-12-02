from __future__ import division

import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import cPickle

from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_validation import KFold
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, GridSearchCV, StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from skimage.feature import hog
from skimage import color

from preprocess import StandardizeData, DoIncrementalPCA, TransformData, LoadAllTrainData, DoPCA
from util import LoadData, ShowImage, GetAllLabels, LoadAllVGGData


def SaveModel(clf):
	with open('dumped_classifer.pkl', 'wb') as fid:
		cPickle.dump(clf, fid)


def GIST_KNN(gist_data_file, labelsFile):
	data = scio.loadmat(gist_data_file)['gist']
	labels = GetAllLabels(labelsFile)[:,1]


def DoAdaBoost(gist_data_file, labelsFile):
	data = scio.loadmat(gist_data_file)['gist']
	labels = GetAllLabels(labelsFile)[:,1]

	#rbf_feature = RBFSampler(gamma=1, n_components=1000)
	#data = rbf_feature.fit_transform(data)

	#myclf = RandomForestClassifier(class_weight='balanced', min_impurity_split=1e-4)
	myclf = RandomForestClassifier(min_impurity_split=1e-2)
	#myclf = LogisticRegression(class_weight='balanced', tol=1e-5, C=0.1)
	#myclf = SGDClassifier(class_weight='balanced', loss='hinge', eta0=0.01, learning_rate='constant')

	#clf = AdaBoostClassifier(base_estimator=myclf, n_estimators=25, algorithm='SAMME')
	clf = AdaBoostClassifier(base_estimator=myclf, n_estimators=50, learning_rate=0.0001)

	scores = cross_val_score(clf, data, labels)
	print scores.mean()


def DoSVM_VGG(dataDir, listOfVGGFiles, labelsFile):
	data = LoadAllVGGData(dataDir, listOfVGGFiles)
	numSamples = data.shape[0]
	numFeatures = np.prod(data.shape[1:])
	data = data.reshape(numSamples, numFeatures)
	labels = GetAllLabels(labelsFile)[:,1]

	print data.shape
	print labels.shape

	# Scale and PCA
	scaler = StandardScaler()
	data = scaler.fit_transform(data)

	pca = PCA(n_components=1000)
	data = pca.fit_transform(data)
	print 'Variance explained: ', np.sum(pca.explained_variance_ratio_)

	numFolds=5
	kf = KFold(data.shape[0], numFolds, shuffle=True)
	#kf = StratifiedKFold(numFolds, shuffle=True)

	print "Training..."
	clf = SVC(decision_function_shape='ovr', kernel='rbf', gamma=0.1, C=1, tol=1e-5)
	for train_idx, valid_idx in kf:
	#for train_idx, valid_idx in kf.split(data, labels):
		print 'Fold'
		inputs = data[train_idx,:]
		targets = labels[train_idx]
		val_inputs = data[valid_idx,:]
		val_targets = labels[valid_idx]
		clf = clf.fit(inputs, targets)
		predictions = clf.predict(val_inputs)
		assert len(val_targets) == len(predictions)

		train_pred = clf.predict(inputs)
		
		print 'Train acc: ', accuracy_score(targets, train_pred)
		print 'Validation acc: ', accuracy_score(val_targets, predictions)

	return clf, scaler


def DoSVM_GridSearch(gist_data_file, labelsFile):
	data = scio.loadmat(gist_data_file)['gist']
	labels = GetAllLabels(labelsFile)[:,1]
	assert data.shape[0] == labels.shape[0]

	# Scale and PCA
	scaler = StandardScaler()
	data = scaler.fit_transform(data)


	C_range = np.logspace(-3, 2, 5)
	gamma_range = np.logspace(-3, 2, 5)
	cv = StratifiedShuffleSplit(n_splits=4, test_size=0.2)
	param_grid = dict(gamma=gamma_range, C=C_range)
	svClf = SVC(tol=1e-6, decision_function_shape='ovr')
	grid = GridSearchCV(svClf, param_grid=param_grid, cv=cv, verbose=1, n_jobs=10)


	grid.fit(data, labels)
	print "Best parameters: %s (%0.2f)" % (grid.best_params_, grid.best_score_)

	scores = grid.cv_results_['mean_test_score'].reshape(len(C_range), len(gamma_range))
	np.savez('scores_25', s=scores)
	print scores



def DoSVM_Gist(gist_data_file, labelsFile):
	data = scio.loadmat(gist_data_file)['gist']
	labels = GetAllLabels(labelsFile)[:,1]

	# Scale and PCA
	scaler = StandardScaler()
	data = scaler.fit_transform(data)

	#pca = PCA(n_components=200)
	#data = pca.fit_transform(data)
	#print 'Variance explained: ', np.sum(pca.explained_variance_ratio_)

	numFolds=5
	kf = KFold(data.shape[0], numFolds, shuffle=True)
	#kf = StratifiedKFold(numFolds, shuffle=True)

	print "Training..."
	#clf = SVC(class_weight='balanced', decision_function_shape='ovr', kernel='rbf', gamma=0.001, tol=1e-6)

	#clf = SVC(decision_function_shape='ovr', kernel='rbf', gamma=0.0005, C=3.5, tol=1e-6)
	#clf = SVC(decision_function_shape='ovr', kernel='rbf', gamma=0.0005, C=5, tol=1e-6)
	clf = SVC(decision_function_shape='ovr', kernel='rbf', gamma=0.001, C=2.15443, tol=1e-6)
	#clf = SVC(decision_function_shape='ovr', kernel='rbf', gamma=0.0005, C=2, tol=1e-6)

	#clf = SVC(decision_function_shape='ovr', kernel='rbf', gamma=0.001, C=10, tol=1e-6)
	#clf = SVC(decision_function_shape='ovr', kernel='poly', gamma=0.1, tol=1e-5, degree=6, C=0.01, coef0=0)

	for train_idx, valid_idx in kf:
	#for train_idx, valid_idx in kf.split(data, labels):
		print 'Fold'
		inputs = data[train_idx,:]
		targets = labels[train_idx]
		val_inputs = data[valid_idx,:]
		val_targets = labels[valid_idx]
		clf = clf.fit(inputs, targets)
		predictions = clf.predict(val_inputs)
		assert len(val_targets) == len(predictions)

		train_pred = clf.predict(inputs)
		
		print 'Train acc: ', accuracy_score(targets, train_pred)
		print 'Validation acc: ', accuracy_score(val_targets, predictions)

	return clf, scaler, data, labels


def GetROC(clf, data, labels):
	n_classes = 8
	classes = np.arange(n_classes) + 1
	samples = np.arange(data.shape[0])
	np.random.shuffle(samples)
	
	test_prop = 0.2
	numTest = int(data.shape[0] * test_prop)
	testSet = data[0:numTest,:]
	y_test = labels[0:numTest]
	y_test = label_binarize(y_test, classes)
	y_score = clf.decision_function(testSet)

	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	for i in range(n_classes):
		fpr[i], tpr[i], _ = roc_curve(y_test[:,i], y_score[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])
	np.savez('roc', fpr=fpr, tpr=tpr)


def PlotROC(fpr, tpr):
	plt.figure()
	lw = 2
	plt.plot(fpr[2], tpr[2], color='darkorange',
	         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")


def PredictOnTest(gist_data_test_file, clf, scaler=None, pca=None):
	data = scio.loadmat(gist_data_test_file)['gist_test']
	if scaler:
		data = scaler.fit_transform(data)
	if pca:
		data = pca.fit_transform(data)
	pred = clf.predict(data)
	return pred


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
	# Do SVM on HOG features
	numClasses = 8
	classes = range(1,numClasses+1)

	# Scale data
	#scaler = StandardizeData(dataDir, listOfFiles)

	newData = np.empty((0,))
	
	#for i in range(0, len(listOfFiles)):
	for i in range(0, 1):
		print listOfFiles[i]
		datafiledir = dataDir + '/' + listOfFiles[i]
		data = LoadData(datafiledir, 'train')
		inputs = data['inputs_train']
		#targets = data['targets_train']
		#numSamples = inputs.shape[0]
		#numFeatures = np.prod(inputs.shape[1:])
		#targets = targets[:,1]

		inputs = color.rgb2gray(inputs)
		for img in inputs:
			print '.',
			sys.stdout.flush()
			if newData.shape[0] == 0:
				newData = hog(img)
			else:
				newimg = hog(img)
				newData = np.vstack((newData, newimg))
		print ''
	print newData.shape
	#np.savez('hogimages', img=newData)

	ShowImage(newData[0], gray=1)


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

	listOfVGGTrainingSetFiles = ['VGG16_train_1_1000.npz', 'VGG16_train_1001_2000.npz', \
							  'VGG16_train_2001_3000.npz', 'VGG16_train_3001_4000.npz', \
							  'VGG16_train_4001_5000.npz', 'VGG16_train_5001_6000.npz', \
							  'VGG16_train_6001_7000.npz']

	#DoSVM(dataDir, listOfTrainingSetFiles, numEpochs=15)
	#DoSVMFull(dataDir, listOfTrainingSetFiles)
	#DoSVM_new(dataDir, listOfTrainingSetFiles)

	## Basic SVM
	clf, scaler, data, labels = DoSVM_Gist('Data/gist_data.mat', 'Data/train.csv')
	#res = PredictOnTest('Data/gist_data_test.mat', clf, scaler=scaler)
	#np.savez('test_pred', labels=res)
	#SaveModel(clf)

	## ROC
	GetROC(clf, data, labels)

	## Testing boosting
	#DoAdaBoost('Data/gist_data.mat', 'Data/train.csv')

	## RBF Parameter Sweeping
	#DoSVM_GridSearch('Data/gist_data.mat', 'Data/train.csv')


	## Using SVM on VGG data
	#DoSVM_VGG(dataDir, listOfVGGTrainingSetFiles, 'Data/train.csv')



