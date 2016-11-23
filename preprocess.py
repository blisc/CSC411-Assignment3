# Basic data preprocessing -- data visualization, basic statistics,
# dimensionality reduction

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import SGD
from util import LoadData, ShowImage


def _getAllLabels(dataDir, listOfFiles):
	allLabels = np.empty(shape=(0,0))
	for datafile in listOfFiles:
		print 'Reading', datafile, '...'
		datafiledir = dataDir + '/' + datafile
		data = LoadData(datafiledir, 'train')
		labels = data['targets_train']
		assert labels.shape[1] == 2
		labels = labels[:,1]
		if allLabels.shape == (0,0):
			allLabels = labels
		else:
			allLabels = np.hstack((allLabels,labels))
	return allLabels


def _convertAllLabels(allLabels):
	num_samples = allLabels.shape[0]
	allLabels = allLabels.reshape(num_samples,1)
	return MultiLabelBinarizer().fit_transform(allLabels)


def _initAutoEncoder(numFeatures, numHiddenUnits, activation='relu'):
	# Input image placeholder
	input_img = Input(shape=(numFeatures,))

	# Encoded representation of image
	encoded = Dense(numHiddenUnits, activation=activation)(input_img)
	# Decoded lossy representation of image
	decoded = Dense(numFeatures, activation='sigmoid')(encoded)
	# Maps input image to reconstructed image
	autoencoder = Model(input=input_img, output=decoded)

	# Maps input image to encoded image
	encoder = Model(input=input_img, output=encoded)
	# Placeholder for encoded image
	encoded_input = Input(shape=(numHiddenUnits,))

	# Decoder model
	decoder_layer = autoencoder.layers[-1]
	decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

	sgd = SGD(lr=0.01, decay=0.0, momentum=0.8, nesterov=False)
	autoencoder.compile(optimizer=sgd, loss='mean_squared_error')
	return autoencoder, encoder, decoder

def LoadAllTrainData(dataDir, listOfFiles):
	allImages = np.empty(shape=(0,0))
	allLabels = np.empty(shape=(0,0))

	for datafile in listOfFiles:
		print 'Reading ', datafile, '...'
		datafiledir = dataDir + '/' + datafile
		data = LoadData(datafiledir, 'train')
		images = data['inputs_train']
		if allImages.shape == (0,0):
			allImages = images
		else:
			allImages = np.concatenate((allImages,images))
		labels = data['targets_train']
		assert labels.shape[1] == 2
		labels = labels[:,1]
		if allLabels.shape == (0,0):
			allLabels = labels
		else:
			allLabels = np.hstack((allLabels,labels))
			
	oneHotLabels = _convertAllLabels(allLabels)

	return allImages, oneHotLabels
	
	
def DoAutoEncoder(dataDir, listOfFiles):
	# Get first dataset
	print listOfFiles[0]
	datafiledir = dataDir + '/' + listOfFiles[0]
	data = LoadData(datafiledir, 'train')
	inputs = data['inputs_train']
	numSamples = inputs.shape[0]
	numFeatures = np.prod(inputs.shape[1:])

	# Flatten inputs
	inputs = inputs.reshape(numSamples, numFeatures)

	# Initialize model
	autoencoder, encoder, decoder = _initAutoEncoder(numFeatures, 100)

	# Fit first data set
	autoencoder.fit(inputs, inputs, batch_size=250, shuffle=True, nb_epoch=5)

	# Fit the rest
	for i in range(1,len(listOfFiles)):
		print listOfFiles[i]
		datafiledir = dataDir + '/' + listOfFiles[i]
		data = LoadData(datafiledir, 'train')
		inputs = data['inputs_train']
		assert numSamples == inputs.shape[0]
		assert numFeatures == np.prod(inputs.shape[1:])
		inputs = inputs.reshape(numSamples, numFeatures)
		autoencoder.fit(inputs, inputs, batch_size=250, shuffle=True, nb_epoch=5)

	# Draw first image
	datafiledir = dataDir + '/' + listOfFiles[0]
	data = LoadData(datafiledir, 'train')
	inputs = data['inputs_train']
	inputsFlat = inputs.reshape(numSamples, numFeatures)
	firstImage = inputsFlat[0,:]
	print firstImage.shape
	firstImage = firstImage.reshape(1,numFeatures)
	encoded_img = encoder.predict(firstImage)
	decoded_img = decoder.predict(encoded_img)
	decoded_img = decoded_img.reshape(1,128,128,3)
	ShowImage(decoded_img[0])


def DoNN(dataDir, listOfFiles):
	# First convert all the labels using one-hot encoding
	allLabels = _getAllLabels(dataDir, listOfFiles)
	oneHotLabels = _convertAllLabels(allLabels)

	# Defaults: lbfgs, 200 batch, RELU, constant learning rate,
	#			max_iter=200, tol=1e-4
	i = 0
	clf = MLPClassifier(hidden_layer_sizes=(100,), warm_start=True)
	for datafile in listOfFiles:
		print "Training on", datafile
		datafiledir = dataDir + '/' + datafile

		# Load data
		data = LoadData(datafiledir, 'train')
		inputs = data['inputs_train']
		assert len(inputs.shape) == 4

		# Flatten the inputs so it's 1D
		num_samples = inputs.shape[0]
		num_features = inputs.shape[1] * inputs.shape[2] * inputs.shape[3]
		flatInputs = inputs.reshape(num_samples, num_features)

		# Get correct subset of labels
		start = i * num_samples
		end = i * num_samples + num_samples
		labels = oneHotLabels[start:end,:]
		assert labels.shape[0] == num_samples
		assert num_samples == 1000 # Hard code for now

		# Debugger
		#name = 'labels_' + str(i) + '.out'
		#np.savetxt(name, oneHotLabels[start:end,:], delimiter=',')

		# Fit
		flatInpEnc = MultiLabelBinarizer().fit_transform(flatInputs)
		clf.fit(flatInputs, flatInpEnc)
		i += 1


def DoPCA(inputs, numComponents=10):
	# Performs PCA and transforms data into numComponents dimension space
	# inputs shape: [1, numFeatures]

	pca = PCA(n_components=numComponents)
	lowDimInputs = pca.fit_transform(inputs)
	print(pca.explained_variance_ratio_)

	return lowDimInputs


def GetDataDistribution(dataDir, listOfFiles):
	# Returns numpy array of labels for all training data

	allLabels = _getAllLabels(dataDir, listOfFiles)
	num_samples = allLabels.shape[0]
	np.savetxt('labels.out', allLabels.reshape(num_samples,1), delimiter=',')
	numUnique = len(np.unique(allLabels))
	return allLabels, numUnique


def PlotHistogramOfLabels(dataDir, listOfFiles):
	allLabels, numUniqueLabels = GetDataDistribution(dataDir, listOfFiles)
	plt.figure()
	plt.hist(allLabels, numUniqueLabels)
	plt.show()
	raw_input("Press Enter.")



