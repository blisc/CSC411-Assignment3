# Basic data preprocessing -- data visualization, basic statistics,
# dimensionality reduction

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from keras.layers import Input, Dense
from keras.models import Model, load_model
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping

from skimage import filters, feature
from scipy import ndimage
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

	#sgd = SGD(lr=0.1, decay=0.0, momentum=0.9, nesterov=False)
	#autoencoder.compile(optimizer=sgd, loss='mean_squared_error')
	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
	#adam = Adam(lr=0.0005)
	#autoencoder.compile(optimizer=adam, loss='mean_squared_error')
	return autoencoder, encoder, decoder


def _DoAutoEncoder(dataDir, listOfFiles):
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
	early_stop = EarlyStopping(min_delta=0.000001)
	hist = autoencoder.fit(inputs, inputs, batch_size=20, shuffle=True, \
						   nb_epoch=70, validation_split=0.2, callbacks=[early_stop])
	tmp = np.array(hist.history['loss'])
	print (hist.history['loss'])

	# Save model
	#autoencoder.save('my_model_test.h5')

	# Plot loss
	plt.figure()
	plt.plot(tmp)
	plt.show()

	# Fit the rest
	for i in range(1,len(listOfFiles)):
	#	autoencoder = load_model('my_model_test.h5')
		print listOfFiles[i]
		datafiledir = dataDir + '/' + listOfFiles[i]
		data = LoadData(datafiledir, 'train')
		inputs = data['inputs_train']
		assert numSamples == inputs.shape[0]
		assert numFeatures == np.prod(inputs.shape[1:])
		inputs = inputs.reshape(numSamples, numFeatures)
		hist = autoencoder.fit(inputs, inputs, batch_size=20, shuffle=True, \
							   nb_epoch=70, validation_split=0.2, callbacks=[early_stop])
		tmp = np.hstack((tmp, hist.history['loss']))
		print(hist.history['loss'])
		plt.figure()
		plt.plot(tmp)
		plt.show()

	np.savez('loss', loss=tmp)

	print "Saving model..."
	autoencoder.save('my_model_test.h5')

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


def DoAutoEncoder(dataDir, listOfFiles):
	# Initialize
	d_file = dataDir + '/' + listOfFiles[0]
	d = LoadData(d_file, 'train')
	inputs = d['inputs_train']
	losses = np.empty((0,))
	numSamples = inputs.shape[0]
	numFeatures = np.prod(inputs.shape[1:])
	autoencoder, encoder, decoder = _initAutoEncoder(numFeatures, 100)
	early_stop = EarlyStopping(min_delta=0.00000001, patience=2, mode='min')

	# Do training
	for i in range(10):
		print '=============== Epoch %d ===============' % i
		for j in range(len(listOfFiles)):
			d_file = dataDir + '/' + listOfFiles[j]
			d = LoadData(d_file, 'train')
			inputs = d['inputs_train']
			assert numSamples == inputs.shape[0]
			assert numFeatures == np.prod(inputs.shape[1:])
			inputs = inputs.reshape(numSamples, numFeatures)
			rnd_idx = np.arange(inputs.shape[0])
			np.random.shuffle(rnd_idx)
			inputs = inputs[rnd_idx]
			hist = autoencoder.fit(inputs, inputs, batch_size=10, shuffle=True, \
								   nb_epoch=10, validation_split=0.2, callbacks=[early_stop])
			losses = np.hstack((losses, hist.history['loss']))

	np.savez('loss', loss=losses)
	plt.plot(losses)
	plt.show()

	print "Saving model..."
	autoencoder.save('my_model.h5')

	_drawReconstructions(encoder, decoder, dataDir, listOfFiles, 1)


def _drawReconstructions(encoder, decoder, dataDir, listOfFiles, imageID):
	# HARD CODE. Make sure the data files are in increasing order in main.py.
	if imageID > 1000 and imageID < 2001:
		datafiledir = dataDir + '/' + listOfFiles[1]
	elif imageID > 2000 and imageID < 3001:
		datafiledir = dataDir + '/' + listOfFiles[2]
	elif imageID > 3000 and imageID < 4001:
		datafiledir = dataDir + '/' + listOfFiles[3]
	elif imageID > 4000 and imageID < 5001:
		datafiledir = dataDir + '/' + listOfFiles[4]
	elif imageID > 5000 and imageID < 6001:
		datafiledir = dataDir + '/' + listOfFiles[5]
	elif imageID > 6000 and imageID < 7001:
		datafiledir = dataDir + '/' + listOfFiles[6]
	else:
		datafiledir = dataDir + '/' + listOfFiles[0]
	img = imageID % 1000

	# Draw first image
	data = LoadData(datafiledir, 'train')
	inputs = data['inputs_train']
	numSamples = inputs.shape[0]
	numFeatures = np.prod(inputs.shape[1:])
	inputsFlat = inputs.reshape(numSamples, numFeatures)
	myImage = inputsFlat[img,:]
	print myImage.shape
	myImage = myImage.reshape(1,numFeatures)
	encoded_img = encoder.predict(myImage)
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


def FilterImage(trainingSetFile, img):
	# Function is used to test out filters
	data = LoadData(trainingSetFile, 'train')
	inputs = data['inputs_train']
	numSamples = inputs.shape[0]
	numFeatures = np.prod(inputs.shape[1:])
	myImage = inputs[img]

#========= ndimage sobel edge detector
	myImageGray = _rgb2gray(myImage)
	sx = ndimage.sobel(myImageGray, axis=0, mode='constant')
	sy = ndimage.sobel(myImageGray, axis=1, mode='constant')
	sob = np.hypot(sx, sy)
	ShowImage(sob)

#========= sobel edge detector
#	myImageGray = _rgb2gray(myImage)
#	edges = filters.sobel(myImageGray)
#	ShowImage(edges)

#========= canny edge detector
#	myImageGray = _rgb2gray(myImage)
#	edges = feature.canny(myImageGray)
#	ShowImage(edges)

#========= convert to grayscale
#	myImage = _rgb2gray(myImage)
#	ShowImage(myImage, number=0)

#========= segmentation
#	xpass = ndimage.gaussian_filter(myImage, sigma=0.1)
#
#	mask = (xpass > xpass.mean()).astype(np.float)
#	mask += 0.1 * xpass
#	img = mask + 0.2*np.random.randn(*mask.shape)
#	binary_img = img > 0.5
#	hist, bin_edges = np.histogram(img)
#	bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
#
#	plt.plot(bin_centers, hist, lw=2)
#	plt.show()
#
#	ShowImage(binary_img)

#========= high pass
#	xpass = myImage - xpass
#	xpass = filters.threshold_otsu(myImage)
#	ShowImage(xpass, number=1)


def _rgb2gray(img):
	return np.dot(img[...,:3], [0.299, 0.587, 0.114])


def GaussianFilter(data):
	# data: [samples,dim1,dim2,channels]
	assert len(data.shape) == 4
	assert data.shape[1] == data.shape[2] == 128
	assert data.shape[3] == 3
	f_data = ndimage.gaussian_filter(data,sigma=1)
	return f_data


def Get_RGB_Intensity_Stats(dataDir, listOfFiles, filterData='', statsType='avg'):
	# filterData = {'', 'gauss'}
	# statsType = {'avg', 'max', 'min'}

	R_intensity = defaultdict(int)
	G_intensity = defaultdict(int)
	B_intensity = defaultdict(int)
	classPopulation = defaultdict(int) # Number of samples from each class

	for i in range(0, len(listOfFiles)):
		print listOfFiles[i]
		datafiledir = dataDir + '/' + listOfFiles[i]
		d = LoadData(datafiledir, 'train')
		if filterData == '':
			inputs = d['inputs_train']
		elif filterData == 'gauss':
			inputs = GaussianFilter(d['inputs_train'])
		targets = d['targets_train']
		targets = targets[:,1]
		assert inputs.shape[0] == targets.shape[0]

		for j in range(1,9):
			# Get samples from class 'j'
			idxSamples = np.where(targets == j)[0]
			classSamples = inputs[idxSamples]

			if statsType == 'avg':
				red = np.sum(classSamples[:,:,:,0])
				green = np.sum(classSamples[:,:,:,1])
				blue = np.sum(classSamples[:,:,:,2])
			else:
				numSamples = len(idxSamples)
				imgSize = np.prod(classSamples[:,:,:,0].shape[1:])
				r = np.reshape(classSamples[:,:,:,0], (numSamples, imgSize))
				g = np.reshape(classSamples[:,:,:,1], (numSamples, imgSize))
				b = np.reshape(classSamples[:,:,:,2], (numSamples, imgSize))
				if statsType == 'max':
					red = np.sum(np.amax(r,axis=1))
					green = np.sum(np.amax(g,axis=1))
					blue = np.sum(np.amax(b,axis=1))
				elif statsType == 'min':
					red = np.sum(np.amin(r,axis=1))
					green = np.sum(np.amin(g,axis=1))
					blue = np.sum(np.amin(b,axis=1))

			R_intensity[j] += red
			G_intensity[j] += green
			B_intensity[j] += blue
			classPopulation[j] += len(classSamples)

	# Hard code image size 128x128. Divide by (class population * 128 * 128)
	# to get the average intensities.
	for i in range(1,9):
		classPop = classPopulation[i]
		if statsType == 'avg':
			R_intensity[i] = R_intensity[i] / (128 * 128 * classPop)
			G_intensity[i] = G_intensity[i] / (128 * 128 * classPop)
			B_intensity[i] = B_intensity[i] / (128 * 128 * classPop)
		elif statsType == 'max' or statsType == 'min':
			R_intensity[i] = R_intensity[i] / classPop
			G_intensity[i] = G_intensity[i] / classPop
			B_intensity[i] = B_intensity[i] / classPop

	#print classPopulation
	#print R_intensity, G_intensity, B_intensity
	return R_intensity, G_intensity, B_intensity


def PlotAverageIntensities(red, green, blue, statsType='avg'):
	numClasses = 8
	redMeans = list()
	greenMeans = list()
	blueMeans = list()
	for i in range(1,numClasses+1):
		redMeans.append(red[i])
		greenMeans.append(green[i])
		blueMeans.append(blue[i])

	width = 0.35
	entryWidth = 3*width+0.2
	ind = np.arange(numClasses)
	fig, ax = plt.subplots()
	rectsR = ax.bar(ind*entryWidth, redMeans, width, color='r', label='Red')
	rectsG = ax.bar(ind*entryWidth+width, greenMeans, width, color='g', label='Green')
	rectsB = ax.bar(ind*entryWidth+(2*width), blueMeans, width, color='b', label='Blue')
	
	# add some text for labels, title and axes ticks
	if statsType == 'avg':
		ax.set_ylabel('Average Intensity (Normalized)')
		ax.set_title('Average RGB Intensities')
	elif statsType == 'max':
		ax.set_ylabel('Max Intensity (Normalized)')
		ax.set_title('Max RGB Intensities')
	elif statsType == 'min':
		ax.set_ylabel('Min Intensity (Normalized)')
		ax.set_title('Min RGB Intensities')
	ax.set_xticks(ind * entryWidth + 1.5*width)
	ax.set_xticklabels(('1', '2', '3', '4', '5', '6', '7', '8'))
	#ax.legend()

	plt.show()


if __name__ == '__main__':
	#trainingSetFile = 'Data/NPZ_data/train_1_1000.npz'
	#img = 41
	#FilterImage(trainingSetFile, img)

	dataDir = 'Data/NPZ_data'
	listOfTrainingSetFiles = ['train_1_1000.npz', 'train_1001_2000.npz', \
							  'train_2001_3000.npz', 'train_3001_4000.npz', \
							  'train_4001_5000.npz', 'train_5001_6000.npz', \
							  'train_6001_7000.npz']

	# Get average RGB intensity histogram
	#R, G, B = Get_RGB_Intensity_Stats(dataDir, listOfTrainingSetFiles, filterData='gauss')
	#R, G, B = Get_RGB_Intensity_Stats(dataDir, listOfTrainingSetFiles, statsType='max')
	R, G, B = Get_RGB_Intensity_Stats(dataDir, listOfTrainingSetFiles, statsType='min')
	PlotAverageIntensities(R, G, B, statsType='min')



