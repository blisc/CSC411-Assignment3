from preprocess import LoadAllTrainData, GetAllLabels
from util import DataTable, ShowImage, Plot2D, LoadData, DataTable 
from model import Model
from deeplearningmodels.imagenet_utils import preprocess_input, decode_predictions
import scipy.io as scio
import numpy as np

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

	dataDir = 'Data/NPZ_data'
	listOfTrainingSetFiles = ['train_1_1000.npz', 'train_1001_2000.npz', \
								'train_2001_3000.npz', 'train_3001_4000.npz', \
								'train_4001_5000.npz', 'train_5001_6000.npz', \
								'train_6001_7000.npz']				
	gist_data_file = "Data/gist_data"
	labelsFile = "Data/train.csv"
	

	# dt = DataTable('Data/val', 'Data/train.csv', 'train')
	# dt.processImages()
	# dt = DataTable('Data/test_128', 'Data/train.csv', 'test')
	# dt.processImages()
								
	# listOfTrainingSetFiles = ['train_6001_7000.npz']

	# Get the distribution of training set labels
	#PlotHistogramOfLabels(dataDir, listOfTrainingSetFiles)

	# Train Model
	# print "Loading Data"
	# images = scio.loadmat(gist_data_file)['gist']
	# _,labels = GetAllLabels(labelsFile)
	
	# Get VGG16 features
	# model = Model()
	# VGG16 = model.VGG16_extract()
	# for file in listOfTrainingSetFiles:
		# images, labels = LoadAllTrainData('Data/NPZ_data/', [file], "224_")
		# images = images.astype(np.float32)
		# for i in [0,100,200,300,400,500,600,700,800,900]:
			# feature = VGG16.predict(preprocess_input(images[i:i+100]))
			# np.savez_compressed("4096_{}_{}".format(file,i), inputs_train=feature, targets_train=[-1])
	# testImages = LoadData('Data/NPZ_data/224_test_1_2000.npz', 'test')
	# testImages =  testImages['inputs_test'].astype(np.float32)
	# for i in [0,100,200,300,400,500,600,700,800,900]:
		# feature = VGG16.predict(preprocess_input(testImages[i:i+100]))
		# np.savez_compressed("4096_224_test_1_970_{}.npz".format(i), inputs_test=feature)
		# feature = VGG16.predict(preprocess_input(testImages[i+1000:i+1100]))
		# np.savez_compressed("4096_224_test_1_970_{}.npz".format(i+1000), inputs_test=feature)
	
	# for file in listOfTrainingSetFiles:
		# allImages = np.empty(shape=(0,0))
		# for i in [0,100,200,300,400,500,600,700,800,900]:
			# data = LoadData("4096_{}_{}.npz".format(file,i), 'train')
			# images = data['inputs_train']
			# if allImages.shape == (0,0):
				# allImages = images
			# else:
				# allImages = np.concatenate((allImages,images))
		# print "Saving ", file
		# print allImages.shape
		# np.savez_compressed("VGG16_"+file, inputs_train=allImages, targets_train=[-1])
	
	# for file in ['4096_224_test_1_970_{}.npz']:
		# allImages = np.empty(shape=(0,0))
		# for i in [0,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900]:
			# data = LoadData(file.format(i), 'test')
			# images = data['inputs_test']
			# if allImages.shape == (0,0):
				# allImages = images
			# else:
				# allImages = np.concatenate((allImages,images))
		# print "Saving ", file
		# print allImages.shape
		# np.savez_compressed("VGG16_test_1_970.npz", inputs_test=allImages)
	
	# print "Processing Data"
	# images, _ = LoadAllTrainData('Data/NPZ_data/', listOfTrainingSetFiles, "VGG16_")
	# data = dict()
	# images = images.reshape(-1, 7*7*512)
	# data["inputs_train"] = images[:-88].astype(np.float32)
	# data["targets_train"] = labels[:-88].astype(np.float32)
	# data["inputs_val"] = images[-88:].astype(np.float32)
	# data["targets_val"] = labels[-88:].astype(np.float32)
 	# # model = Model(batchSize=128, trainingIterations=100, kernel_1=[5, 5, 3, 32],kernel_2 = [5, 5, 32, 64],linear_hidden_size = 256)
 	# model = Model(batchSize=128, trainingIterations=1621,linear_hidden_size = 256)
	# model.createModelVGGTop()
 	# model.train(data)

	# # Find incorrect validation data
	# listOfTrainingSetFiles = ['train_6001_7000.npz']
	# images, _ = LoadAllTrainData('Data/NPZ_data/', listOfTrainingSetFiles, "VGG16_")
	# data = dict()
	# print(images[0,:,:,0])
	# test = np.mean(images,axis=(1,2,3))
	# print test.shape
	# np.savetxt("./test.txt", test, delimiter='\n')
	# images = images.reshape(-1, 7*7*512)
	# data["inputs_train"] = images[:-88].astype(np.float32)
	# data["targets_train"] = labels[:-88].astype(np.float32)
	# data["inputs_val"] = images[-88:].astype(np.float32)
	# data["targets_val"] = labels[-88:].astype(np.float32)
	# model = Model(batchSize=128, trainingIterations=5401,linear_hidden_size = 256)
	# model.createModelVGGTop()
	# model.inference("./results/tmp/valid/checkpoint-3240",data["inputs_train"][0:100])
	# print(np.argmax(data["targets_train"][0:100], 1))	
	
	# testImages = LoadData('Data/NPZ_data/VGG16_test_1_970.npz', 'test')
	# testImages = testImages["inputs_test"]
	# print(testImages[0,:,:,0])
	
	# # Test VGG16
	# listOfTrainingSetFiles = ['train_6001_7000.npz']
	# images, _ = LoadAllTrainData('Data/NPZ_data/', listOfTrainingSetFiles, "VGG16_")
	# testImages = LoadData('Data/NPZ_data/VGG16_test_1_2000.npz', 'test')
	# testImages = testImages['inputs_test'].reshape(-1,7*7*512)
	# images = testImages[0:100]
	# # ShowImage(images[1000],"Test")
	# images = images.reshape(-1,7*7*512)
	# for i in range(10):
		# ShowImage(data["inputs_val"][i,:,:,:],"before_{}".format(i))
	# test = np.array([imresize(data["inputs_val"][0],(224,224),interp="lanczos")])
	# print(images[-88])
	# print test.shape
	# for i,img in enumerate(data["inputs_val"]):
		# if i==0:
			# continue
		# test = np.concatenate((test,[imresize(img,(224,224),interp="lanczos")]))
	# test = test.astype(np.float32)
	# for i in range(10):
		# ShowImage(test[i,:,:,:],i)
	# print(test[0])
	# print test.shape
	# model = Model()
	# # VGG16 = model.VGG16_vanilla()
	# VGG16 = model.VGG16_test()
	# preds = VGG16.predict(images)
	# print(decode_predictions(preds))
	
	# Get inference results
	testImages = LoadData('Data/NPZ_data/VGG16_test_1_2000.npz', 'test')
	testImages = testImages['inputs_test'].reshape(-1,7*7*512)
	model = Model(linear_hidden_size = 512)
	model.createModelVGGTop()
	model.inference("./results/VGG16/VGGTop_Hidden_512_WD_0.0/valid/checkpoint-1620",testImages)
	
	# gist
	# gist_data_file = "Data/gist_data_test.mat"
	# labelsFile = "Data/train.csv"
	# print "Loading Data"
	# images = scio.loadmat(gist_data_file)['gist_test']
	# _,labels = GetAllLabels(labelsFile)
	# model = Model()
	# model.createModelFC1()
	# model.inference("results/gist-FC/valid/checkpoint-1080",images)
	
	
	# Reduce dimensionality
	# DoAutoEncoder(dataDir, listOfTrainingSetFiles)


#	# Preprocess
#	lowDimInputs = DoPCA(flatInputs, numComponents=2)
#	print(lowDimInputs.shape)
#
#	colors = ['r', 'y', 'b', 'g', 'm', 'c', 'k', 'w']
#	Plot2D(lowDimInputs, labels, colors, 2)


if __name__ == '__main__':
	main(debug=0)



