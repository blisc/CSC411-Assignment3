from util import DataTable, ShowImage


def main():
	pathToImages = 'Data/train/'
	labelsFile = 'Data/train.csv'
	data = DataTable(pathToImages, labelsFile)
	data.processImages(numImages=10)
	data.readLabels()
	i=0
	for line in data.getLabels():
		i+=1
		if i==11:
			break
		print line

	i=0
	for img in data.getInputs():
		i+=1
		if i==3:
			break
		ShowImage(img,i)


if __name__ == '__main__':
	main()
