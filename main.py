from util import DataTable

def main():
	pathToImages = 'Data/train/'
	labelsFile = 'Data/train.csv'
	data = DataTable(pathToImages, labelsFile)
	data.processImages()
	data.readLabels()
	i=0
	for line in data.labels:
		i+=1
		if i==11:
			break
		print line

if __name__ == '__main__':
	main()
