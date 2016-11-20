from util import DataTable

def main():
	pathToImages = 'Data/train/'
	labelsFile = 'Data/train.csv'
	data = DataTable(pathToImages, labelsFile)
	data.processImages()

if __name__ == '__main__':
	main()
