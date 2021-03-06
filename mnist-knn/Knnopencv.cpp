#include "Knnopencv.h"


Knnopencv::Knnopencv()
{
}
void Knnopencv::getNumFrequency(CvMat*& labels){
	
	std:map<int, int> labelCountMap;


		for (int i = 0; i<labels->rows; i++)
	{	
			int num = labels->data.fl[i];
			if (labelCountMap.find(num) == labelCountMap.end())
			{
				labelCountMap[num] = 1;
			}
			else{
				labelCountMap[num] = labelCountMap[num] + 1;
			}
		

	}

		//print it out
		typedef std::map<int, int>::iterator it_type;
		for (it_type iterator = labelCountMap.begin(); iterator != labelCountMap.end(); iterator++) {
			printf("num = %d, count = %d\n", iterator->first, iterator->second);
		}
}
void Knnopencv::extractTrainingData(int& numImages, CvMat *& trainingVectors, CvMat*& trainingLabels)
{
	FILE *fp = fopen("D:\\baiducloud\\tech\\OpenCV\\basicOCR\\data\\mnist\\train-images-idx3-ubyte\\train-images.idx3-ubyte", "rb");

	FILE *fp2 = fopen("D:\\baiducloud\\tech\\OpenCV\\basicOCR\\data\\mnist\\train-labels-idx1-ubyte\\train-labels.idx1-ubyte", "rb");

	int magicNumber = readFlippedInteger(fp);
	int numImages2 = readFlippedInteger(fp);

	int numRows = readFlippedInteger(fp);

	int numCols = readFlippedInteger(fp);

	fseek(fp2, 0x08, SEEK_SET);

	//store all the images and labels
	int size = numRows*numCols;
	trainingVectors = cvCreateMat(numImages, size, CV_32FC1);
	trainingLabels = cvCreateMat(numImages, 1, CV_32FC1);
	//with memory in place, we read data from the files
	BYTE *temp = new BYTE[size];
	BYTE tempClass = 0;
	for (int i = 0; i<numImages; i++)
	{

		fread((void*)temp, size, 1, fp);

		fread((void*)(&tempClass), sizeof(BYTE), 1, fp2);

		trainingLabels->data.fl[i] = tempClass;

		for (int k = 0; k<size; k++)
			trainingVectors->data.fl[i*size + k] = temp[k];
	}

	//Data has been stored into the opencv matrix, it's ok to relase the file handler now
	fclose(fp);
	fclose(fp2);
	delete[] temp;

	getNumFrequency(trainingLabels);

}

void predictonTrainingSamples(KNearest &knn, CvMat *& testVectors,
	CvMat *& actualLabels, int numImages){

	CvMat *testLabels = cvCreateMat(numImages, 1, CV_32FC1);
	knn.find_nearest(testVectors, 5, testLabels);


	//Get the error rate and print the result out
	int totalCorrect = 0;
	for (int i = 0; i < numImages; i++){
		if (testLabels->data.fl[i] == actualLabels->data.fl[i])
		{
			totalCorrect++;
		}
		else{
			printf("\n Error: image id=%d number %f was mistaken as %f", i, actualLabels->data.fl[i], testLabels->data.fl[i]);
		}

	}
	printf("\nError Rate: %.1f%%",
		(double)100 - (double)totalCorrect * 100 / (double)numImages);
	
}
void Knnopencv::test()
{
	printf("Use KNN learning algorithm to recognize handwritten digit\n");
	//number of taining samples to be used
	int numImages = 6000;
	CvMat *trainingVectors = 0;
	CvMat *trainingLabels = 0;
	extractTrainingData(numImages, trainingVectors, trainingLabels);

	KNearest knn(trainingVectors, trainingLabels);

	//predictonTrainingSamples(knn, trainingVectors, trainingLabels, numImages);
	cvReleaseMat(&trainingVectors);
	cvReleaseMat(&trainingLabels);

	//Recognition: Using Knn

	//test number to be used
	numImages = 1000;//for tesing number
	CvMat *testVectors = 0;
	CvMat *actualLabels = 0;
	extractTestingData(numImages, testVectors,  actualLabels);

	CvMat *testLabels = cvCreateMat(numImages, 1, CV_32FC1);
	knn.find_nearest(testVectors, 5, testLabels);
	
	
	//Get the error rate and print the result out
	int totalCorrect = 0;
	for (int i = 0; i < numImages; i++){
		if (testLabels->data.fl[i] == actualLabels->data.fl[i])
		{
			totalCorrect++;
		}
		else{
			printf("\n Error: image id=%d number %f was mistaken as %f", i, actualLabels->data.fl[i], testLabels->data.fl[i]);
		}
			
	}
	printf("\nError Rate: %.1f%%", 
		(double)100- (double)totalCorrect * 100 / (double)numImages);

	cvReleaseMat(&testVectors);
	cvReleaseMat(&testLabels);
	cvReleaseMat(&actualLabels);
	return;



}
void Knnopencv::extractTestingData(int& numImages, CvMat*&testVectors, CvMat*& actualLabels)
{
	FILE *fp = fopen("D:\\baiducloud\\tech\\OpenCV\\basicOCR\\data\\mnist\\t10k-images-idx3-ubyte\\t10k-images.idx3-ubyte", "rb");
	FILE *fp2 = fopen("D:\\baiducloud\\tech\\OpenCV\\basicOCR\\data\\mnist\\t10k-labels-idx1-ubyte\\t10k-labels.idx1-ubyte", "rb");

	int magicNumber = readFlippedInteger(fp);
	int numImages2 = readFlippedInteger(fp);
	int numRows = readFlippedInteger(fp);

	int numCols = readFlippedInteger(fp);
	int size = numRows*numCols;

	fseek(fp2, 0x08, SEEK_SET);
	testVectors = cvCreateMat(numImages, size, CV_32FC1);
	actualLabels = cvCreateMat(numImages, 1, CV_32FC1);

	//create some temporary variables
	BYTE *temp = new BYTE[size];
	BYTE tempClass = 1;
	

	//due to time consideration, test only a portion of the test 
	for (int i = 0; i<numImages; i++)
	{

		fread((void*)temp, size, 1, fp);

		fread((void*)(&tempClass), sizeof(BYTE), 1, fp2);

		actualLabels->data.fl[i] = (float)tempClass;

		for (int k = 0; k<size; k++)
		{
			testVectors->data.fl[i*size + k] = temp[k];
		}

	}
	//release the file handle as all file data extraction are done
	fclose(fp);
	fclose(fp2);
	delete[] temp;

	getNumFrequency(actualLabels);
}

int Knnopencv::readFlippedInteger(FILE *fp)
{
	int ret = 0;

	BYTE *temp;

	temp = (BYTE*)(&ret);
	fread(&temp[3], sizeof(BYTE), 1, fp);
	fread(&temp[2], sizeof(BYTE), 1, fp);
	fread(&temp[1], sizeof(BYTE), 1, fp);

	fread(&temp[0], sizeof(BYTE), 1, fp);
	return ret;
}
Knnopencv::~Knnopencv()
{
}
