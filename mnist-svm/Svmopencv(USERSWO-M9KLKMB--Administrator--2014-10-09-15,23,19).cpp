#include "Svmopencv.h"


Svmopencv::Svmopencv()
{
}

int Svmopencv::readFlippedInteger(FILE *fp)
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
void Svmopencv::test()
{
	FILE *fp = fopen("D:\\baiducloud\\tech\\OpenCV\\basicOCR\\data\\mnist\\train-images-idx3-ubyte\\train-images.idx3-ubyte", "rb");

	FILE *fp2 = fopen("D:\\baiducloud\\tech\\OpenCV\\basicOCR\\data\\mnist\\train-labels-idx1-ubyte\\train-labels.idx1-ubyte", "rb");

	int magicNumber = readFlippedInteger(fp);
	int numImages = readFlippedInteger(fp);

	int numRows = readFlippedInteger(fp);

	int numCols = readFlippedInteger(fp);

	fseek(fp2, 0x08, SEEK_SET);

	//store all the images and labels
	int size = numRows*numCols;
	CvMat *trainingVectors = cvCreateMat(numImages, size, CV_32FC1);
	CvMat *trainingLabels = cvCreateMat(numImages, 1, CV_32FC1);
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


	KNearest knn(trainingVectors, trainingLabels);

	printf("Maximum k: %d", knn.get_max_k());
	fclose(fp);
	fclose(fp2);
	cvReleaseMat(&trainingVectors);
	cvReleaseMat(&trainingLabels);
	//Recognition: Using K-Nearest Neighbors

	fp = fopen("D:\\baiducloud\\tech\\OpenCV\\basicOCR\\data\\mnist\\t10k-images-idx3-ubyte\\t10k-images.idx3-ubyte", "rb");
	fp2 = fopen("D:\\baiducloud\\tech\\OpenCV\\basicOCR\\data\\mnist\\t10k-labels-idx1-ubyte\\t10k-labels.idx1-ubyte", "rb");

	magicNumber = readFlippedInteger(fp);
	numImages = readFlippedInteger(fp);
	numRows = readFlippedInteger(fp);

	numCols = readFlippedInteger(fp);

	fseek(fp2, 0x08, SEEK_SET);

	numImages = 200;
	CvMat *testVectors = cvCreateMat(numImages, size, CV_32FC1);
	CvMat *testLabels = cvCreateMat(numImages, 1, CV_32FC1);
	CvMat *actualLabels = cvCreateMat(numImages, 1, CV_32FC1);

	//create some temporary variables
	temp = new BYTE[size];
	tempClass = 1;
	//CvMat *currentTest = cvCreateMat(1, size, CV_32FC1);
	//CvMat *currentLabel = cvCreateMat(1, 1, CV_32FC1);
	int totalCorrect = 0;

	//due to time consideration, test only a portion of the test 
	
	for (int i = 0; i<numImages; i++)
	{

		fread((void*)temp, size, 1, fp);

		fread((void*)(&tempClass), sizeof(BYTE), 1, fp2);

		actualLabels->data.fl[i] = (float)tempClass;

		for (int k = 0; k<size; k++)
		{
			testVectors->data.fl[i*size + k] = temp[k];
			//currentTest->data.fl[k] = temp[k];
		}

		/*knn.find_nearest(currentTest, 5, currentLabel);

		testLabels->data.fl[i] = currentLabel->data.fl[0];

		if (currentLabel->data.fl[0] == actualLabels->data.fl[i])
			totalCorrect++;*/
	}

	knn.find_nearest(testVectors, 5, testLabels);
	
	//Get the correction rate
	for (int i = 0; i < numImages; i++){
		if (testLabels->data.fl[i] == actualLabels->data.fl[i])
		{
			totalCorrect++;
			//printf("\n error: number %f as right with %f", actualLabels->data.fl[i], testLabels->data.fl[i]);
		}
		else{
			printf("\n error: test id=%d number %f was mistaken as %f", actualLabels->data.fl[i], testLabels->data.fl[i]);
		}
			
	}

	printf("\nTime: %d Accuracy: %f ", (int)time, (double)totalCorrect * 100 / (double)numImages);

	cvWaitKey(0);
	return;



}
Svmopencv::~Svmopencv()
{
}
