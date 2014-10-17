#include "DTopencv.h"


DTopencv::DTopencv()
{
}
void DTopencv::extractTrainingData(int& numImages, CvMat *& trainingVectors, CvMat*& trainingLabels)
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

}

CvDTree* DTopencv::mnist_create_dtree(const CvMat* data, const CvMat* missing,
	const CvMat* responses, float p_weight){
	CvDTree* dtree;
	CvMat* var_type;
	int i, hr1 = 0, hr2 = 0, p_total = 0;
	float priors[] = { 1, p_weight };

	var_type = cvCreateMat(data->cols + 1, 1, CV_8U);
	cvSet(var_type, cvScalarAll(CV_VAR_CATEGORICAL)); // all the variables are categorical

	dtree = new CvDTree;

	dtree->train(data, CV_ROW_SAMPLE, responses, 0, 0, var_type, missing,
		CvDTreeParams(8, // max depth
		10, // min sample count
		0, // regression accuracy: N/A here
		true, // compute surrogate split, as we have missing data
		15, // max number of categories (use sub-optimal algorithm for larger numbers)
		10, // the number of cross-validation folds
		true, // use 1SE rule => smaller tree
		true, // throw away the pruned tree branches
		priors // the array of priors, the bigger p_weight, the more attention
		// to the poisonous mushrooms
		// (a mushroom will be judjed to be poisonous with bigger chance)
		));

	// compute hit-rate on the training database, demonstrates predict usage.
	for (i = 0; i < data->rows; i++)
	{
		CvMat sample, mask;
		cvGetRow(data, &sample, i);
		cvGetRow(missing, &mask, i);
		double r = dtree->predict(&sample, &mask)->value;
		int d = fabs(r - responses->data.fl[i]) >= FLT_EPSILON;
		if (d)
		{
			if (r != 'p')
				hr1++;
			else
				hr2++;
		}
		p_total += responses->data.fl[i] == 'p';
	}

	printf("Results on the training database:\n"
		"\tPoisonous mushrooms mis-predicted: %d (%g%%)\n"
		"\tFalse-alarms: %d (%g%%)\n", hr1, (double)hr1 * 100 / p_total,
		hr2, (double)hr2 * 100 / (data->rows - p_total));

	cvReleaseMat(&var_type);

	return dtree;

}
void DTopencv::test()
{
	//number of taining samples to be used
	int numImages = 6000;
	CvMat *trainingVectors = 0;
	CvMat *trainingLabels = 0;
	extractTrainingData(numImages, trainingVectors, trainingLabels);
	//Do the training
	CvDTree* dtree;
	dtree = mnist_create_dtree(trainingVectors, 0, trainingLabels,0 );
	
	cvReleaseMat(&trainingVectors);
	cvReleaseMat(&trainingLabels);

	//Recognition: Using DT

	//test number to be used
	//numImages = 1000;//for tesing number
	//CvMat *testVectors = 0;
	//CvMat *actualLabels = 0;
	//extractTestingData(numImages, testVectors,  actualLabels);

	//CvMat *testLabels = cvCreateMat(numImages, 1, CV_32FC1);
	//DT.predict(testVectors, testLabels);
	//
	//
	////Get the error rate and print the result out
	//int totalCorrect = 0;
	//for (int i = 0; i < numImages; i++){
	//	if (testLabels->data.fl[i] == actualLabels->data.fl[i])
	//	{
	//		totalCorrect++;
	//	}
	//	else{
	//		printf("\n Error: image id=%d number %f was mistaken as %f", i, actualLabels->data.fl[i], testLabels->data.fl[i]);
	//	}
	//		
	//}
	//printf("\nError Rate: %.1f%%", 
	//	(double)100- (double)totalCorrect * 100 / (double)numImages);

	//cvReleaseMat(&testVectors);
	//cvReleaseMat(&testLabels);
	//cvReleaseMat(&actualLabels);
	return;



}
void DTopencv::extractTestingData(int& numImages, CvMat*&testVectors, CvMat*& actualLabels)
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
}

int DTopencv::readFlippedInteger(FILE *fp)
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
DTopencv::~DTopencv()
{
}
