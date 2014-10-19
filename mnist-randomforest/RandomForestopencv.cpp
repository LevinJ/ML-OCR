#include "RandomForestopencv.h"


RandomForestopencv::RandomForestopencv()
{
}
void RandomForestopencv::extractTrainingData(int& numImages, CvMat *& trainingVectors, CvMat*& trainingLabels)
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

CvRTrees* RandomForestopencv::mnist_create_RandomForestree(const CvMat* data, const CvMat* missing,
	const CvMat* responses, float p_weight){
	CvRTrees* RandomForestree;
	CvMat* var_type;
	int i, hr1 = 0, hr2 = 0, p_total = 0;
	/*float priors[] = { 1, p_weight };*/
	float *priors = 0;

	var_type = cvCreateMat(data->cols + 1, 1, CV_8U);
	cvSet(var_type, cvScalarAll(CV_VAR_CATEGORICAL)); // all the variables are categorical

	RandomForestree = new CvRTrees;

	RandomForestree->train(data, CV_ROW_SAMPLE, responses, 0, 0, var_type, missing,

		//CvRTParams(10, 10, 0, false, 15, 0, true, 4, 20, 0.01f, CV_TERMCRIT_ITER)
		CvRTParams(20,//the depth of the tree
		10,//minimum samples required at a leaf node for it to be split
		0, //regression_accuracy
		true, //use_surrogates
		2,//max_categories 
		0, //priors
		true,//calc_var_importance
		0, //nactive_vars
		20, //max_num_of_trees_in_the_forest
		0.01f,//forest_accuracy
		CV_TERMCRIT_ITER)
		);

	// compute hit-rate on the training database, demonstrates predict usage.
	for (i = 0; i < data->rows; i++)
	{
		CvMat sample, mask;
		cvGetRow(data, &sample, i);
		double r = RandomForestree->predict(&sample);
		if (responses->data.fl[i] != r)
		{
			hr1++;
		}
	}
	printf("\nSample Error Rate: %.1f%%", (double)hr1 * 100 / (double)data->rows);

	cvReleaseMat(&var_type);

	return RandomForestree;

}
void RandomForestopencv::test()
{
	//number of taining samples to be used
	int numImages = 6000;
	CvMat *trainingVectors = 0;
	CvMat *trainingLabels = 0;
	extractTrainingData(numImages, trainingVectors, trainingLabels);
	//Do the training
	CvRTrees* RandomForestree;
	RandomForestree = mnist_create_RandomForestree(trainingVectors, 0, trainingLabels,0 );
	
	cvReleaseMat(&trainingVectors);
	cvReleaseMat(&trainingLabels);

	//Recognition: Using RandomForest

	//test number to be used
	numImages = 1000;//for tesing number
	CvMat *testVectors = 0;
	CvMat *actualLabels = 0;
	extractTestingData(numImages, testVectors,  actualLabels);
	int err = 0;
	for (int i = 0; i < testVectors->rows; i++)
	{
		CvMat sample;
		cvGetRow(testVectors, &sample, i);
		double r = RandomForestree->predict(&sample);
		if (actualLabels->data.fl[i] != r)
		{
			err++;
		}
	}
	printf("\nError Rate: %.1f%%", (double)err * 100 / (double)testVectors->rows);

	//print_variable_importance(RandomForestree);
	delete RandomForestree;
	cvReleaseMat(&testVectors);
	cvReleaseMat(&actualLabels);
	return;



}
void RandomForestopencv::extractTestingData(int& numImages, CvMat*&testVectors, CvMat*& actualLabels)
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

int RandomForestopencv::readFlippedInteger(FILE *fp)
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

void RandomForestopencv::print_variable_importance(CvRTrees * RandomForestree)
{
	//const CvMat* var_importance = RandomForestree->get_var_importance();
	Mat var_importance = RandomForestree->getVarImportance();
	int i;
	char input[1000];
	Mat importancesort;

	if (var_importance.cols == 0)
	{
		printf("Error: Variable importance can not be retrieved\n");
		return;
	}
	sortIdx(var_importance, importancesort, CV_SORT_EVERY_ROW + CV_SORT_DESCENDING);

	double var_improtatacne_total = 0;

	for (int i = 0; i < importancesort.rows; i++){
		for (int j = 0; j < 80; j ++ ){
			int index = importancesort.at<int>(i, j);
			double val = var_importance.at<double>(0, index);
			printf("Bit %d, %d: %g%%\n", index / 28 + 1, index % 28, val*100.);
		}
	}
}
RandomForestopencv::~RandomForestopencv()
{
}
