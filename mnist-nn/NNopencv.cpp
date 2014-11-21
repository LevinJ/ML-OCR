#include "NNopencv.h"


NNopencv::NNopencv()
{
}
void NNopencv::extractTrainingData(int& numImages, CvMat *& trainingVectors, CvMat*& trainingLabels)
{
	FILE *fp = fopen("D:\\baiducloud\\tech\\OpenCV\\basicOCR\\data\\mnist\\train-images-idx3-ubyte\\train-images.idx3-ubyte", "rb");

	FILE *fp2 = fopen("D:\\baiducloud\\tech\\OpenCV\\basicOCR\\data\\mnist\\train-labels-idx1-ubyte\\train-labels.idx1-ubyte", "rb");

	int magicNumber = readFlippedInteger(fp);
	int numImages2 = readFlippedInteger(fp);

	int numRows = readFlippedInteger(fp);

	int numCols = readFlippedInteger(fp);

	fseek(fp2, 0x08, SEEK_SET);

	//store all the images and labels
	int number_of_classes = 10;
	int size = numRows*numCols;
	trainingVectors = cvCreateMat(numImages, size, CV_32FC1);
	trainingLabels = cvCreateMat(numImages, number_of_classes, CV_32FC1);
	cvZero(trainingLabels);
	//with memory in place, we read data from the files
	BYTE *temp = new BYTE[size];
	BYTE tempClass = 0;
	for (int i = 0; i<numImages; i++)
	{

		fread((void*)temp, size, 1, fp);

		fread((void*)(&tempClass), sizeof(BYTE), 1, fp2);

		//trainingLabels->data.fl[i] = tempClass;
		CV_MAT_ELEM(*trainingLabels, float, i, tempClass) = 1.0;

		for (int k = 0; k<size; k++)
			trainingVectors->data.fl[i*size + k] = temp[k];
	}

	//Data has been stored into the opencv matrix, it's ok to relase the file handler now
	fclose(fp);
	fclose(fp2);
	delete[] temp;

}
void predictonTrainingSamples(NNopencv * nn, CvANN_MLP& Networks, CvMat *& testVectors, 
	 CvMat *& actualLabels, int numImages){

	CvMat *testLabels = cvCreateMat(numImages, 10, CV_32FC1);
	nn->NeuralNetworksPredict(Networks, testVectors, testLabels);


	//Get the error rate and print the result out
	int totalCorrect = 0;
	float tlabel = 0;
	for (int i = 0; i < numImages; i++){
		//find the digit that has been identified by NN
		float maxValue = 0;
		for (int j = 0; j < 10; j++){
			float jclass = CV_MAT_ELEM(*testLabels, float, i, j);
			if (jclass > maxValue){
				maxValue = jclass;
				tlabel = j;
			}
		}
		//tlabel is now the digit identified
		/*if (testLabels->data.fl[i] == actualLabels->data.fl[i])*/
		if (tlabel == actualLabels->data.fl[i])
		{
			totalCorrect++;
		}
		else{
			printf("\n Error: image id=%d number %f was mistaken as %f", i, actualLabels->data.fl[i], tlabel);
		}

	}
	printf("\nError Rate on training samples: %.1f%%",
		(double)100 - (double)totalCorrect * 100 / (double)numImages);
}
void NNopencv::test()
{
	//number of taining samples to be used
	int numImages = 6000;
	CvMat *trainingVectors = 0;
	CvMat *trainingLabels = 0;
	extractTrainingData(numImages, trainingVectors, trainingLabels);
	
	// defining the network  
	CvANN_MLP Networks;
	// The number of iteration  
	int MaxIte = 2;
	NeuralNetworksTraing(Networks, trainingVectors, trainingLabels,MaxIte);
	// save the networks  
	Networks.save("NerualNetworks-ite=2-1000hidden.xml");

	/*predictonTrainingSamples(this, Networks, trainingVectors,
		trainingLabels, numImages);*/
	cvReleaseMat(&trainingVectors);
	cvReleaseMat(&trainingLabels);

	//Recognition: Using NN

	numImages = 1000;//for tesing number
	CvMat *testVectors = 0;
	CvMat *actualLabels = 0;
	extractTestingData(numImages, testVectors, actualLabels);
	

	CvMat *testLabels = cvCreateMat(numImages, 10, CV_32FC1);
	
	NeuralNetworksPredict(Networks, testVectors, testLabels);
	/*NeuralNetworksPredict(Networks, trainingVectors, testLabels);
	actualLabels = trainingLabels;*/
	
	
	//Get the error rate and print the result out
	int totalCorrect = 0;
	float tlabel = 0;
	for (int i = 0; i < numImages; i++){
		//find the digit that has been identified by NN
		float maxValue = 0;	
		for (int j = 0; j < 10; j++){
			float jclass = CV_MAT_ELEM(*testLabels, float, i, j);
			if (jclass > maxValue){
				maxValue = jclass;
				tlabel = j;
			}
		}
		//tlabel is now the digit identified
		/*if (testLabels->data.fl[i] == actualLabels->data.fl[i])*/
		if (tlabel == actualLabels->data.fl[i])
		{
			totalCorrect++;
		}
		else{
			printf("\n Error: image id=%d number %f was mistaken as %f", i, actualLabels->data.fl[i], tlabel);
		}
			
	}
	printf("\nError Rate: %.1f%%", 
		(double)100- (double)totalCorrect * 100 / (double)numImages);

	cvReleaseMat(&testVectors);
	cvReleaseMat(&testLabels);
	cvReleaseMat(&actualLabels);
	return;



}
void NNopencv::extractTestingData(int& numImages, CvMat*&testVectors, CvMat*& actualLabels)
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

int NNopencv::readFlippedInteger(FILE *fp)
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
void NNopencv::NeuralNetworksTraing(CvANN_MLP& NeuralNetworks, const cv::Mat& InputMat,
	const cv::Mat& OutputMat, int MaxIte)
{
	// Network architecture  
	std::vector<int> LayerSizes;
	LayerSizes.push_back(InputMat.cols);    // input layer  
	LayerSizes.push_back(100);             // hidden layer has 1000 neurons  
	LayerSizes.push_back(OutputMat.cols);   // output layer  


	// Activate function  
	int ActivateFunc = CvANN_MLP::SIGMOID_SYM;
	double Alpha = 1;
	double Beta = 1;


	// create the network  
	NeuralNetworks.create(cv::Mat(LayerSizes), ActivateFunc, Alpha, Beta);



	// Training Params  
	CvANN_MLP_TrainParams TrainParams;
	TrainParams.train_method = CvANN_MLP_TrainParams::BACKPROP;
	TrainParams.bp_dw_scale = 0.0001;
	TrainParams.bp_moment_scale = 0;

	// iteration number  
	CvTermCriteria TermCrlt;
	TermCrlt.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;
	TermCrlt.epsilon = 0.0001f;
	TermCrlt.max_iter = MaxIte;
	TrainParams.term_crit = TermCrlt;


	// Training the networks  
	NeuralNetworks.train(InputMat, OutputMat, cv::Mat(), cv::Mat(), TrainParams);

}
/**
* @brief NeuralNetworksPredict
* @param NeuralNetworks
* @param Input
* @param Output
*/
void NNopencv::NeuralNetworksPredict(const CvANN_MLP& NeuralNetworks, CvMat *& Input,
	CvMat *& OutputVector)
{
	// Neural network predict  
	//cv::Mat OutputVector;
	NeuralNetworks.predict(Input, OutputVector);

	// change the output vector  
	//DecodeOutputVector(OutputVector, Output, OutputVector.cols);


}
NNopencv::~NNopencv()
{
}
