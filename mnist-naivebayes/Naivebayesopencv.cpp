#include "Naivebayesopencv.h"
using namespace cv;
using namespace std;

#include <fstream>
#include <sstream>

Naivebayesopencv::Naivebayesopencv()
{
}
///////////////////////
// Functions
static void read_imgList(const string& filename, vector<Mat>& images) {
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}
	string line;
	while (getline(file, line)) {
		images.push_back(imread(line, 0));
	}
}

static  Mat formatImagesForPCA(const vector<Mat> &data)
{
	Mat dst(static_cast<int>(data.size()), data[0].rows*data[0].cols, CV_32F);
	for (unsigned int i = 0; i < data.size(); i++)
	{
		Mat image_row = data[i].clone().reshape(1, 1);
		Mat row_i = dst.row(i);
		image_row.convertTo(row_i, CV_32F);
	}
	return dst;
}

static Mat toGrayscale(InputArray _src) {
	Mat src = _src.getMat();
	// only allow one channel
	if (src.channels() != 1) {
		CV_Error(CV_StsBadArg, "Only Matrices with one channel are supported");
	}
	// create and return normalized image
	Mat dst;
	cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
	return dst;
}

struct params
{
	Mat data;
	int ch;
	int rows;
	PCA pca;
	string winName;
};

static void onTrackbar(int pos, void* ptr)
{
	cout << "Retained Variance = " << pos << "%   ";
	cout << "re-calculating PCA..." << std::flush;

	double var = pos / 100.0;

	struct params *p = (struct params *)ptr;

	p->pca = PCA(p->data, cv::Mat(), CV_PCA_DATA_AS_ROW, var);

	Mat point = p->pca.project(p->data.row(0));
	Mat reconstruction = p->pca.backProject(point);
	reconstruction = reconstruction.reshape(p->ch, p->rows);
	reconstruction = toGrayscale(reconstruction);

	imshow(p->winName, reconstruction);
	cout << "done!   # of principal components: " << p->pca.eigenvectors.rows << endl;
}
void Naivebayesopencv::testPCA(){
	
	/*if (argc != 2) {
		cout << "usage: " << argv[0] << " <image_list.txt>" << endl;
		exit(1);
	}*/

	// Get the path to your CSV.
	string imgList = "data/att_faces/image_list.txt";

	// vector to hold the images
	vector<Mat> images;

	// Read in the data. This can fail if not valid
	try {
		read_imgList(imgList, images);
	}
	catch (cv::Exception& e) {
		cerr << "Error opening file \"" << imgList << "\". Reason: " << e.msg << endl;
		exit(1);
	}

	// Quit if there are not enough images for this demo.
	if (images.size() <= 1) {
		string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
		CV_Error(CV_StsError, error_message);
	}

	// Reshape and stack images into a rowMatrix
	Mat data = formatImagesForPCA(images);

	// perform PCA
	PCA pca(data, cv::Mat(), CV_PCA_DATA_AS_ROW, 0.95); // trackbar is initially set here, also this is a common value for retainedVariance

	// Demonstration of the effect of retainedVariance on the first image
	Mat point = pca.project(data.row(0)); // project into the eigenspace, thus the image becomes a "point"
	Mat reconstruction = pca.backProject(point); // re-create the image from the "point"
	reconstruction = reconstruction.reshape(images[0].channels(), images[0].rows); // reshape from a row vector into image shape
	reconstruction = toGrayscale(reconstruction); // re-scale for displaying purposes

	// init highgui window
	string winName = "Reconstruction | press 'q' to quit";
	namedWindow(winName, WINDOW_NORMAL);

	// params struct to pass to the trackbar handler
	params p;
	p.data = data;
	p.ch = images[0].channels();
	p.rows = images[0].rows;
	p.pca = pca;
	p.winName = winName;

	// create the tracbar
	int pos = 95;
	createTrackbar("Retained Variance (%)", winName, &pos, 100, onTrackbar, (void*)&p);

	// display until user presses q
	imshow(winName, reconstruction);

	int key = 0;
	while (key != 'q')
		key = waitKey();
}

void Naivebayesopencv::extractTrainingData(int& numImages, CvMat *& trainingVectors, CvMat*& trainingLabels)
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


void Naivebayesopencv::test()
{
	//testPCA();
	//number of taining samples to be used
	int numImages = 1000;
	CvMat *trainingVectors = 0;
	CvMat *trainingLabels = 0;
	extractTrainingData(numImages, trainingVectors, trainingLabels);
	//train the data with Naivebayes
	
	// Perform a PCA:
	
	PCA pca(Mat(trainingVectors), Mat(), CV_PCA_DATA_AS_ROW, 300);

	Mat newTraining = pca.project(Mat(trainingVectors));

	CvNormalBayesClassifier Naivebayes;
	Naivebayes.train(newTraining, trainingLabels, Mat(), Mat(), false);
	/*Naivebayes.train(trainingVectors, trainingLabels, Mat(), Mat(), false);*/


	cvReleaseMat(&trainingVectors);
	cvReleaseMat(&trainingLabels);

	//Recognition: Using Naivebayes

	//test number to be used
	numImages = 1000;//for tesing number
	CvMat *testVectors = 0;
	CvMat *actualLabels = 0;
	extractTestingData(numImages, testVectors,  actualLabels);

	CvMat *testLabels = cvCreateMat(numImages, 1, CV_32FC1);

	PCA pca2(Mat(testVectors), Mat(), CV_PCA_DATA_AS_ROW, 300);

	Mat newTesignVector = pca2.project(Mat(testVectors));

	CvMat cvMatnewTestingvector = newTesignVector;
	Naivebayes.predict(&cvMatnewTestingvector, testLabels);
	//Naivebayes.predict(testVectors, testLabels);

	
	
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
void Naivebayesopencv::extractTestingData(int& numImages, CvMat*&testVectors, CvMat*& actualLabels)
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

int Naivebayesopencv::readFlippedInteger(FILE *fp)
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
Naivebayesopencv::~Naivebayesopencv()
{
}
