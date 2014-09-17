#include "Knntest.h"

using namespace cv;
using namespace std;
void cvDoubleMatPrint(const CvMat* mat)
{
	int i, j;
	for (i = 0; i < mat->rows; i++)
	{
		printf("row %d: ", i);
		for (j = 0; j < mat->cols; j++)
		{
			printf("%f, ", cvmGet(mat, i, j));
		}
		printf("\n");
	}
}
Knntest::Knntest()
{
}

void Knntest::test()
{
	/*float trainingData[10][2];
	srand(time(0));
	for (int i = 0; i<5; i++){
		trainingData[i][0] = rand() % 255 + 1;
		trainingData[i][1] = rand() % 255 + 1;
		trainingData[i + 5][0] = rand() % 255 + 255;
		trainingData[i + 5][1] = rand() % 255 + 255;
	}
	Mat trainingDataMat(10, 2, CV_32FC1, trainingData);
	float labels[10] = { 0, 0, 0, 0, 0, 1, 1, 1, 1, 1 };
	Mat labelsMat(10, 1, CV_32FC1, labels);*/
	const int K = 2;

	float trainingData[] = { 3, 104,

		2, 100,

		1, 81, 
		101, 10,
		99,5,
		98,2 };

	float labels[] = { 1, 1, 1, 2, 2, 2 };



	CvMat trainingDataMat = cvMat(6, 2, CV_32FC1, trainingData);
	CvMat labelsMat = cvMat(6, 1, CV_32FC1, labels);

	cvDoubleMatPrint(&trainingDataMat);
	cvDoubleMatPrint(&labelsMat);
	
	CvKNearest knn(&trainingDataMat, &labelsMat, 0, false, 10);

	float sampleData[] = { 98, 2};
	CvMat samplesMat = cvMat(1, 2, CV_32FC1, sampleData);

	//CvMat* nearests = cvCreateMat(1, K, CV_32SC1);

	float response = knn.find_nearest(&samplesMat, K, 0, 0, 0, 0);
	printf("%f", response); 
	cvWaitKey(0);

}


void cvInitizlizeMatwithArr(CvMat* mat, void *arr)
{
	int i, j;
	float *arrf = (float *)arr;
	for (i = 0; i < mat->rows; i++)
	{
		for (j = 0; j < mat->cols; j++)
		{
			cvmSet(mat, i, j, arrf[i*mat->width + j]);
		}
	}
}
void Knntest::test2()
{
	
	const int K = 10;
	int i, j, k, accuracy;
	float response;
	int train_sample_count = 100;
	CvRNG rng_state = cvRNG(-1);
	CvMat* trainData = cvCreateMat(train_sample_count, 2, CV_32FC1);
	CvMat* trainClasses = cvCreateMat(train_sample_count, 1, CV_32FC1);
	IplImage* img = cvCreateImage(cvSize(500, 500), 8, 3);
	float _sample[2];
	CvMat sample = cvMat(1, 2, CV_32FC1, _sample);
	cvZero(img);

	CvMat trainData1, trainData2, trainClasses1, trainClasses2;

	// form the training samples
	cvGetRows(trainData, &trainData1, 0, train_sample_count / 2);
	/*cvRandArr(&rng_state, &trainData1, CV_RAND_NORMAL, cvScalar(200), cvScalar(50));*/
	cvRandArr(&rng_state, &trainData1, CV_RAND_NORMAL, cvScalar(20), cvScalar(5));

	cvGetRows(trainData, &trainData2, train_sample_count / 2, train_sample_count);
	cvRandArr(&rng_state, &trainData2, CV_RAND_NORMAL, cvScalar(300), cvScalar(50));

	cvGetRows(trainClasses, &trainClasses1, 0, train_sample_count / 2);
	cvSet(&trainClasses1, cvScalar(1));

	cvGetRows(trainClasses, &trainClasses2, train_sample_count / 2, train_sample_count);
	cvSet(&trainClasses2, cvScalar(2));

	cvDoubleMatPrint(trainData);
	cvDoubleMatPrint(trainClasses);

	// learn classifier
	CvKNearest knn(trainData, trainClasses, 0, false, K);
	CvMat* nearests = cvCreateMat(1, K, CV_32FC1);

	for (i = 0; i < img->height; i++)
	{
		for (j = 0; j < img->width; j++)
		{
			sample.data.fl[0] = (float)j;
			sample.data.fl[1] = (float)i;

			// estimate the response and get the neighbors' labels
			response = knn.find_nearest(&sample, K, 0, 0, nearests, 0);
			/*printf("i=%d, j=%d, result= %f", i, j, response);
			cvDoubleMatPrint(nearests);
*/
			// compute the number of neighbors representing the majority
			for (k = 0, accuracy = 0; k < K; k++)
			{
				if (nearests->data.fl[k] == response)
					accuracy++;
			}
			// highlight the pixel depending on the accuracy (or confidence)
			cvSet2D(img, i, j, response == 1 ?
				(accuracy > 5 ? CV_RGB(180, 0, 0) : CV_RGB(180, 120, 0)) :
				(accuracy > 5 ? CV_RGB(0, 180, 0) : CV_RGB(120, 120, 0)));
		}
	}

	// display the original training samples
	for (i = 0; i < train_sample_count / 2; i++)
	{
		CvPoint pt;
		pt.x = cvRound(trainData1.data.fl[i * 2]);
		pt.y = cvRound(trainData1.data.fl[i * 2 + 1]);
		cvCircle(img, pt, 5, CV_RGB(255, 0, 0), CV_FILLED);
		pt.x = cvRound(trainData2.data.fl[i * 2]);
		pt.y = cvRound(trainData2.data.fl[i * 2 + 1]);
		cvCircle(img, pt, 2, CV_RGB(0, 255, 0), CV_FILLED);
	}

	cvNamedWindow("classifier result", 1);
	cvShowImage("classifier result", img);
	cvWaitKey(0);

	cvReleaseMat(&trainClasses);
	cvReleaseMat(&trainData);

}

void Knntest::test3()
{
	const int K = 2;
	int i, j, k, accuracy;
	float response;
	int train_sample_count = 6;
	CvRNG rng_state = cvRNG(-1);
	CvMat* trainData = cvCreateMat(train_sample_count, 2, CV_32FC1);
	CvMat* trainClasses = cvCreateMat(train_sample_count, 1, CV_32FC1);
	IplImage* img = cvCreateImage(cvSize(500, 500), 8, 3);
	float _sample[2];
	CvMat sample = cvMat(1, 2, CV_32FC1, _sample);
	cvZero(img);

	//CvMat trainData1, trainData2, trainClasses1, trainClasses2;

	// form the training samples
	/*cvGetRows(trainData, &trainData1, 0, train_sample_count / 2);
	cvRandArr(&rng_state, &trainData1, CV_RAND_NORMAL, cvScalar(20), cvScalar(5));

	cvGetRows(trainData, &trainData2, train_sample_count / 2, train_sample_count);
	cvRandArr(&rng_state, &trainData2, CV_RAND_NORMAL, cvScalar(300), cvScalar(50));

	cvGetRows(trainClasses, &trainClasses1, 0, train_sample_count / 2);
	cvSet(&trainClasses1, cvScalar(1));

	cvGetRows(trainClasses, &trainClasses2, train_sample_count / 2, train_sample_count);
	cvSet(&trainClasses2, cvScalar(2));*/



	float trainingDataar[] = { 3, 104,

		2, 100,

		1, 81,
		101, 10,
		99, 5,
		98, 2 };

	float labelsarr[] = { 1, 1, 1, 2, 2, 2 };

	cvInitizlizeMatwithArr(trainData, trainingDataar);
	

	cvDoubleMatPrint(trainData);
	cvInitizlizeMatwithArr(trainClasses, labelsarr);
	cvDoubleMatPrint(trainClasses);

	// learn classifier
	CvKNearest knn(trainData, trainClasses, 0, false, K);
	CvMat* nearests = cvCreateMat(1, K, CV_32FC1);

	for (i = 0; i < img->height; i++)
	{
		for (j = 0; j < img->width; j++)
		{
			sample.data.fl[0] = (float)j;
			sample.data.fl[1] = (float)i;

			// estimate the response and get the neighbors' labels
			response = knn.find_nearest(&sample, K, 0, 0, nearests, 0);
			/*printf("i=%d, j=%d, result= %f", i, j, response);
			cvDoubleMatPrint(nearests);
			*/
			// compute the number of neighbors representing the majority
			for (k = 0, accuracy = 0; k < K; k++)
			{
				if (nearests->data.fl[k] == response)
					accuracy++;
			}
			// highlight the pixel depending on the accuracy (or confidence)
			cvSet2D(img, i, j, response == 1 ?
				(accuracy > 5 ? CV_RGB(180, 0, 0) : CV_RGB(180, 120, 0)) :
				(accuracy > 5 ? CV_RGB(0, 180, 0) : CV_RGB(120, 120, 0)));
		}
	}

	// display the original training samples
	//for (i = 0; i < train_sample_count; i++)
	//{
	//	/*CvPoint pt;
	//	pt.x = cvRound(trainData1.data.fl[i * 2]);
	//	pt.y = cvRound(trainData1.data.fl[i * 2 + 1]);
	//	cvCircle(img, pt, 5, CV_RGB(255, 0, 0), CV_FILLED);
	//	pt.x = cvRound(trainData2.data.fl[i * 2]);
	//	pt.y = cvRound(trainData2.data.fl[i * 2 + 1]);
	//	cvCircle(img, pt, 2, CV_RGB(0, 255, 0), CV_FILLED);*/


	//}

	for (i = 0; i < trainData->rows; i++)
	{
		float cls = cvmGet(trainClasses, i, 0);
			CvPoint pt;
			pt.x = cvRound(cvmGet(trainData, i, 0));
			pt.y = cvRound(cvmGet(trainData, i, 1));
			
		if (cls == 1){
			cvCircle(img, pt, 5, CV_RGB(255, 0, 0), CV_FILLED);
		}
		else
		{
			cvCircle(img, pt, 2, CV_RGB(0, 255, 0), CV_FILLED);
		}
	}

	sample.data.fl[0] = (float)50;
	sample.data.fl[1] = (float)300;
	response = knn.find_nearest(&sample, K, 0, 0, nearests, 0);
	printf("res=%f", response);

	CvPoint pt;
	pt.x = cvRound(cvmGet(&sample, 0, 0));
	pt.y = cvRound(cvmGet(&sample, 0, 1));
	cvCircle(img, pt, 10, CV_RGB(255, 250, 0), CV_FILLED);



	cvNamedWindow("classifier result", 1);
	cvShowImage("classifier result", img);
	cvWaitKey(0);

	cvReleaseMat(&trainClasses);
	cvReleaseMat(&trainData);

}


Knntest::~Knntest()
{
}
