#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/ml/ml.hpp>
#include <stdio.h>
#include <ctype.h>
//#include <stdlib.h>
#include <ctime>

typedef unsigned char       BYTE;
using namespace cv;
using namespace std;


class Knnopencv
{
public:
	void extractTrainingData(int& numImages, CvMat*&trainingVectors, CvMat*& trainingLabels);
	void extractTestingData(int& numImages, CvMat*&testVectors, CvMat*& actualLabels);
	int readFlippedInteger(FILE *fp);
	void test();
	void  getNumFrequency(CvMat*& labels);
	Knnopencv();
	~Knnopencv();
};

