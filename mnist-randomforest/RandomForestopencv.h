#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/ml/ml.hpp>
#include <stdio.h>
#include <ctype.h>
//#include <stdlib.h>
#include <ctime>
#include "../testBase.h"

typedef unsigned char       BYTE;
using namespace cv;
using namespace std;


class RandomForestopencv: public testBase
{
public:
	void extractTrainingData(int& numImages, CvMat*&trainingVectors, CvMat*& trainingLabels);
	void extractTestingData(int& numImages, CvMat*&testVectors, CvMat*& actualLabels);
	int readFlippedInteger(FILE *fp);
	CvRTrees* mnist_create_RandomForestree(const CvMat* data, const CvMat* missing,
		const CvMat* responses, float p_weight);
	void print_variable_importance(CvRTrees* RandomForestree);
	void test();
	RandomForestopencv();
	~RandomForestopencv();
};

