#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#include <ctype.h>
#include "basicOCR.h"
#include "Knntest.h"
#include "SvmTest.h"
#include "mnist-knn/Knnopencv.h"
#include "mnist-svm/Svmopencv.h"
#include "mnist-decisiontree/DTopencv.h"
#include "mnist-randomforest/Randomforestopencv.h"
#include "mnist-naivebayes/Naivebayesopencv.h"
#include "mnist-nn/NNopencv.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	//Knnopencv obj;
	clock_t start;
	double diff;
	start = clock();


	//Svmopencv obj;
	NNopencv obj;
	//DTopencv obj;
	//RandomForestopencv obj;
	//Knnopencv obj;
	//Naivebayesopencv obj;
	obj.test();
	diff = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	printf("\nOverall Duration:%.0f(Seconds)", diff);

	int wait;
	scanf("%d", &wait);

	return 0;
}
