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
	printf("This application is designed to compare and analyze diffrent learning algorithms on MNIST data set\n1 for SVM\n2 for ANN\n3 for KNN\n4 for decision tree\n5 for random forest\n");
	clock_t start;
	double diff;
	start = clock();

	
	testBase *mlTest = 0;
	
	int choice;
	scanf("%d", &choice);

	switch (choice)
	{
		case 1:
			mlTest = new Svmopencv();
			break;
		case 2:
			mlTest = new NNopencv();
			break;
		case 3:
			mlTest = new Knnopencv();
			break;
		case 4:
			mlTest = new DTopencv();
			break;
		case 5:
			mlTest = new RandomForestopencv();
			break;

		default:
			break;
	}
	mlTest->test();
	diff = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	printf("\nOverall Duration:%.0f(Seconds)", diff);

	int wait;
	scanf("%d", &wait);
	delete mlTest;

	return 0;
}
