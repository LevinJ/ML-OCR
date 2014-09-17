/*
 *  preprocessing.h
 *  
 *
 *  Created by damiles on 18/11/08.
 *  Copyright 2008 __MyCompanyName__. All rights reserved.
 *
 */
#ifdef _CH_
#pragma package <opencv>
#endif

#ifndef _EiC
//#include <cv.h>
//#include <highgui.h>
//#include <ml.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/ml/ml.hpp>
#include <stdio.h>
#include <ctype.h>
#endif

class basicOCR{
	public:
		float classify(IplImage* img,int showResult);
		basicOCR ();
		void test();	
	private:
		char file_path[255];
		int train_samples;
		int classes;
		CvMat* trainData;
		CvMat* trainClasses;
		int size;
		static const int K=10;
		CvKNearest *knn;
		void getData();
		void train();
};
