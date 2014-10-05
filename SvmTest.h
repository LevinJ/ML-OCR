#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/ml/ml.hpp>
#include <stdio.h>
#include <ctype.h>

using namespace cv;
class SvmTest
{
public:
	SvmTest();
	void test();
	~SvmTest();
};

