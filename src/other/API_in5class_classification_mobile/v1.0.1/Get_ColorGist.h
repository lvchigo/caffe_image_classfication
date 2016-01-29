#pragma once
#include <cv.h>
#include <highgui.h>
#include <cxcore.h>
#include <ml.h>
#include "cvaux.h"

using namespace std;
using namespace cv;

namespace ImageTypeAJudge_2_2_0
{
	Mat create_filter(const int fc, const int width, const int height);
	vector<Mat> create_gabor(const int nscales, const int *orientations_per_scale, const int width, const int height);
#if 1	
	void color_prefilt(Mat& src, Mat& gfc, const int padding);
	void color_gist_gabor(Mat src, const int w, const vector<Mat>& G,float *res);
#else
	void prefilt(Mat& src, Mat& gfc, const int padding);
	void gist_gabor(Mat src, const int w, const vector<Mat>& G,float *res);
#endif
}