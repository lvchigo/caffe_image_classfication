#pragma once
//#include <cuda_runtime.h>
//#include <google/protobuf/text_format.h>
#include <queue>  // for std::priority_queue
#include <utility>  // for pair

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <dirent.h>
#include <unistd.h>

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>
#include <opencv/ml.h>
#include <opencv/cvaux.h>

#include "API_fast_segment/API_fast_segment_mobile2.h"
#include <cmath>

using namespace cv;
using namespace std;

/***********************************Init*************************************/
/// construct function 
API_FAST_SEGMENT::API_FAST_SEGMENT()
{
}

/// destruct function 
API_FAST_SEGMENT::~API_FAST_SEGMENT(void)
{
}

int API_FAST_SEGMENT::do_fast_segment(
		uchar* 	bgr, 				//[In]:image->bgr
		string  iid,
		int 	width, 				//[In]:image->width
		int 	height,				//[In]:image->height
		int 	channel,			//[In]:image->channel
		int*	rect)				//[Out]:Res:<x1,y1,x2,y2>
{
	if( !bgr || (width<16) || (height<16) || (channel != 3) ) 
	{	
		cout<<"image err!!" << endl;
		return -1;
	}	

	/*****************************Init*****************************/
	int w,h,nRet = 0;
	int step = 3*width;
	int graySize = width*height;

	vector<unsigned int> imgInput;
	//IplImage to vector
	for ( h=0;h<height;h++) {
		for ( w=0;w<width;w++) {
			unsigned int t=0;
			t += (uchar)(bgr[h*step + 3*w +2]);
			t<<=8;
			t += (uchar)(bgr[h*step + 3*w +1]);
			t<<=8;
			t += (uchar)(bgr[h*step + 3*w]);
			imgInput.push_back(t);
		}
	}

	/************************ImageStretchByHistogram*****************************/
	vector<int> saliencyMap;
	GetSaliencyMap( imgInput, width, height, saliencyMap);

	//vector to IplImage
	uchar *imgSaliency = new uchar[graySize];
	memset(imgSaliency, 0, graySize * sizeof(uchar));
	IplImage *imgGray = cvCreateImage(cvSize(width,height), IPL_DEPTH_8U, 1);
	for (h=0;h<height;h++) {
		for (w=0;w<width;w++) {
            ((uchar *)(imgGray->imageData + h*width))[w] = saliencyMap[h*width + w];
			imgSaliency[h*width + w] = saliencyMap[h*width + w];
        }
    }

	char szImgPath[1024];
	sprintf(szImgPath, "res/gray/%s.jpg", iid.c_str() );
	cvSaveImage(szImgPath, imgGray);

	//FindMaxContour
	nRet = FindMaxContour( imgSaliency, width, height, 1, rect );
	
	delete imgSaliency;
	cvReleaseImage(&imgGray);imgGray = 0;
	
	return nRet;
}


//===========================================================================
///	RGB2LAB
//===========================================================================
void API_FAST_SEGMENT::RGB2LAB(
	const vector<unsigned int>&		ubuff,
	vector<double>&					lvec,
	vector<double>&					avec,
	vector<double>&					bvec)
{
	int sz = ubuff.size();
	lvec.resize(sz);
	avec.resize(sz);
	bvec.resize(sz);

	for( int j = 0; j < sz; j++ )
	{
		int r = (ubuff[j] >> 16) & 0xFF;
		int g = (ubuff[j] >>  8) & 0xFF;
		int b = (ubuff[j]      ) & 0xFF;

		double xval = 0.412453 * r + 0.357580 * g + 0.180423 * b;
		double yval = 0.212671 * r + 0.715160 * g + 0.072169 * b;
		double zVal = 0.019334 * r + 0.119193 * g + 0.950227 * b;

		xval /= (255.0 * 0.950456);
		yval /=  255.0;
		zVal /= (255.0 * 1.088754);

		double fX, fY, fZ;
		double lval, aval, bval;

		if (yval > 0.008856)
		{
			fY = pow(yval, 1.0 / 3.0);
			lval = 116.0 * fY - 16.0;
		}
		else
		{
			fY = 7.787 * yval + 16.0 / 116.0;
			lval = 903.3 * yval;
		}

		if (xval > 0.008856)
			fX = pow(xval, 1.0 / 3.0);
		else
			fX = 7.787 * xval + 16.0 / 116.0;

		if (zVal > 0.008856)
			fZ = pow(zVal, 1.0 / 3.0);
		else
			fZ = 7.787 * zVal + 16.0 / 116.0;

		aval = 500.0 * (fX - fY)+128.0;
		bval = 200.0 * (fY - fZ)+128.0;

		lvec[j] = lval;
		avec[j] = aval;
		bvec[j] = bval;
	}
}

//==============================================================================
///	GaussianSmooth
///
///	Blur an image with a separable binomial kernel passed in.
//==============================================================================
void API_FAST_SEGMENT::GaussianSmooth(
	const vector<double>&			inputImg,
	const int&						width,
	const int&						height,
	const vector<double>&			kernel,
	vector<double>&					smoothImg)
{
	int center = int(kernel.size())/2;

	int sz = width*height;
	smoothImg.clear();
	smoothImg.resize(sz);
	vector<double> tempim(sz);
	int rows = height;
	int cols = width;
   //--------------------------------------------------------------------------
   // Blur in the x direction.
   //---------------------------------------------------------------------------
	{int index(0);
	for( int r = 0; r < rows; r++ )
	{
		for( int c = 0; c < cols; c++ )
		{
			double kernelsum(0);
			double sum(0);
			for( int cc = (-center); cc <= center; cc++ )
			{
				if(((c+cc) >= 0) && ((c+cc) < cols))
				{
					sum += inputImg[r*cols+(c+cc)] * kernel[center+cc];
					kernelsum += kernel[center+cc];
				}
			}
			tempim[index] = sum/kernelsum;
			index++;
		}
	}}

	//--------------------------------------------------------------------------
	// Blur in the y direction.
	//---------------------------------------------------------------------------
	{int index = 0;
	for( int r = 0; r < rows; r++ )
	{
		for( int c = 0; c < cols; c++ )
		{
			double kernelsum(0);
			double sum(0);
			for( int rr = (-center); rr <= center; rr++ )
			{
				if(((r+rr) >= 0) && ((r+rr) < rows))
				{
				   sum += tempim[(r+rr)*cols+c] * kernel[center+rr];
				   kernelsum += kernel[center+rr];
				}
			}
			smoothImg[index] = sum/kernelsum;
			index++;
		}
	}}
}

//===========================================================================
///	GetSaliencyMap
///
/// Outputs a saliency map with a value assigned per pixel. The values are
/// normalized in the interval [0,255] if normflag is set true (default value).
//===========================================================================
void API_FAST_SEGMENT::GetSaliencyMap(
	const vector<unsigned int>&		inputimg,
	const int&						width,
	const int&						height,
	vector<int>&					output) 
{
	int i;
	int sz = width*height;
	double size = 1.0/sz ;
	vector<double> salmap;
	salmap.clear();
	salmap.resize(sz);

	vector<double> lvec, avec, bvec;
	RGB2LAB(inputimg, lvec, avec, bvec);

	//--------------------------
	// Obtain Lab average values
	//--------------------------
	double avgl = 0;
	double avga = 0;
	double avgb = 0;
	for( i = 0; i < sz; i++ )
	{
		avgl += lvec[i]*size;
		avga += avec[i]*size;
		avgb += bvec[i]*size;
	}

	vector<double> slvec, savec, sbvec;

	//----------------------------------------------------
	// The kernel can be [1 2 1] or [1 4 6 4 1] as needed.
	// The code below show usage of [1 2 1] kernel.
	//----------------------------------------------------
	vector<double> kernel;
	kernel.push_back(1.0);
	kernel.push_back(2.0);
	kernel.push_back(1.0);

	GaussianSmooth(lvec, width, height, kernel, slvec);
	GaussianSmooth(avec, width, height, kernel, savec);
	GaussianSmooth(bvec, width, height, kernel, sbvec);

	for( i = 0; i < sz; i++ )
	{
		salmap[i] = (slvec[i]-avgl)*(slvec[i]-avgl) +
					(savec[i]-avga)*(savec[i]-avga) +
					(sbvec[i]-avgb)*(sbvec[i]-avgb);
	}

	output.clear();
	Normalize(salmap, width, height, output);
}

int API_FAST_SEGMENT::FindMaxContour( 
		uchar* 	gray, 				//[In]:image->bgr
		int 	width, 				//[In]:image->width
		int 	height,				//[In]:image->height
		int 	channel,			//[In]:image->channel
		int* 	rect )
{
	if( !gray || (width<16) || (height<16) || (channel != 1) ) 
	{	
		cout<<"image err!!" << endl;
		return NULL;
	}
		
	int i,j,contour_num;
	double area, maxarea = 0;

	IplImage *imgGray = cvCreateImage(cvSize(width,height), IPL_DEPTH_8U, 1);
	for(i=0; i<height; i++)
    {
        for(j = 0; j<width; j++)
        {
            ((uchar *)(imgGray->imageData + i*width))[j] = gray[i*width + j];
        }
    }
	
	CvMemStorage* storage = cvCreateMemStorage(0);
	CvSeq *contour = 0, *contmax = 0;

	//FindContours
	contour_num = cvFindContours(imgGray, storage, &contour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	for( ; contour; contour = contour->h_next)
	{
		area = fabs(cvContourArea(contour, CV_WHOLE_SEQ));
		if(area > maxarea)
		{
			contmax = contour;
			maxarea = area;
		}
	}
	//printf("maxarea == %lf\n", maxarea);
	CvRect aRect = cvBoundingRect(contmax, 0);

	rect[0] = aRect.x;
	rect[1] = aRect.y;
	rect[2] = aRect.x+aRect.width;
	rect[3] = aRect.y+aRect.height;

	/*****************************cvReleaseImage*****************************/
	cvReleaseMemStorage(&storage);
	cvReleaseImage(&imgGray);imgGray = 0;

	return 0;
}  


