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

#include "API_fast_segment/API_fast_segment_mobile.h"

using namespace cv;
using namespace std;

/***********************************COMMIN*************************************/
static bool ImgSortComp( const pair< vector< int >, float > elem1, const pair< vector< int >, float > elem2)
{
	return (elem1.second > elem2.second);
}

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
	int nRet = 0;

	/************************rgb2gray*****************************/
	uchar *gray = getMaxChannel(bgr, width, height, channel);
	uchar *graySmooth = gaussianFilter(gray, width, height, 1);

	/************************ImageStretchByHistogram*****************************/
	uchar* grayHist = ImageStretchByHistogram(graySmooth, width, height, 1);
	uchar *graySmooth2 = gaussianFilter(grayHist, width, height, 1);

	//Salient_LC Img
	uchar *imgLC = Salient_LC(graySmooth2, width, height, 1);

	//FindMaxContour
	nRet = FindMaxContour( imgLC, width, height, 1, rect );

	//sv image
	IplImage* grayCorrespond = cvCreateImage( cvSize(width*5, height), IPL_DEPTH_8U, 1 );
	IplImage* ipl_gray = api_commen.uchar2ipl(gray, width, height, 1);	
	IplImage* ipl_graySmooth = api_commen.uchar2ipl(graySmooth, width, height, 1);	
	IplImage* ipl_grayHist = api_commen.uchar2ipl(grayHist, width, height, 1);	
	IplImage* ipl_graySmooth2 = api_commen.uchar2ipl(graySmooth2, width, height, 1);	
	IplImage* ipl_imgLC = api_commen.uchar2ipl(imgLC, width, height, 1);	

	cvSetImageROI( grayCorrespond, cvRect( 0, 0, width, height ) );
    cvCopy( ipl_gray, grayCorrespond );

	cvSetImageROI( grayCorrespond, cvRect( width, 0, width, height ) );
    cvCopy( ipl_graySmooth, grayCorrespond );

	cvSetImageROI( grayCorrespond, cvRect( width*2, 0, width, height ) );
    cvCopy( ipl_grayHist, grayCorrespond );

	cvSetImageROI( grayCorrespond, cvRect( width*3, 0, width, height ) );
    cvCopy( ipl_graySmooth2, grayCorrespond );

	cvSetImageROI( grayCorrespond, cvRect( width*4, 0, width, height ) );
    cvCopy( ipl_imgLC, grayCorrespond );

	//output info
    cvResetImageROI( grayCorrespond );

	char szImgPath[1024];
	sprintf(szImgPath, "res/gray/%s.jpg", iid.c_str() );
	cvSaveImage( szImgPath, grayCorrespond );

	//cvReleaseImage
	cvReleaseImage(&ipl_gray);ipl_gray = 0;
	cvReleaseImage(&ipl_graySmooth);ipl_graySmooth = 0;
	cvReleaseImage(&ipl_grayHist);ipl_grayHist = 0;
	cvReleaseImage(&ipl_graySmooth2);ipl_graySmooth2 = 0;
	cvReleaseImage(&ipl_imgLC);ipl_imgLC = 0;
	cvReleaseImage(&grayCorrespond);grayCorrespond = 0;


	delete imgLC;
	delete graySmooth;
	delete gray;
	
	return nRet;
}

uchar* API_FAST_SEGMENT::getMaxChannel(
	uchar* 	bgr, 				//[In]:image->bgr
	int 	width, 				//[In]:image->width
	int 	height,				//[In]:image->height
	int 	channel)			//[In]:image->channel
{
	if( !bgr || (width<16) || (height<16) || (channel != 3) ) 
	{	
		cout<<"image err!!" << endl;
		return NULL;
	}	

	int i, j, k, tmp, index, maxValue;
	int step = 3*width;
	int minPixel[4] = {10000000,10000000,10000000,10000000};
	int	maxPixel[4] = {0,0,0,0};
	uchar *maxChannel = new uchar[width*height];

	//rgb2gray
	uchar *gray = rgb2gray(bgr, width, height, channel, minPixel[3], maxPixel[3]);

	for(i=0; i<height; i++)
    {
        for(j = 0; j<width; j++)
        {
        	for(k = 0; k<channel; k++)
        	{
	            tmp = (uchar)(bgr[i*step + 3*j + k]);  // b-g-r
				if( maxPixel[k] < tmp ) 
					maxPixel[k] = tmp;
				if( minPixel[k] > tmp ) 
					minPixel[k] = tmp;
        	}
        }
    }

	//get max channel
	index = -1;
	maxValue = 0;
	for (i=0;i<4;i++)
	{
		tmp = abs(maxPixel[i]-minPixel[i]);
		if( maxValue < tmp ) 
		{
			maxValue = tmp;
			index = i;
		}
	}

	//get data
	for(i=0; i<height; i++)
    {
        for(j = 0; j<width; j++)
        {
        	if ( (index>=0) && (index<3) )
				maxChannel[i*width + j] = (uchar)(bgr[i*step + 3*j + index]);
			else if (index==3)
				maxChannel[i*width + j] = gray[i*width + j];
       	}
	}

	delete gray;

    return maxChannel;
}

uchar* API_FAST_SEGMENT::rgb2gray(
	uchar* 	bgr, 				//[In]:image->bgr
	int 	width, 				//[In]:image->width
	int 	height,				//[In]:image->height
	int 	channel,			//[In]:image->channel
	int 	&minPixel,
	int 	&maxPixel)
{
	if( !bgr || (width<16) || (height<16) || (channel != 3) ) 
	{	
		cout<<"image err!!" << endl;
		return NULL;
	}	

	int i,j,tmp;
	int step = 3*width;
	uchar *gray = new uchar[width*height];

	maxPixel = 0;
	minPixel = 10000000;

	for(i=0; i<height; i++)
    {
        for(j = 0; j<width; j++)
        {
            //Gray = (R*38 + G*75 + B*15) >> 7
            tmp = (uchar) ((bgr[i*step + 3*j +2]*38 +
                bgr[i*step + 3*j +1]*75 +
                bgr[i*step + 3*j ]*15) >> 7);  // b
            gray[i*width + j] = tmp;

			if( maxPixel < tmp ) 
				maxPixel = tmp;
			if( minPixel > tmp ) 
				minPixel = tmp;
        }
    }

    return gray;
}


uchar* API_FAST_SEGMENT::ImageStretchByHistogram(
	uchar*	src,				//[In]:image->bgr
	int 	width,				//[In]:image->width
	int 	height, 			//[In]:image->height
	int 	channel)			//[In]:image->channel
{
	if( !src || (width<16) || (height<16) || (channel != 1) ) 
	{	
		cout<<"image err!!" << endl;
		return NULL;
	}
	
	uchar v;
	int i,k,x,y;
	long wMulh = height * width;
	
	double p[256],p1[256],num[256];
	memset(p,0,sizeof(p));
	memset(p1,0,sizeof(p1));
	memset(num,0,sizeof(num));

	uchar *dst = new uchar[width*height];

	//statistics
	for( x=0;x<width;x++)
	{
		for( y=0;y<height;y++){
			v = src[y*width + x];
			num[v]++;
		}
	}
	
	//calculate probability
	for( i=0;i<256;i++)
	{
		p[i]=num[i]*1.0/wMulh;
	}

	//p1[i]=sum(p[j]);	j<=i;
	for( i=0;i<256;i++)
	{
		for( k=0;k<=i;k++)
			p1[i] += p[k];
	}

	// histogram transformation
	for( x=0;x<width;x++)
	{
		for( y=0;y<height;y++){
			v = src[y*width + x];
			dst[y*width + x] = int(p1[v]*255+0.5);            
		}
	}
	
	return dst;
}

uchar* API_FAST_SEGMENT::gaussianFilter(uchar* data, int width, int height, int channel)
{
	if( !data || (width<16) || (height<16) || (channel != 1) ) 
	{	
		cout<<"image err!!" << endl;
		return NULL;
	}
	
    int i, j, k, m, n, index, sum;
    int templates[9] = { 1, 2, 1,
                         2, 4, 2,
                         1, 2, 1 };

	sum = width*height*channel;
	uchar *dst = new uchar[sum];
	memset(dst, 0, sum * sizeof(uchar));
	for(k = 0;k < channel;k++)
    {
	    for(i = 1;i < height - 1;i++)
	    {
	        for(j = 1;j < width - 1;j++)
	        {
	        	sum = 0;
	            index = 0;		
	            for(m = i - 1;m < i + 2;m++)
	            {
	                for(n = j - 1; n < j + 2;n++)
	                {
	                    sum += data[m*width*channel+n*channel+k] * templates[index];
						index++;
	                }
	            }
	            dst[i*width*channel+j*channel+k] = int(sum*1.0/16+0.5);
        	}
        }
    }

	return dst;
}


//Y. Zhai and M. Shah, "Visual attention detection in video sequences using spatiotemporal cues" in ACM Multimedia,2006.
uchar* API_FAST_SEGMENT::Salient_LC(
	uchar*	gray,				//[In]:image->gray
	int 	width,				//[In]:image->width
	int 	height, 			//[In]:image->height
	int 	channel)			//[In]:image->channel
{
	if( !gray || (width<16) || (height<16) || (channel != 1) ) 
	{	
		cout<<"image err!!" << endl;
		return NULL;
	}
	
	int i,j,k,count,tmpPixel,index;
	float tmp,sumTmp;
	
	float size    = 1.0/ (height * width) ;

	int graySize = 256;
	float tmpMin = 1000000.0;
	float tmpMax = -1000000.0;
	int *pixFQ = new int[graySize];
	memset(pixFQ, 0, graySize * sizeof(int));
	float *pixScore = new float[graySize];
	memset(pixScore, 0, graySize * sizeof(float));

	uchar *dst = new uchar[width*height];
	
	for (i = 0; i < height; i++) 
	{
		for (j = 0; j < width; j++)
		{
			tmpPixel = gray[i*width + j];
			pixFQ[tmpPixel]++;
		}
	}

	//count LC
	for (i = 0; i < graySize; i++) 
	{
		for (j = 0; j < graySize; j++)
		{
			if (i==j)
				continue;

			pixScore[i] += (abs(i-j))*pixFQ[j];
		}
		pixScore[i] = pixScore[i]*size;
		if ( pixScore[i] < tmpMin )
			tmpMin = pixScore[i];
		if ( pixScore[i] > tmpMax )
			tmpMax = pixScore[i];
	}

	//get map
	for (i = 0; i < height; i++) 
	{
		for (j = 0; j < width; j++)
		{
			index = gray[i*width + j];
			tmpPixel = (int)( 255.0*((tmpMax-pixScore[index])*1.0/(tmpMax-tmpMin)) + 0.5);
			if (tmpPixel>=200)
				dst[i*width + j] = 255;
			else
				dst[i*width + j] = 0;
		}
	}

	if (pixFQ) {delete [] pixFQ;pixFQ = NULL;}
	if (pixScore) {delete [] pixScore;pixScore = NULL;}
	
	return dst;
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


