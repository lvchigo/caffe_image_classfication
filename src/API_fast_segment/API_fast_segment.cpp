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

#include <time.h>
#include <sys/mman.h> /* for mmap and munmap */
#include <sys/types.h> /* for open */
#include <sys/stat.h> /* for open */
#include <fcntl.h>     /* for open */
#include <pthread.h>

#include <vector>
#include <list>
#include <map>
#include <algorithm>

#include "plog/Log.h"
#include "API_commen/API_commen.h"
#include "API_commen/TErrorCode.h"
#include "API_fast_segment/API_fast_segment.h"

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

void API_FAST_SEGMENT::Init()
{
	int nRet = TOK;
	int i,j,tmpPixel;
	int graySize = 256;
	int grayLen = graySize*graySize;

	pixDist = new float[grayLen];
	memset(pixDist, 0, grayLen * sizeof(float));

	int grayD[65536]={0};

	for (i = 0; i < graySize; i++) 
	{
		for (j = 0; j < graySize; j++)
		{
			tmpPixel = abs(i-j);
			//printf("%d,",tmpPixel);
		}
		//printf("\n");
	}
}

void API_FAST_SEGMENT::Release()
{
	if (pixDist) {delete [] pixDist;pixDist = NULL;}
}

int API_FAST_SEGMENT::do_fast_segment( IplImage *img, string iid, Vec4i &rect )
{
	if(!img || (img->width<16) || (img->height<16) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
	{	
		cout<<"image err!!" << endl;
		return TEC_INVALID_PARAM;
	}	

	/*****************************Init*****************************/
	int i,m,n,rWidth,rHeight,nRet = 0;
	char szImgPath[1024];
	string strImgPath;
	
	/************************ResizeImg*****************************/
	float ratio = 1.0;
	IplImage* imgResize = api_commen.ResizeImg( img, ratio, 128 );	//320:30ms,512:50ms,720:80ms
	if(!imgResize || imgResize->nChannels != 3 || imgResize->depth != IPL_DEPTH_8U) 
	{	
		LOOGE<<"Fail to ResizeImg";
		cvReleaseImage(&imgResize);imgResize = 0;
		return TEC_INVALID_PARAM;
	}

	/************************color to gray*****************************/
	IplImage *imgGray = cvCreateImage(cvGetSize(imgResize), imgResize->depth, 1);
	cvCvtColor( imgResize, imgGray, CV_BGR2GRAY );

	IplImage* imgCorrespond = cvCreateImage( cvSize(imgResize->width*4, imgResize->height*2), imgResize->depth, imgResize->nChannels );
	IplImage* grayCorrespond = cvCreateImage( cvSize(imgGray->width*4, imgGray->height*4), imgGray->depth, imgGray->nChannels );

	cvSetImageROI( imgCorrespond, cvRect( 0, 0, imgResize->width, imgResize->height ) );
    cvCopy( imgResize, imgCorrespond );
	cvSetImageROI( imgCorrespond, cvRect( 0, imgResize->height, imgResize->width, imgResize->height ) );
    cvCopy( imgResize, imgCorrespond );

	cvSetImageROI( grayCorrespond, cvRect( 0, 0, imgGray->width, imgGray->height ) );
    cvCopy( imgGray, grayCorrespond );

	/*****************************cvSmooth Img*****************************/
	IplImage *imgSmooth = cvCreateImage(cvGetSize(imgGray), imgGray->depth, imgGray->nChannels);
	cvSmooth(imgGray,imgSmooth,CV_MEDIAN,3,3);

	IplImage *imgSmooth2 = cvCreateImage(cvGetSize(imgResize), imgResize->depth, imgResize->nChannels);
	cvSmooth(imgResize,imgSmooth2,CV_MEDIAN,3,3);

	/************************ImageStretchByHistogram*****************************/
	IplImage *imgGray255 = cvCreateImage(cvGetSize(imgGray), imgGray->depth, 1);
	nRet = ImageStretchByHistogram(imgSmooth, imgGray255);

	cvSetImageROI( grayCorrespond, cvRect( 0, imgGray->height, imgGray->width, imgGray->height ) );
    cvCopy( imgGray255, grayCorrespond );

	IplImage *imgImageHistogram = cvCreateImage(cvGetSize(imgResize), imgResize->depth, imgResize->nChannels);
	nRet = ImageHistogram(imgSmooth2,imgImageHistogram);

	//Salient_LC Img
	IplImage *imgLC = cvCreateImage(cvGetSize(imgGray), imgGray->depth, imgGray->nChannels);
	IplImage *imgLCT1 = cvCreateImage(cvGetSize(imgGray), imgGray->depth, imgGray->nChannels);
	IplImage *imgLCT2 = cvCreateImage(cvGetSize(imgGray), imgGray->depth, imgGray->nChannels);
	nRet = Salient_LC(imgGray255, imgLC, imgLCT1, imgLCT2, 64, rect);
	//nRet = Salient_HC(imgImageHistogram, imgLC, imgLCT1, imgLCT2, 12, rect);

	cvSetImageROI( grayCorrespond, cvRect( imgGray->width, 0, imgGray->width, imgGray->height ) );
	cvCopy( imgLC, grayCorrespond );

	cvSetImageROI( grayCorrespond, cvRect( imgGray->width*2, 0, imgGray->width, imgGray->height ) );
	cvCopy( imgLCT1, grayCorrespond );

	cvSetImageROI( grayCorrespond, cvRect( imgGray->width*3, 0, imgGray->width, imgGray->height ) );
	cvCopy( imgLCT2, grayCorrespond );

/*	//Salient_HC Img
	IplImage *imgHC = cvCreateImage(cvGetSize(imgGray), imgGray->depth, imgGray->nChannels);
	IplImage *imgHCT1 = cvCreateImage(cvGetSize(imgGray), imgGray->depth, imgGray->nChannels);
	IplImage *imgHCT2 = cvCreateImage(cvGetSize(imgGray), imgGray->depth, imgGray->nChannels);
	//nRet = Salient_LC(imgGray255, imgHC, imgHCT1, imgHCT2, 64, rect);
	nRet = Salient_HC(imgImageHistogram, imgHC, imgHCT1, imgHCT2, 12, rect);

	cvSetImageROI( grayCorrespond, cvRect( imgGray->width, imgGray->height, imgGray->width, imgGray->height ) );
	cvCopy( imgHC, grayCorrespond );

	cvSetImageROI( grayCorrespond, cvRect( imgGray->width*2, imgGray->height, imgGray->width, imgGray->height ) );
	cvCopy( imgHCT1, grayCorrespond );

	cvSetImageROI( grayCorrespond, cvRect( imgGray->width*3, imgGray->height, imgGray->width, imgGray->height ) );
	cvCopy( imgHCT2, grayCorrespond ); 
*/
	//Salient_RC Img
	//IplImage *imgRC = cvCreateImage(cvGetSize(imgResize), 8, 1);
	Mat matRC;
	nRet = Salient_RC(imgResize, iid, matRC);
	if ( nRet!=0 ) 
	{	
		//cout<<"no image roi!!" << endl;
		cvReleaseImage(&imgImageHistogram);imgImageHistogram = 0;
		cvReleaseImage(&imgSmooth);imgSmooth = 0;
		cvReleaseImage(&imgSmooth2);imgSmooth2 = 0;
		cvReleaseImage(&imgLC);imgLC = 0;
		cvReleaseImage(&imgLCT1);imgLCT1 = 0;
		cvReleaseImage(&imgLCT2);imgLCT2 = 0;
		cvReleaseImage(&grayCorrespond);grayCorrespond = 0;
		cvReleaseImage(&imgCorrespond);imgCorrespond = 0;
		cvReleaseImage(&imgGray);imgGray = 0;
		cvReleaseImage(&imgGray255);imgGray255 = 0;
		cvReleaseImage(&imgResize);imgResize = 0;
		return TEC_INVALID_PARAM;
	}
	
	IplImage imgRC = matRC;
	cvSetImageROI( grayCorrespond, cvRect( 0, imgGray->height*2, imgGray->width, imgGray->height ) );
	cvCopy( &imgRC, grayCorrespond );

	sprintf(szImgPath, "res/rc/%s.jpg", iid.c_str() );
	cvSaveImage( szImgPath, &imgRC );

	//FindMaxContour
	IplImage *imgThreshold128 = cvCreateImage(cvGetSize(imgResize), imgResize->depth, 1);
	IplImage *pContourImg128 = cvCreateImage(cvGetSize(imgResize), imgResize->depth, 3); 
	int T1_Binbary = 1;
	int T2_Binbary = 255;
	nRet = FindMaxContour( &imgRC, iid, 0, T1_Binbary, T2_Binbary, imgThreshold128, pContourImg128, rect );

	cvSetImageROI( grayCorrespond, cvRect( imgGray->width, imgGray->height*2, imgGray->width, imgGray->height ) );
	cvCopy( imgThreshold128, grayCorrespond );

	cvSetImageROI( imgCorrespond, cvRect( imgResize->width, 0, imgResize->width, imgResize->height ) );
	cvCopy( pContourImg128, imgCorrespond );

	//FindMaxContour
	IplImage *imgThreshold164 = cvCreateImage(cvGetSize(imgResize), imgResize->depth, 1);
	IplImage *pContourImg164 = cvCreateImage(cvGetSize(imgResize), imgResize->depth, 3); 
	T1_Binbary = 16;
	T2_Binbary = 255;
	nRet = FindMaxContour( &imgRC, iid, 0, T1_Binbary, T2_Binbary, imgThreshold164, pContourImg164, rect );

	cvSetImageROI( grayCorrespond, cvRect( imgGray->width*2, imgGray->height*2, imgGray->width, imgGray->height ) );
	cvCopy( imgThreshold164, grayCorrespond );

	cvSetImageROI( imgCorrespond, cvRect( imgResize->width*2, 0, imgResize->width, imgResize->height ) );
	cvCopy( pContourImg164, imgCorrespond );

	//FindMaxContour
	IplImage *imgThreshold = cvCreateImage(cvGetSize(imgResize), imgResize->depth, 1);
	IplImage *pContourImg = cvCreateImage(cvGetSize(imgResize), imgResize->depth, 3); 
	T1_Binbary = 64;
	T2_Binbary = 255;
	nRet = FindMaxContour( &imgRC, iid, 0, T1_Binbary, T2_Binbary, imgThreshold, pContourImg, rect );

	cvSetImageROI( grayCorrespond, cvRect( imgGray->width*3, imgGray->height*2, imgGray->width, imgGray->height ) );
	cvCopy( imgThreshold, grayCorrespond );

	cvSetImageROI( imgCorrespond, cvRect( imgResize->width*3, 0, imgResize->width, imgResize->height ) );
	cvCopy( pContourImg, imgCorrespond );

	//FindMaxContour
	IplImage *imgThreshold1_128 = cvCreateImage(cvGetSize(imgResize), imgResize->depth, 1);
	IplImage *pContourImg1_128 = cvCreateImage(cvGetSize(imgResize), imgResize->depth, 3); 
	T1_Binbary = 50;
	T2_Binbary = 150;
	nRet = FindMaxContour( imgLC, iid, 1, T1_Binbary, T2_Binbary, imgThreshold1_128, pContourImg1_128, rect );

	cvSetImageROI( grayCorrespond, cvRect( 0, imgGray->height*3, imgGray->width, imgGray->height ) );
	cvCopy( imgLC, grayCorrespond );

	cvSetImageROI( grayCorrespond, cvRect( imgGray->width, imgGray->height*3, imgGray->width, imgGray->height ) );
	cvCopy( imgThreshold1_128, grayCorrespond );

	cvSetImageROI( imgCorrespond, cvRect( imgResize->width, imgResize->height, imgResize->width, imgResize->height ) );
	cvCopy( pContourImg1_128, imgCorrespond );

	//FindMaxContour
	IplImage *imgThreshold1_164 = cvCreateImage(cvGetSize(imgResize), imgResize->depth, 1);
	IplImage *pContourImg1_164 = cvCreateImage(cvGetSize(imgResize), imgResize->depth, 3); 
	T1_Binbary = 100;
	T2_Binbary = 200;
	nRet = FindMaxContour( imgLC, iid, 1, T1_Binbary, T2_Binbary, imgThreshold1_164, pContourImg1_164, rect );

	cvSetImageROI( grayCorrespond, cvRect( imgGray->width*2, imgGray->height*3, imgGray->width, imgGray->height ) );
	cvCopy( imgThreshold1_164, grayCorrespond );

	cvSetImageROI( imgCorrespond, cvRect( imgResize->width*2, imgResize->height, imgResize->width, imgResize->height ) );
	cvCopy( pContourImg1_164, imgCorrespond );

	//FindMaxContour
	IplImage *imgThreshold1 = cvCreateImage(cvGetSize(imgResize), imgResize->depth, 1);
	IplImage *pContourImg1 = cvCreateImage(cvGetSize(imgResize), imgResize->depth, 3); 
	T1_Binbary = 128;
	T2_Binbary = 200;
	nRet = FindMaxContour( imgLC, iid, 1, T1_Binbary, T2_Binbary, imgThreshold1, pContourImg1, rect );

	cvSetImageROI( grayCorrespond, cvRect( imgGray->width*3, imgGray->height*3, imgGray->width, imgGray->height ) );
	cvCopy( imgThreshold1, grayCorrespond );

	cvSetImageROI( imgCorrespond, cvRect( imgResize->width*3, imgResize->height, imgResize->width, imgResize->height ) );
	cvCopy( pContourImg1, imgCorrespond );
	
	//output info
    cvResetImageROI( grayCorrespond );
	sprintf(szImgPath, "res/gray/%s.jpg", iid.c_str() );
	cvSaveImage( szImgPath, grayCorrespond );

	cvResetImageROI( imgCorrespond );
	sprintf(szImgPath, "res/color/%s.jpg", iid.c_str() );
	cvSaveImage( szImgPath, imgCorrespond );
	
	//cvReleaseImage
	cvReleaseImage(&imgImageHistogram);imgImageHistogram = 0;
	cvReleaseImage(&imgSmooth);imgSmooth = 0;
	cvReleaseImage(&imgSmooth2);imgSmooth2 = 0;
	cvReleaseImage(&imgThreshold128);imgThreshold128 = 0;
	cvReleaseImage(&pContourImg128);pContourImg128 = 0;
	cvReleaseImage(&imgThreshold164);imgThreshold164 = 0;
	cvReleaseImage(&pContourImg164);pContourImg164 = 0;
	cvReleaseImage(&imgThreshold);imgThreshold = 0;
	cvReleaseImage(&pContourImg);pContourImg = 0;
	cvReleaseImage(&imgThreshold1_128);imgThreshold1_128 = 0;
	cvReleaseImage(&pContourImg1_128);pContourImg1_128 = 0;
	cvReleaseImage(&imgThreshold1_164);imgThreshold1_164 = 0;
	cvReleaseImage(&pContourImg1_164);pContourImg1_164 = 0;
	cvReleaseImage(&imgThreshold1);imgThreshold1 = 0;
	cvReleaseImage(&pContourImg1);pContourImg1 = 0;
	//cvReleaseImage(&imgHC);imgHC = 0;
	//cvReleaseImage(&imgHCT1);imgHCT1 = 0;
	//cvReleaseImage(&imgHCT2);imgHCT2 = 0;
	//cvReleaseImage(&imgRC);imgRC = 0;
	cvReleaseImage(&imgLC);imgLC = 0;
	cvReleaseImage(&imgLCT1);imgLCT1 = 0;
	cvReleaseImage(&imgLCT2);imgLCT2 = 0;
	cvReleaseImage(&grayCorrespond);grayCorrespond = 0;
	cvReleaseImage(&imgCorrespond);imgCorrespond = 0;
	cvReleaseImage(&imgGray);imgGray = 0;
	cvReleaseImage(&imgGray255);imgGray255 = 0;
	cvReleaseImage(&imgResize);imgResize = 0;
	
	return 0;
}

int API_FAST_SEGMENT::FindMaxContour( IplImage* src, string iid, int type, int T1_Binbary, int T2_Binbary, 
	IplImage *imgThreshold, IplImage *pContourImg, Vec4i &rect )  
{
	if(!src || (src->width<16) || (src->height<16) || src->nChannels != 1 || src->depth != IPL_DEPTH_8U) 
	{	
		cout<<"image err!!" << endl;
		return TEC_INVALID_PARAM;
	}	
	
	int contour_num;
	double area, maxarea = 0;
	
	CvMemStorage* storage = cvCreateMemStorage(0);
	CvSeq *contour = 0, *contmax = 0;

	//image binary
	if (type == 1)
		cvCanny( src, imgThreshold, T1_Binbary, T2_Binbary, 3 );
	else
		cvThreshold(src, imgThreshold, T1_Binbary, 255, CV_THRESH_BINARY);
	
	//FindContours
	contour_num = cvFindContours(imgThreshold, storage, &contour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
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
	cvCvtColor(src, pContourImg, CV_GRAY2BGR);
	cvDrawContours(pContourImg, contmax, CV_RGB(0,0,255), CV_RGB(0, 255, 0), 1, 1, 8, cvPoint(0,0)); 
	cvRectangle( pContourImg, cvPoint(aRect.x, aRect.y),
               cvPoint(aRect.x+aRect.width, aRect.y+aRect.height), CV_RGB(255,0,0), 2, 8, 0);

	rect[0] = aRect.x;
	rect[1] = aRect.y;
	rect[2] = aRect.x+aRect.width;
	rect[3] = aRect.y+aRect.height;

	/*****************************cvReleaseImage*****************************/
	cvReleaseMemStorage(&storage);

	return 0;
}  


int API_FAST_SEGMENT::ImageStretchByHistogram(IplImage *src1,IplImage *dst1)
{
	assert(src1->width==dst1->width);
	double p[256],p1[256],num[256];

	memset(p,0,sizeof(p));
	memset(p1,0,sizeof(p1));
	memset(num,0,sizeof(num));
	int height=src1->height;
	int width=src1->width;
	long wMulh = height * width;

	//statistics
	for(int x=0;x<src1->width;x++)
	{
		for(int y=0;y<src1-> height;y++){
			uchar v=((uchar*)(src1->imageData + src1->widthStep*y))[x];
			num[v]++;
		}
	}
	//calculate probability
	for(int i=0;i<256;i++)
	{
		p[i]=num[i]/wMulh;
	}

	//p1[i]=sum(p[j]);	j<=i;
	for(int i=0;i<256;i++)
	{
		for(int k=0;k<=i;k++)
			p1[i]+=p[k];
	}

	// histogram transformation
	for(int x=0;x<src1->width;x++)
	{
		for(int y=0;y<src1-> height;y++){
			uchar v=((uchar*)(src1->imageData + src1->widthStep*y))[x];
			((uchar*)(dst1->imageData + dst1->widthStep*y))[x]= p1[v]*255+0.5;            
		}
	}
	return 0;
}

int API_FAST_SEGMENT::ImageHistogram(IplImage *src1,IplImage *dst1)
{
	int i;
	IplImage* imgChannel[3] = { 0, 0, 0 };

	if( src1 )
	{
		for( i = 0; i < src1->nChannels; i++ )
		{
			imgChannel[i] = cvCreateImage( cvGetSize( src1 ), IPL_DEPTH_8U, 1 ); 
		}
		
		cvSplit( src1, imgChannel[0], imgChannel[1], imgChannel[2], 0 );//BGRA
		for( i = 0; i < dst1->nChannels; i++ )
		{
			cvEqualizeHist( imgChannel[i], imgChannel[i] );
		}
		
		cvMerge( imgChannel[0], imgChannel[1], imgChannel[2], 0, dst1 );
		for( i = 0; i < src1->nChannels; i++ )
		{
			if( imgChannel[i] )
			{
				cvReleaseImage( &imgChannel[i] );
				//imgChannel[i] = 0;
			}
		}
	}

	return 0;
}



//Y. Zhai and M. Shah, "Visual attention detection in video sequences using spatiotemporal cues" in ACM Multimedia,2006.
int API_FAST_SEGMENT::Salient_LC(IplImage *src, IplImage *dst, IplImage *dst2, IplImage *dst3, int numColorBlock, Vec4i &rect)
{
	int nRet = TOK;
	int i,j,k,count,tmpPixel,index;
	float tmp,sumTmp;
	
	int height     = src->height;
	int width      = src->width;
	int step       = src->widthStep;
	int channels   = src->nChannels;
	float size    = 1.0/ (height * width) ;

	int graySize = 256;
	float tmpMin = 1000000.0;
	float tmpMax = -1000000.0;
	int *pixFQ = new int[graySize];
	memset(pixFQ, 0, graySize * sizeof(int));
	float *pixScore = new float[graySize];
	memset(pixScore, 0, graySize * sizeof(float));
	
	for (i = 0; i < height; i++) 
	{
		for (j = 0; j < width; j++)
		{
			tmpPixel = ((uchar *)(src->imageData + i*step))[j];
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
			index = ((uchar *)(src->imageData + i*step))[j];
			tmpPixel = (int)( 255.0*((tmpMax-pixScore[index])*1.0/(tmpMax-tmpMin)) + 0.5);
			((uchar *)(dst->imageData + i*step))[j] = tmpPixel;

			if (tmpPixel>=160)
				((uchar *)(dst2->imageData + i*step))[j] = tmpPixel;
			else
				((uchar *)(dst2->imageData + i*step))[j] = 0;

			if (tmpPixel>=200)
				((uchar *)(dst3->imageData + i*step))[j] = tmpPixel;
			else
				((uchar *)(dst3->imageData + i*step))[j] = 0;
		}
	}

	if (pixFQ) {delete [] pixFQ;pixFQ = NULL;}
	if (pixScore) {delete [] pixScore;pixScore = NULL;}
	
	return 0;
}

int API_FAST_SEGMENT::Salient_RC(IplImage *src, string iid, Mat &matOutput)
{
	int i,j,nRet = TOK;
	int step = src->width;
	int tmp,sum = 0;
	float tmpMin = 1000000;
	float tmpMax = 0;
	CmSaliencyRC tCmSaliencyRC;

	Mat matInput = cv::cvarrToMat(src,false);
	matInput.convertTo(matInput, CV_32FC3, 1.0/255);
	Mat matRC = tCmSaliencyRC.GetRC(matInput, iid);
	if ( matRC.empty() ) 
	{	
		//cout<<"no image roi!!" << endl;
		return -1;
	}
	
	matRC.convertTo(matRC, CV_32FC3, 255.0);
	//matRC.convertTo(matRC, CV_32FC1, 255.0);
	//cv::normalize(matOutput, matOutput, 0, 255, NORM_MINMAX, CV_32F);

	matOutput = Mat::zeros(src->height, src->width, CV_8UC1); 
	normalize(matRC,matRC,1.0,0.0,NORM_MINMAX);
	matRC.convertTo(matOutput, CV_8UC1, 255, 0);

	//get map
	for (i = 0; i < src->height; i++) 
	{
		for (j = 0; j < src->width; j++)
		{
			tmp = matOutput.at<uchar>(i,j);
			if (tmp>=1)
				sum++;
		}
	}
	printf("sum:%d\n",sum);

	if ( sum<3 ) 
	{	
		//cout<<"no image roi!!" << endl;
		return -1;
	}
	
	return 0;
}

/*
int API_FAST_SEGMENT::Salient_RC2(IplImage *src, string iid, IplImage *dst)
{
	int i,j,tmp,nRet = TOK;
	int tmpMin = 1000000;
	int tmpMax = 0;
	int step = src->width;
	CmSaliencyRC tCmSaliencyRC;

	Mat matInput = cv::cvarrToMat(src,false);
	matInput.convertTo(matInput, CV_32FC3, 1.0/255);
	Mat matOutput = tCmSaliencyRC.GetRC(matInput, iid);
	matOutput.convertTo(matOutput, CV_32FC3, 255.0);
	if ( matOutput.empty() ) 
	{	
		cout<<"no image roi!!" << endl;	
		return TEC_INVALID_PARAM;
	}
	

	IplImage ipl_Salient_RC = matOutput;
	//dst = cvCreateImage(cvGetSize(src), 8, 1);

	//get map
	for (i = 0; i < src->height; i++) 
	{
		for (j = 0; j < src->width; j++)
		{
			tmp = ((uchar *)((&ipl_Salient_RC)->imageData + i*step))[j];
			if ( tmp < tmpMin )
				tmpMin = tmp;
			if ( tmp > tmpMax )
				tmpMax = tmp;
		}
	}
	printf("tmpMin:%d,tmpMax:%d\n",tmpMin,tmpMax);
	
	for (i = 0; i < src->height; i++) 
	{
		for (j = 0; j < src->width; j++)
		{
			tmp = ((uchar *)((&ipl_Salient_RC)->imageData + i*step))[j];
			tmp = (int)( 255.0*((tmp-tmpMin)*1.0/(tmpMax-tmpMin)) + 0.5);
			((uchar *)(dst->imageData + i*step))[j] = tmp;
		}
	}
	
	return 0;
}*/

//M.M.Cheng,"Global contrast based salient region detection" IEEE TPAMI(CVPR 2011), 2014.
int API_FAST_SEGMENT::Salient_HC(IplImage *src, IplImage *dst, IplImage *dst2, IplImage *dst3, int numColorBlock, Vec4i &rect)
{
	int nRet = TOK;
	int i,j,k,count,tmpPixel,index,sumIndex;
	float tmp,sumTmp;
	
	int height     = src->height;
	int width      = src->width;
	int step       = src->widthStep;
	int channels   = src->nChannels;
	int blockPixel = ceil( 256.0/numColorBlock );
	float size    = 1.0/ (height * width) ;

	float tmpMin = 1000000.0;
	float tmpMax = -1000000.0;

	int feat_len = 0;
	if ( channels == 3)
		feat_len = numColorBlock*numColorBlock*numColorBlock;
	else if ( channels == 1)
		feat_len = numColorBlock;
	else
		return -1;
	
	long *pixFQ = new long[feat_len];
	memset(pixFQ, 0, feat_len * sizeof(long));
	float *pixScore = new float[feat_len];
	memset(pixScore, 0, feat_len * sizeof(float));

	//Color Hist
	for (i = 0; i < height; i++) 
	{
		for (j = 0; j < width; j++)
		{
			sumIndex = 0;
			for(k=0;k<channels;k++)
			{
				tmpPixel = ((uchar *)(src->imageData + i*step))[j*channels + k];
				index = int(tmpPixel*1.0/blockPixel+0.5);	//norm	

				if (index >= numColorBlock)
					index = numColorBlock-1;

				if (k==0)
					sumIndex = index;
				else if (k==1)
					sumIndex += index*numColorBlock;
				else if (k==2)
					sumIndex += index*numColorBlock*numColorBlock;
			}
			pixFQ[sumIndex]++;
		}
	}

	//count LC
	tmpMin = 1000000.0;
	tmpMax = -1000000.0;
	for (i = 0; i < feat_len; i++) 
	{
		Vec3i tPix1;
		index = int(i/(numColorBlock*numColorBlock));
		tPix1[2] = index;
		index = int((i-index*numColorBlock*numColorBlock)/numColorBlock);
		tPix1[1] = index;
		tPix1[0] = i%numColorBlock;
		for (j = 0; j < feat_len; j++)
		{
			if (i==j)
				continue;

			Vec3i tPix2;
			index = int(j/(numColorBlock*numColorBlock));
			tPix2[2] = index;
			index = int((j-index*numColorBlock*numColorBlock)/numColorBlock);
			tPix2[1] = index;
			tPix2[0] = j%numColorBlock;

			sumIndex = 0;
			for (k = 0; k < channels; k++)
			{
				sumIndex += (tPix1[k]-tPix2[k])*(tPix1[k]-tPix2[k]);
			}
			pixScore[i] += (sqrt(sumIndex))*pixFQ[j]*size;
		}
		if ( pixScore[i] < tmpMin )
			tmpMin = pixScore[i];
		if ( pixScore[i] > tmpMax )
			tmpMax = pixScore[i];
	}
	//printf("2:tmpMin:%f,tmpMax:%f\n",tmpMin,tmpMax);

	if (tmpMax==tmpMin)
	{
		for (i = 0; i < height; i++) 
		{
			for (j = 0; j < width; j++)
			{
				((uchar *)(dst->imageData + i*step))[j] = 0;
			}
		}
		
		if (pixFQ) {delete [] pixFQ;pixFQ = NULL;}
		if (pixScore) {delete [] pixScore;pixScore = NULL;}
	
		return 0;
	}
				
	//get map
	for (i = 0; i < height; i++) 
	{
		for (j = 0; j < width; j++)
		{
			sumIndex = 0;
			for(k=0;k<channels;k++)
			{
				tmpPixel = ((uchar *)(src->imageData + i*step))[j*channels + k];
				index = int(tmpPixel*1.0/blockPixel+0.5);	//norm	

				if (index >= numColorBlock)
					index = numColorBlock-1;

				if (k==0)
					sumIndex = index;
				else if (k==1)
					sumIndex += index*numColorBlock;
				else if (k==2)
					sumIndex += index*numColorBlock*numColorBlock;
			}
		
			//printf("sumIndex-%d\n",sumIndex);
			//printf("pixScore[%d]:%f,tmpMin:%f,tmpMax:%f\n",sumIndex,pixScore[sumIndex],tmpMin,tmpMax);
			tmpPixel = (int)( 255.0*((tmpMax-pixScore[sumIndex])*1.0/(tmpMax-tmpMin)) +0.5);
			if (tmpPixel<0)
				tmpPixel = 0;
			else if (tmpPixel>255)
				tmpPixel = 255;

			((uchar *)(dst->imageData+i*dst->widthStep ))[j] = tmpPixel;

			if (tmpPixel>=160)
				((uchar *)(dst2->imageData + i*dst2->widthStep))[j] = tmpPixel;
			else
				((uchar *)(dst2->imageData + i*dst2->widthStep))[j] = 0;

			if (tmpPixel>=200)
				((uchar *)(dst3->imageData + i*dst3->widthStep))[j] = tmpPixel;
			else
				((uchar *)(dst3->imageData + i*dst3->widthStep))[j] = 0;
		}
	}
	if (pixFQ) {delete [] pixFQ;pixFQ = NULL;}
	if (pixScore) {delete [] pixScore;pixScore = NULL;}
	
	return 0;
}


int API_FAST_SEGMENT::ColorHistogram(IplImage *image, int numColorBlock, vector< float > &Res)
{
	int nRet = TOK;
	int i,j,k,count,tmpPixel,index;
	
	int height     = image->height;
	int width      = image->width;
	int step       = image->widthStep;
	int channels   = image->nChannels;
	int blockPixel = int( 255.0/numColorBlock + 0.5 );
	double size = 1.0/ (height * width) ;

	int feat_len = 0;
	if ( channels == 3)
		feat_len = numColorBlock*numColorBlock*numColorBlock;
	else if ( channels == 1)
		feat_len = numColorBlock;
	else
		return -1;
	
	float *pCH = new float[feat_len];
	memset(pCH, 0, feat_len * sizeof(float));

	for (i = 0; i < height; i++) 
	{
		for (j = 0; j < width; j++)
		{
			index = 0;
			for(k=0;k<channels;k++)
			{
				tmpPixel = ((uchar *)(image->imageData + i*step))[j*channels + k];
				index = int(tmpPixel*1.0/blockPixel);	//norm	
				index >> 2;

				if ( (i<3) && (j<3) )
					printf("tmpPixel:%d,blockPixel:%d,index:%d\n",tmpPixel,blockPixel,index);
			}
			pCH[index] += 1;
		}
	}

	//norm
	Res.clear();
	for (i = 0; i < feat_len; i++) 
	{
		pCH[i] = pCH[i] * size ;
		Res.push_back(pCH[i]);
	}

	if (pCH) {delete [] pCH;pCH = NULL;}
	
	return nRet;
}




