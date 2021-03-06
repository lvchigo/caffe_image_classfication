#define _MAIN

#include <string>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h> 
#include <sys/types.h> 
#include <iostream>
#include <unistd.h>

#include <cv.h>
#include <cxcore.h>
#include <highgui.h>

#include "API_commen.h"
#include "TErrorCode.h"

#include <opencv/highgui.h>

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

#include <iostream>
#include <opencv2/opencv.hpp>

#include "RobustTextDetection.h"
#include "ConnectedComponent.h"

using namespace std;
using namespace cv;

int ads_textdetect( char *szQueryList, int inputDelta, char *svPath  )
{
	/*****************************Init*****************************/
	char loadImgPath[256];
	char szImgPath[256];
	int i, j, svImg, tank, nRet = 0;
	long inputLabel, nCount;
	double allGetLabelTime,tGetLabelTime;
	unsigned long long ImageID = 0;
	FILE *fpListFile = 0 ;

	API_COMMEN api_commen;

	int rWidth,rHeight;
	int MaxLen = 320;	//maxlen:720-same with online
	char imgPath[1024] = {0};

	CvRect tmpRect;
	double tmpProbability;
	map< CvRect ,double > mapRectScore;
	map< CvRect ,double >::iterator itRectScore;

	/*****************************Init*****************************/
	/* Quite a handful or params */
	RobustTextParam param;
	param.minMSERArea        = 10;
	param.maxMSERArea        = 2000;
	param.cannyThresh1       = 20;
	param.cannyThresh2       = 100;

	param.maxConnCompCount   = 3000;
	param.minConnCompArea    = 75;
	param.maxConnCompArea    = 600;

	param.minEccentricity    = 0.1;
	param.maxEccentricity    = 0.995;
	param.minSolidity        = 0.4;
	param.maxStdDevMeanRatio = 0.5;
	
	/********************************Open Query List*****************************/
	fpListFile = fopen(szQueryList,"r");
	if (!fpListFile) 
	{
		cout << "0.can't open " << szQueryList << endl;
		return TEC_INVALID_PARAM;
	}

	nCount = 0;
	/*****************************Process one by one*****************************/
	while(EOF != fscanf(fpListFile, "%s", loadImgPath ))
	{
		IplImage *img = cvLoadImage(loadImgPath);
		if(!img || (img->width<16) || (img->height<16) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
		{	
			cout<<"Can't open " << loadImgPath << endl;

			cvReleaseImage(&img);img = 0;
			continue;
		}	
		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld img...\n",nCount);

		tGetLabelTime = (double)getTickCount();
		/************************getRandomID*****************************/
		mapRectScore.clear();
		//api_commen.getRandomID( ImageID );
		ImageID = api_commen.GetIDFromFilePath(loadImgPath);

		if(!img || (img->width<16) || (img->height<16) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
		{	
			cout<<"image err!!" << endl;
			return TEC_INVALID_PARAM;
		}	

		/***********************************Resize Image width && height*****************************/
		IplImage *imgResize;
		if( ( img->width>MaxLen ) || ( img->height>MaxLen ) )
		{
			nRet = api_commen.GetReWH( img->width, img->height, MaxLen, rWidth, rHeight );	
			if (nRet != 0)
			{
			   	cout<<"GetReWH err!!" << endl;
				return TEC_INVALID_PARAM;
			}

			/*****************************Resize Img*****************************/
			imgResize = cvCreateImage(cvSize(rWidth, rHeight), img->depth, img->nChannels);
			cvResize( img, imgResize );
		}
		else
		{
			imgResize = cvCreateImage(cvSize(img->width, img->height), img->depth, img->nChannels);
			cvCopy( img, imgResize, NULL );
		}
			
		/*****************************push data*****************************/
		Mat imgMat( imgResize );
		
		/* Apply Robust Text Detection */
		/* ... remove this temp output path if you don't want it to write temp image files */
		string temp_output_path = "/Users/saburookita/Personal Projects/RobustTextDetection/";
		RobustTextDetection detector(param, temp_output_path );
		pair<Mat, Rect> result = detector.apply( imgMat );

		/* Get the region where the candidate text is */
		Mat stroke_width( result.second.height, result.second.width, CV_8UC1, Scalar(0) );
		Mat(result.first, result.second).copyTo( stroke_width);

		/* Append the original and stroke width images together */
/*	    cvtColor( stroke_width, stroke_width, CV_GRAY2BGR );
	    Mat appended( imgMat.rows, imgMat.cols + stroke_width.cols, CV_8UC3 );
	    imgMat.copyTo( Mat(appended, Rect(0, 0, imgMat.cols, imgMat.rows)) );
	    stroke_width.copyTo( Mat(appended, Rect(imgMat.cols, 0, stroke_width.cols, stroke_width.rows)) );*/
	
		sprintf(szImgPath, "%s/%ld_text.jpg", svPath, ImageID);
		imwrite(szImgPath, stroke_width);

		sprintf(szImgPath, "%s/%ld.jpg", svPath, ImageID);
		cvSaveImage(szImgPath, imgResize);
		
		/*********************************Release*************************************/
		cvReleaseImage(&img);img = 0;
		cvReleaseImage(&imgResize);imgResize = 0;

		tGetLabelTime = (double)getTickCount() - tGetLabelTime;
		tGetLabelTime = tGetLabelTime*1000./cv::getTickFrequency();
		allGetLabelTime += tGetLabelTime;
	}

	/*********************************close file*************************************/
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	

	/*********************************Print Info*********************************/
	if ( nCount != 0 ) 
	{
		printf( "nCount:%ld,GetLabelTime:%.4fms\n", nCount,allGetLabelTime*1.0/nCount );
	}
	
	cout<<"Done!! "<<endl;
	
	return nRet;
}

int main(int argc, char* argv[])
{
	int  ret = 0;

	if (argc == 5 && strcmp(argv[1],"-adstextdetect") == 0)
	{
		ads_textdetect(argv[2], atoi(argv[3]), argv[4] );
	}
	else
	{
		cout << "usage:\n" << endl;
		cout << "\tDemo_adstextdetect -adstextdetect queryList.txt delta svPath\n" << endl;
		return ret;
	}
	return ret;
}

