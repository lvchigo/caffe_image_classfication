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
#include "API_maincolor.h"
#include "TErrorCode.h"

using namespace cv;
using namespace std;

int MainColor( char *szQueryList, int numColorBlock )
{
	/*****************************Init*****************************/
	char loadImgPath[256];
	char szImgPath[256];
	int i, j,m,n, label, svImg, rWidth, rHeight, nRet = 0;
	long inputLabel, nCount;
	unsigned long long ImageID = 0;
	double allGetLabelTime,tGetLabelTime;
	FILE *fpListFile = 0 ;

	API_COMMEN api_commen;
	API_MAINCOLOR_1_0_0 api_maincolor;
	vector< pair< vector< int >, float > > Res;
	
	/********************************Open Query List*****************************/
	fpListFile = fopen(szQueryList,"r");
	if (!fpListFile) 
	{
		cout << "0.can't open " << szQueryList << endl;
		return TEC_INVALID_PARAM;
	}

	nCount = 0;
	tGetLabelTime = 0.0;
	allGetLabelTime = 0.0;
	/*****************************Process one by one*****************************/
	while(EOF != fscanf(fpListFile, "%s", loadImgPath ))
	{
		IplImage *img = cvLoadImage(loadImgPath);
		if(!img || (img->width<16) || (img->height<16) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
		{	
			cout<<"Can't open " << loadImgPath << endl;
			continue;
		}	
		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld img...\n",nCount);

		/************************getRandomID*****************************/
		api_commen.getRandomID( ImageID );

		/*****************************GetReWH Img*****************************/
		nRet = api_commen.GetReWH( img->width, img->height, 64.0, rWidth, rHeight );
		if (nRet != 0)
		{	
			cout<<"Fail to GetReWH!! "<<endl;	
			return nRet;
		}
		
		/*****************************Resize Img*****************************/
		IplImage *img_resize = cvCreateImage(cvSize(rWidth, rHeight), img->depth, img->nChannels);
		cvResize( img, img_resize );

		/*****************************check Img*****************************/
		IplImage* correspond = cvCreateImage( cvSize(img_resize->width*2, img_resize->height*1), img_resize->depth, img_resize->nChannels );
		cvSetImageROI( correspond, cvRect( 0, 0, img_resize->width, img_resize->height ) );
	    cvCopy( img_resize, correspond );

		/************************change img format*****************************/
		uchar* bgr = api_commen.ipl2uchar( img_resize );
		
		/*****************************GetLabelFeat*****************************/	
		Res.clear();
		tGetLabelTime = (double)getTickCount();
		nRet = api_maincolor.Predict( bgr, img_resize->width, img_resize->height, img_resize->nChannels, numColorBlock, Res);	
		tGetLabelTime = (double)getTickCount() - tGetLabelTime;
		tGetLabelTime = tGetLabelTime*1000./cv::getTickFrequency();
		allGetLabelTime += tGetLabelTime;
		if (nRet != 0)
		{
		   cout<<"Fail to Predict!! "<<endl;
		   continue;
		}

		/*****************************check Img*****************************/
		IplImage* imgPredict = cvCreateImage( cvGetSize(img_resize), img_resize->depth, img_resize->nChannels );
		for (m = 0; m < img_resize->height; m++) 
		{
			for (n = 0; n < img_resize->width; n++) 
			{
				((uchar *)(imgPredict->imageData + m*img_resize->widthStep))[n*img_resize->nChannels + 0] = Res[0].first[0];	//B
				((uchar *)(imgPredict->imageData + m*img_resize->widthStep))[n*img_resize->nChannels + 1] = Res[0].first[1];	//G
				((uchar *)(imgPredict->imageData + m*img_resize->widthStep))[n*img_resize->nChannels + 2] = Res[0].first[2];	//R
			}
		}
		
		cvSetImageROI( correspond, cvRect( img_resize->width, 0, img_resize->width, img_resize->height ) );
		cvCopy( imgPredict, correspond );

		/*****************************output info*****************************/
	    cvResetImageROI( correspond );
		sprintf(szImgPath, "res/%.4f_%d_%d_%d_%ld.jpg",
				Res[0].second, Res[0].first[2], Res[0].first[1], Res[0].first[0], ImageID );
		cvSaveImage( szImgPath,correspond );

		/*********************************Release*************************************/
		delete bgr;
		cvReleaseImage(&img);img = 0;
		cvReleaseImage(&img_resize);img_resize = 0;
		cvReleaseImage(&correspond);correspond = 0;
		cvReleaseImage(&imgPredict);imgPredict = 0;
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

	if (argc == 4 && strcmp(argv[1],"-maincolor") == 0) {
		ret = MainColor( argv[2], atoi(argv[3]) );
	}
	else
	{
		cout << "usage:\n" << endl;
		cout << "\tDemo_maincolor -maincolor queryList.txt numColorBlocks\n" << endl;
		return ret;
	}
	return ret;
}
