#include <string>
#include <vector>
#include <stdio.h>
#include <stdlib.h>		//do shell
#include <sys/stat.h> 
#include <sys/types.h> 
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <map>

#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>

#include "API_commen/API_commen.h"
#include "API_commen/TErrorCode.h"
#include "plog/Log.h"
#include "API_fast_segment/API_fast_segment.h"

using namespace cv;
using namespace std;

int test( char *szQueryList, char* KeyFilePath, char* svPath, float MutiLabel_T )
{
	char tPath[256];
	char loadImgPath[256];
	char szImgPath[256];
	char savePath[256];
	vector<string> vecText;
	int i, j, label, svImg, nRet = 0;
	long inputLabel, nCount, nCountObj, nCountFace;
	string strImageID,text,name;
	double allPredictTime;
	FILE *fpListFile = 0;

	RunTimer<double> run;
	API_COMMEN api_commen;
	API_FAST_SEGMENT api_fast_segment;

	//Init
	plog::init(plog::info, "plog.txt"); 

	const static Scalar colors[] =  { 	CV_RGB(0,0,255),	CV_RGB(0,255,255),	CV_RGB(0,255,0),	
										CV_RGB(255,255,0),	CV_RGB(255,0,0),	CV_RGB(255,0,255),
										CV_RGB(0,128,255), 	CV_RGB(255,128,0)} ;

	CvFont font;
 	cvInitFont(&font,CV_FONT_HERSHEY_TRIPLEX,0.35f,0.7f,0,1,CV_AA);
	
	//Open Query List
	fpListFile = fopen(szQueryList,"r");
	if (!fpListFile) 
	{
		cout << "0.can't open " << szQueryList << endl;
		return TEC_INVALID_PARAM;
	}

	nCount = 0;
	nCountObj = 0;
	nCountFace = 0;
	allPredictTime = 0.0;
	//Process one by one
	while(EOF != fscanf(fpListFile, "%s", loadImgPath))
	{
		IplImage *img = cvLoadImage(loadImgPath);
		if(!img || (img->width<32) || (img->height<32) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
		{	
			cout<<"Can't open " << loadImgPath << endl;
			cvReleaseImage(&img);img = 0;
			continue;
		}	
		//printf("loadImgPath:%s\n",loadImgPath);

		//getRandomID
		strImageID = api_commen.GetStringIDFromFilePath( loadImgPath );

		//Predict
		Vec4i rect;
		run.start();
		nRet = api_fast_segment.do_fast_segment( img, strImageID, rect );
		if (nRet!=0)
		{
			LOOGE<<"[Face:Predict Err!!loadImgPath:]"<<loadImgPath;
			cvReleaseImage(&img);img = 0;
			continue;
		}
		run.end();
		//LOOGI<<"[Face:Predict] time:"<<run.time();
		allPredictTime += run.time();

		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld img...,time:%.4f\n",nCount,run.time());
		
		cvReleaseImage(&img);img = 0;
	}

	//close file
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}

	//Print Info
	if ( nCount != 0 ) 
	{
		printf( "nCount:%ld,nCountObj:%ld_%.4f,nCountFace:%ld_%.4f,PredictTime:%.4fms\n", 
			nCount, nCountObj, nCountObj*1.0/nCount, nCountFace, nCountFace*1.0/nCount, allPredictTime*1000.0/nCount );
	}
	
	cout<<"Done!! "<<endl;
	
	return nRet;	
}

int main(int argc, char* argv[])
{
	int  ret = 0;

	if (argc == 6 && strcmp(argv[1],"-test") == 0) {
		ret = test( argv[2], argv[3], argv[4], atof(argv[5]) );
	}
	else
	{
		cout << "usage:\n" << endl;
		cout << "\tDemo_fast_segment -test loadImagePath keyfile svPath MutiLabel_T\n" << endl;
		return ret;
	}
	return ret;
}

