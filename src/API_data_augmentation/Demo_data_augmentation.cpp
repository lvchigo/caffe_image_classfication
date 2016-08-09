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

#include "API_commen.h"
#include "API_data_augmentation.h"
#include "API_xml.h"	//read xml

using namespace cv;
using namespace std; 

int Get_Neg_Patch( char *szQueryList, char *loadXMLPath, char* svPath )
{
	char tPath[256];
	char loadImgPath[256];
	char szImgPath[256];
	char savePath[256];
	vector<string> vecText;
	int i, j, label, svImg, nRet = 0;
	long inputLabel, nCount, nCountObj;
	string strImageID,text,name;
	double allPredictTime;
	FILE *fpListFile = 0;

	RunTimer<double> run;
	API_COMMEN api_commen;
	API_DATA_AUGMENTATION api_DATA_AUGMENTATION;
	API_XML api_xml;

	vector < pair < string,Vec4i > > 	vecInLabelRect;
	vector < pair < string,Vec4i > > 	vecOutLabelRect;

	/***********************************Init**********************************/
	const static Scalar colors[] =  { 	CV_RGB(0,0,255),	CV_RGB(0,255,255),	CV_RGB(0,255,0),	
										CV_RGB(255,255,0),	CV_RGB(255,0,0),	CV_RGB(255,0,255),
										CV_RGB(0,128,255), 	CV_RGB(255,128,0)} ;

	CvFont font;
 	cvInitFont(&font,CV_FONT_HERSHEY_TRIPLEX,0.35f,0.7f,0,1,CV_AA);
	
	/********************************Open Query List*****************************/
	fpListFile = fopen(szQueryList,"r");
	if (!fpListFile) 
	{
		cout << "0.can't open " << szQueryList << endl;
		return -1;
	}

	nCount = 0;
	nCountObj = 0;
	allPredictTime = 0.0;
	/*****************************Process one by one*****************************/
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

		/************************getRandomID*****************************/
		strImageID = api_commen.GetStringIDFromFilePath( loadImgPath );

		//read_xml
		vecInLabelRect.clear();
		sprintf( tPath, "%s/%s.xml", loadXMLPath, strImageID.c_str() );
		nRet = api_xml.load_xml( string(tPath), strImageID, vecInLabelRect );
		if ( (nRet!=0) || (vecInLabelRect.size()<1) )
		{
			cout << "Err to load_xml!" << endl;
			continue;
		}

		/************************Get_Neg_Patch*****************************/
		nRet = api_DATA_AUGMENTATION.Get_Neg_Patch( vecInLabelRect, "other.logo", "other.other", img->width, img->height, 0.2, vecOutLabelRect );
		
		/************************save img data*****************************/
		for(i=0;i<vecOutLabelRect.size();i++)  
		{						
			Scalar color = colors[i%8];
			cvRectangle( img, cvPoint(vecOutLabelRect[i].second[0], vecOutLabelRect[i].second[1]),
	                   cvPoint(vecOutLabelRect[i].second[2], vecOutLabelRect[i].second[3]), color, 2, 8, 0);

			sprintf(szImgPath, "%s", vecOutLabelRect[i].first.c_str() );
			text = szImgPath;
			cvPutText( img, text.c_str(), cvPoint(vecOutLabelRect[i].second[0]+1, vecOutLabelRect[i].second[1]+20), &font, color );
		}

		//save checkimg
		sprintf( savePath, "%s/CheckImg/%s.jpg", svPath, strImageID.c_str() );
		cvSaveImage( savePath, img );

		//write_xml
		sprintf( tPath, "%s/Annoations/", svPath );
		nRet = api_xml.write_xml( strImageID, tPath, vecOutLabelRect );
		if (nRet!=0)
		{
			cout << "Err to write_xml!" << endl;
			cvReleaseImage(&img);img = 0;
			continue;
		}

		//cp img file
		sprintf( tPath, "cp %s %s/JPEGImages/", loadImgPath, svPath );
		system( tPath );

		nCountObj += vecOutLabelRect.size();
		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld img...,time:%.4f\n",nCount,run.time());
		
		cvReleaseImage(&img);img = 0;
	}

	/*********************************close file*************************************/
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}

	/*********************************Print Info*********************************/
	if ( nCount != 0 ) 
	{
		printf( "nCount:%ld,nCountObj:%ld_%.4f,PredictTime:%.4fms\n", 
			nCount, nCountObj, nCountObj*1.0/nCount, allPredictTime*1000.0/nCount );
	}
	
	cout<<"Done!! "<<endl;
	
	return nRet;	
}

int main(int argc, char* argv[])
{
	int  ret = 0;

	//inLabelClass:0-voc,1-coco,2-old in;
	if (argc == 5 && strcmp(argv[1],"-GetNegPatch") == 0) {
		ret = Get_Neg_Patch( argv[2], argv[3], argv[4] );
	}
	else
	{
		cout << "usage:\n" << endl;
		cout << "\tDemo_data_augmentation -GetNegPatch loadImagePath loadXMLPath svPath\n" << endl;
		return ret;
	}
	return ret;
}

