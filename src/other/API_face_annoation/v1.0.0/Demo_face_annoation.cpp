#include <string>
#include <vector>
#include <stdio.h>
#include <stdlib.h>		//do shell
#include <sys/stat.h> 
#include <sys/types.h> 
#include <iostream>
#include <unistd.h>
#include <map>

#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>

#include "API_commen.h"
#include "TErrorCode.h"
#include "plog/Log.h"

#include "API_faceannoation.h"

using namespace cv;
using namespace std;

int frcnn_test( char *szQueryList, char* KeyFilePath, char *layerName, int binGPU, int deviceID )
{
	char tPath[256];
	char loadImgPath[256];
	char szImgPath[256];
	char savePath[256];
	vector<string> vecText;
	int i, j, label, svImg, nRet = 0;
	long inputLabel, nCount, nCountFace;
	string strImageID,text,name;
	double allPredictTime;
	FILE *fpListFile = 0;

	RunTimer<double> run;
	API_COMMEN api_commen;
	API_FACE_ANNOATION api_face_annoation;

	vector< FaceAnnoationInfo > Res;

	/***********************************Init**********************************/
	plog::init(plog::info, "plog.txt"); 

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
		return TEC_INVALID_PARAM;
	}

	/***********************************Init*************************************/
	nRet = api_face_annoation.Init( KeyFilePath, binGPU, deviceID ); 
	if (nRet != 0)
	{
	   cout<<"Fail to initialization "<<endl;
	   return TEC_INVALID_PARAM;
	}

	nCount = 0;
	nCountFace = 0;
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
		
		/************************Predict*****************************/	
		Res.clear();
		run.start();
		nRet = api_face_annoation.Predict( img, Res );
		if ( (nRet!=0) || (Res.size()<1) )
		{
			LOOGE<<"[Predict Err!!loadImgPath:]"<<loadImgPath;
			cvReleaseImage(&img);img = 0;
			continue;
		}
		run.end();
		//LOOGI<<"[Predict] time:"<<run.time();
		allPredictTime += run.time();
		
		/************************save img data*****************************/
		{
			name = Res[0].label;
			if (name=="face")
				sprintf(tPath, "res_predict/face/", tPath );
			else
				sprintf(tPath, "res_predict/noface/", tPath );
		}
		for(i=0;i<Res.size();i++)  
		{						
			Scalar color = colors[i%8];

			if (Res[i].label == "face" )
			{
				api_face_annoation.Draw_Rotate_Box( img, Res[i], color );

				IplImage* Rotate_ROI = api_face_annoation.Get_Rotate_ROI( img, Res[i] );
				sprintf( savePath, "res_predict/roi/%s_%d.jpg", strImageID.c_str(), i );
				cvSaveImage( savePath, Rotate_ROI );
				cvReleaseImage(&Rotate_ROI);Rotate_ROI = 0;
			}
			
			cvRectangle( img, cvPoint(Res[i].rect[0], Res[i].rect[1]),
	                   cvPoint(Res[i].rect[2], Res[i].rect[3]), color, 2, 8, 0);

			sprintf(szImgPath, "%.2f %.2f %s", Res[i].score, Res[i].angle, Res[i].label.c_str() );
			text = szImgPath;
			cvPutText( img, text.c_str(), cvPoint(Res[i].rect[0]+1, Res[i].rect[1]+20), &font, color );

			for (j=0;j<5;j++)
				cvCircle( img, cvPoint(Res[i].annoation[2*j], Res[i].annoation[2*j+1]), 2, colors[j%8], 2, 8, 0 );

			sprintf(tPath, "%s%s-%.2f_", tPath, Res[i].label.c_str(), Res[i].score );

			if ( Res[i].label == "face" )
				nCountFace++;
		}
		sprintf( savePath, "%s%s.jpg", tPath, strImageID.c_str() );
		cvSaveImage( savePath, img );
			
		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld img...,time:%.4f\n",nCount,run.time());
		
		cvReleaseImage(&img);img = 0;
	}

	/*********************************close file*************************************/
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	

	/*********************************Release*************************************/
	api_face_annoation.Release();

	/*********************************Print Info*********************************/
	if ( nCount != 0 ) 
	{
		printf( "nCount:%ld,nCountFace:%ld_%.4f,PredictTime:%.4fms\n", 
			nCount, nCountFace, nCountFace*1.0/nCount, allPredictTime*1000.0/nCount );
	}
	
	cout<<"Done!! "<<endl;
	
	return nRet;	
}

int main(int argc, char* argv[])
{
	int  ret = 0;
	char szKeyFiles[256],szSavePath[256];
	API_COMMEN api_commen;

	if (argc == 7 && strcmp(argv[1],"-frcnn") == 0) {
		ret = frcnn_test( argv[2], argv[3], argv[4], atol(argv[5]), atol(argv[6]) );
	}
	else
	{
		cout << "usage:\n" << endl;
		cout << "\tDemo_facedetect -frcnn queryList.txt keyFilePath layerName binGPU deviceID\n" << endl;
		return ret;
	}
	return ret;
}
