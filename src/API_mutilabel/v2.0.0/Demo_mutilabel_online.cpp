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
#include "API_mutilabel/v2.0.0/API_mutilabel.h"
#include "API_commen/TErrorCode.h"
#include "plog/Log.h"

using namespace cv;
using namespace std;

int frcnn_test( char *szQueryList, char* svPath, char* KeyFilePath, float MutiLabel_T, int binGPU, int deviceID )
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
	API_MUTI_LABEL api_muti_label;

	vector< MutiLabelInfo > Res;

	/***********************************Init**********************************/
	unsigned long long		imageID;				//[In]:image ID for CheckData
	unsigned long long		childID;				//[In]:image child ID for CheckData
	sprintf( tPath, "res/plog.log" );
	plog::init(plog::error, tPath); 
	sprintf( tPath, "res/module-in-logo-detection.log" );
	plog::init<enum_module_in_logo>(plog::info, tPath, 100000000, 100000);

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
	nRet = api_muti_label.Init( KeyFilePath, binGPU, deviceID ); 
	if (nRet != 0)
	{
	   cout<<"Fail to initialization "<<endl;
	   return TEC_INVALID_PARAM;
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
		
		/************************Predict*****************************/	
		Res.clear();
		run.start();
		sprintf( tPath, "%d_%d", nCount, nCountObj );
		nRet = api_muti_label.Predict( img, string(tPath), nCount, nCountObj, MutiLabel_T, Res );
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
		for(i=0;i<Res.size();i++)  
		{						
			Scalar color = colors[i%8];
			cvRectangle( img, cvPoint(Res[i].rect[0], Res[i].rect[1]),
	                   cvPoint(Res[i].rect[2], Res[i].rect[3]), color, 2, 8, 0);

			sprintf(szImgPath, "%.2f %s", Res[i].score, Res[i].label.c_str() );
			text = szImgPath;
			cvPutText( img, text.c_str(), cvPoint(Res[i].rect[0]+1, Res[i].rect[1]+20), &font, color );

			//if (i<3)
			//	printf("Res:%.2f_%.2f_%.2f!!\n",Res[i].feat[0],Res[i].feat[1],Res[i].feat[2]);
		}
		sprintf( savePath, "%s/%s.jpg", svPath, strImageID.c_str() );
		cvSaveImage( savePath, img );

		nCountObj += Res.size();
		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld img...,time:%.4f\n",nCount,run.time());
		
		cvReleaseImage(&img);img = 0;
	}

	/*********************************close file*************************************/
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	

	/*********************************Release*************************************/
	api_muti_label.Release();

	/*********************************Print Info*********************************/
	if ( nCount != 0 ) 
	{
		printf( "nCount:%ld,nCountObj:%ld_%.4f,PredictTime:%.4fms\n", 
			nCount, nCountObj, nCountObj*1.0/nCount, allPredictTime*1000.0/nCount );
	}
	
	cout<<"Done!! "<<endl;
	
	return nRet;	
}

int frcnn_test_face( char *szQueryList, char* KeyFilePath, float MutiLabel_T, int binGPU, int deviceID )
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
	API_MUTI_LABEL api_muti_label;

	vector< MutiLabelInfo > Res_MultiLabel;
	vector< FaceAnnoationInfo > Res_FaceAnnoation;

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
	nRet = api_muti_label.Init( KeyFilePath, binGPU, deviceID ); 
	if (nRet != 0)
	{
	   cout<<"Fail to initialization "<<endl;
	   return TEC_INVALID_PARAM;
	}

	nCount = 0;
	nCountObj = 0;
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
		Res_MultiLabel.clear();
		run.start();
		sprintf( tPath, "%d_%d", nCount, nCountObj );
		nRet = api_muti_label.Predict( img, string(tPath), nCount, nCountObj, MutiLabel_T, Res_MultiLabel );
		if ( (nRet!=0) || (Res_MultiLabel.size()<1) )
		{
			LOOGE<<"[Predict Err!!loadImgPath:]"<<loadImgPath;
			cvReleaseImage(&img);img = 0;
			continue;
		}
		run.end();
		//LOOGI<<"[Predict] time:"<<run.time();
		allPredictTime += run.time();

		/***********************************Face:Predict**********************************/
		Res_FaceAnnoation.clear();
		run.start();
		nRet = api_muti_label.Face_Predict( img, Res_MultiLabel, Res_FaceAnnoation );
		if ( (nRet!=0) || (Res_FaceAnnoation.size()<1) )
		{
			LOOGE<<"[Face:Predict Err!!loadImgPath:]"<<loadImgPath;
			cvReleaseImage(&img);img = 0;
			continue;
		}
		run.end();
		//LOOGI<<"[Face:Predict] time:"<<run.time();
		allPredictTime += run.time();
		
		/************************save img data*****************************/
		{
			name = Res_FaceAnnoation[0].label;
			if (name=="person.face")
				sprintf(tPath, "res_predict/face/", tPath );
			else
				sprintf(tPath, "res_predict/noface/", tPath );
		}
		for(i=0;i<Res_FaceAnnoation.size();i++)  
		{						
			Scalar color = colors[i%8];

			if (Res_FaceAnnoation[i].label == "person.face" )
			{
				api_muti_label.Face_Draw_Rotate_Box( img, Res_FaceAnnoation[i], color );

				IplImage* Rotate_ROI = api_muti_label.Face_Get_Rotate_ROI( img, Res_FaceAnnoation[i] );
				sprintf( savePath, "res_predict/roi/%s_%d.jpg", strImageID.c_str(), i );
				cvSaveImage( savePath, Rotate_ROI );
				cvReleaseImage(&Rotate_ROI);Rotate_ROI = 0;
				nCountFace++;
			}
			
			cvRectangle( img, cvPoint(Res_FaceAnnoation[i].rect[0], Res_FaceAnnoation[i].rect[1]),
	                   cvPoint(Res_FaceAnnoation[i].rect[2], Res_FaceAnnoation[i].rect[3]), color, 2, 8, 0);

			sprintf(szImgPath, "%.2f %.2f %s", Res_FaceAnnoation[i].score, Res_FaceAnnoation[i].angle, Res_FaceAnnoation[i].label.c_str() );
			text = szImgPath;
			cvPutText( img, text.c_str(), cvPoint(Res_FaceAnnoation[i].rect[0]+1, Res_FaceAnnoation[i].rect[1]+20), &font, color );

			for (j=0;j<5;j++)
				cvCircle( img, cvPoint(Res_FaceAnnoation[i].annoation[2*j], Res_FaceAnnoation[i].annoation[2*j+1]), 2, colors[j%8], 2, 8, 0 );

			sprintf(tPath, "%s%s-%.2f_", tPath, Res_FaceAnnoation[i].label.c_str(), Res_FaceAnnoation[i].score );
		}
		sprintf( savePath, "%s%s.jpg", tPath, strImageID.c_str() );
		cvSaveImage( savePath, img );

		nCountObj += Res_MultiLabel.size();
		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld img...,time:%.4f\n",nCount,run.time());
		
		cvReleaseImage(&img);img = 0;
	}

	/*********************************close file*************************************/
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	

	/*********************************Release*************************************/
	api_muti_label.Release();

	/*********************************Print Info*********************************/
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
	char szKeyFiles[256],szSavePath[256];
	API_COMMEN api_commen;

	//inLabelClass:0-voc,1-coco,2-old in;
	if (argc == 8 && strcmp(argv[1],"-test") == 0) {
		ret = frcnn_test( argv[2], argv[3], argv[4], atof(argv[5]), atoi(argv[6]), atoi(argv[7]) );
	}
	else if (argc == 7 && strcmp(argv[1],"-test_face") == 0) {
		ret = frcnn_test_face( argv[2], argv[3], atof(argv[4]), atoi(argv[5]), atoi(argv[6]) );
	}
	else
	{
		cout << "usage:\n" << endl;
		cout << "\tDemo_mutilabel -test loadImagePath svPath keyfile MutiLabel_T binGPU deviceID\n" << endl;
		cout << "\tDemo_mutilabel -test_face loadImagePath keyfile MutiLabel_T binGPU deviceID\n" << endl;
		return ret;
	}
	return ret;
}

