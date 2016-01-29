#include <string>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h> 
#include <sys/types.h> 
#include <iostream>
#include <unistd.h>
#include <map>

#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>

#include "API_commen.h"
#include "API_caffe.h"
#include "API_mainboby.h"
#include "API_xml.h"	//read xml
#include "TErrorCode.h"
#include "plog/Log.h"

#include "kyheader.h"
#include "Objectness_predict.h"
#include "ValStructVec.h"
#include "CmShow.h"

using namespace cv;
using namespace std;

int Get_Bing_ROI( char *szQueryList, char* KeyFilePath, char *layerName, int binGPU, int deviceID )
{
	char tPath[256];
	char loadImgPath[256];
	char szImgPath[256];
	int i, j, label, svImg, nRet = 0;
	int topN=20;
	long inputLabel, nCount;
	unsigned long long ImageID = 0;
	double allPredictTime;
	FILE *fpListFile = 0;

	RunTimer<double> run;
	API_COMMEN api_commen;
	API_MAINBOBY api_mainboby;

	vector < pair < string,float > > Res;

	/***********************************Init*************************************/
	ValStructVec<float, Vec4i> boxHypothese;
	const static Scalar colors[] =  { 	CV_RGB(0,0,255),	CV_RGB(0,255,255),	CV_RGB(0,255,0),	
										CV_RGB(255,255,0),	CV_RGB(255,0,0),	CV_RGB(255,0,255),
										CV_RGB(0,128,255), 	CV_RGB(255,128,0)} ;
	/***********************************Init**********************************/
	plog::init(plog::info, "plog.txt"); 
	
	/********************************Open Query List*****************************/
	fpListFile = fopen(szQueryList,"r");
	if (!fpListFile) 
	{
		cout << "0.can't open " << szQueryList << endl;
		return TEC_INVALID_PARAM;
	}

	/***********************************Init*************************************/
	nRet = api_mainboby.Init( KeyFilePath, layerName, binGPU, deviceID ); 
	if (nRet != 0)
	{
	   cout<<"Fail to initialization "<<endl;
	   return TEC_INVALID_PARAM;
	}

	nCount = 0;
	allPredictTime = 0.0;
	/*****************************Process one by one*****************************/
	while(EOF != fscanf(fpListFile, "%s", loadImgPath))
	{
		IplImage *img = cvLoadImage(loadImgPath);
		if(!img || (img->width<128) || (img->height<128) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
		{	
			cout<<"Can't open " << loadImgPath << endl;
			cvReleaseImage(&img);img = 0;
			continue;
		}	
		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld img...\n",nCount);

		/************************getRandomID*****************************/
		api_commen.getRandomID( ImageID );
		//ImageID = api_commen.GetIDFromFilePath( loadImgPath );

		/************************ResizeImg*****************************/
		IplImage* imgResize = api_mainboby.ResizeImg( img );

		/************************Get_Hypothese*****************************/	
		boxHypothese.clear();
		run.start();
		nRet = api_mainboby.Get_Hypothese( imgResize, boxHypothese );
		//nRet = api_mainboby.Get_Hypothese_Entropy( img, ImageID, boxHypothese );
		if ( (nRet!=0) || (boxHypothese.size()<1) )
		{
			LOOGE<<"[Get_Hypothese Err!!loadImgPath:]"<<loadImgPath;
			cvReleaseImage(&imgResize);imgResize = 0;
			cvReleaseImage(&img);img = 0;
			continue;
		}
		run.end();
		LOOGI<<"[Get_Hypothese] time:"<<run.time();
		allPredictTime += run.time();

		Mat matImg(imgResize);
		/************************save img data*****************************/
		for(i=0;i<boxHypothese.size();i++) 
		//for(i=boxTests.size()-1;i>boxTests.size()-1-topN;i--) 
		{	
			Scalar color = colors[i%8];
			rectangle( matImg, cvPoint(boxHypothese[i][0], boxHypothese[i][1]),
	                   cvPoint(boxHypothese[i][2], boxHypothese[i][3]), color, 3, 8, 0);
			sprintf(szImgPath, "res_Hypothese/%lld_res.jpg",ImageID);
			imwrite( szImgPath, matImg );
		}

		cvReleaseImage(&imgResize);imgResize = 0;
		cvReleaseImage(&img);img = 0;
	}

	/*********************************close file*************************************/
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	

	/*********************************Release*************************************/
	api_mainboby.Release();

	/*********************************Print Info*********************************/
	if ( nCount != 0 ) 
	{
		printf( "nCount:%ld,PredictTime:%.4fms\n", nCount, allPredictTime*1000.0/nCount );
	}
	
	cout<<"Done!! "<<endl;
	
	return nRet;
}	

int Predict_Hypothese( char *szQueryList, char* KeyFilePath, char *layerName, int binGPU, int deviceID )
{
	char tPath[256];
	char loadImgPath[256];
	char szImgPath[256];
	int i, j, label, svImg, nRet = 0;
	int topN=5;
	long inputLabel, nCount;
	unsigned long long ImageID = 0;
	double allPredictTime;
	FILE *fpListFile = 0;

	RunTimer<double> run;
	API_COMMEN api_commen;
	API_MAINBOBY api_mainboby;

	vector < pair < string,float > > Res;

	/***********************************Init**********************************/
	plog::init(plog::info, "plog.txt"); 
	
	/********************************Open Query List*****************************/
	fpListFile = fopen(szQueryList,"r");
	if (!fpListFile) 
	{
		cout << "0.can't open " << szQueryList << endl;
		return TEC_INVALID_PARAM;
	}

	/***********************************Init*************************************/
	nRet = api_mainboby.Init( KeyFilePath, layerName, binGPU, deviceID ); 
	if (nRet != 0)
	{
	   cout<<"Fail to initialization "<<endl;
	   return TEC_INVALID_PARAM;
	}

	nCount = 0;
	allPredictTime = 0.0;
	/*****************************Process one by one*****************************/
	while(EOF != fscanf(fpListFile, "%s", loadImgPath))
	{
		IplImage *img = cvLoadImage(loadImgPath);
		if(!img || (img->width<128) || (img->height<128) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
		{	
			cout<<"Can't open " << loadImgPath << endl;
			cvReleaseImage(&img);img = 0;
			continue;
		}	
		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld img...\n",nCount);

		/************************getRandomID*****************************/
		api_commen.getRandomID( ImageID );
		//ImageID = api_commen.GetIDFromFilePath( loadImgPath );

		/************************ResizeImg*****************************/
		IplImage* imgResize = api_mainboby.ResizeImg( img );

		/************************Get_Hypothese*****************************/	
		run.start();
		nRet = api_mainboby.Predict_Hypothese( imgResize, ImageID, layerName );
		if (nRet!=0)
		{
			LOOGE<<"[Predict_Hypothese Err!!loadImgPath:]"<<loadImgPath;
			cvReleaseImage(&imgResize);imgResize = 0;
			cvReleaseImage(&img);img = 0;
			continue;
		}
		run.end();
		LOOGI<<"[Predict_Hypothese] time:"<<run.time();
		allPredictTime += run.time();

		cvReleaseImage(&imgResize);imgResize = 0;
		cvReleaseImage(&img);img = 0;
	}

	/*********************************close file*************************************/
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	

	/*********************************Release*************************************/
	api_mainboby.Release();

	/*********************************Print Info*********************************/
	if ( nCount != 0 ) 
	{
		printf( "nCount:%ld,PredictTime:%.4fms\n", nCount, allPredictTime*1000.0/nCount );
	}
	
	cout<<"Done!! "<<endl;
	
	return nRet;
}	

int Get_Xml_ROI( char *szQueryList, char* loadXmlPath, char* svImagePath )
{
	char loadImgPath[256];
	char szImgPath[256];
	char svImgFile[256];
	int i, j, label, svImg, binHypothese, nRet = 0;
	int roiNum, topN=15;
	long inputLabel, nCount, svBingPos, svBingNeg, countIOU_0_7;
	unsigned long long ImageID = 0;
	string strImageID;
	double allPredictTime;
	FILE *fpListFile = 0;

	RunTimer<double> run;
	API_COMMEN api_commen;
	API_XML api_xml;

	/***********************************Init**********************************/
	plog::init(plog::info, "plog.txt"); 

	string xmlImageID;
	vector< pair< string, Vec4i > > vecXmlLabelRect;
	vector< pair< string, Vec4i > > vecOutXmlRect;
	
	/********************************Open Query List*****************************/
	fpListFile = fopen(szQueryList,"r");
	if (!fpListFile) 
	{
		cout << "0.can't open " << szQueryList << endl;
		return TEC_INVALID_PARAM;
	}

	nCount = 0;
	allPredictTime = 0.0;
	/*****************************Process one by one*****************************/
	while(EOF != fscanf(fpListFile, "%s", loadImgPath))
	{
		IplImage *img = cvLoadImage(loadImgPath);
		if(!img || (img->width<128) || (img->height<128) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
		{	
			cout<<"Can't open " << loadImgPath << endl;
			cvReleaseImage(&img);img = 0;
			continue;
		}	

		/************************getRandomID*****************************/
		strImageID = api_commen.GetStringIDFromFilePath( loadImgPath );

		/************************loadXml*****************************/
		vecXmlLabelRect.clear();
		sprintf(szImgPath, "%s/%s.xml",loadXmlPath,strImageID.c_str());
		nRet = api_xml.load_xml( szImgPath, xmlImageID, vecXmlLabelRect );
		if ( (nRet!=0) || (vecXmlLabelRect.size()<1) )
		{
			LOOGE<<"[loadXml Err!!loadXml:]"<<szImgPath;
			cvReleaseImage(&img);img = 0;
			continue;
		}

		/************************get xml Hypothese*****************************/
		vecOutXmlRect.clear();
		binHypothese = 1; //0-normal,1-for mainbody v1.0.0
		nRet = api_xml.Get_Xml_Hypothese( xmlImageID, img->width, img->height, vecXmlLabelRect, vecOutXmlRect, binHypothese );
		if ( (nRet!=0) || (vecOutXmlRect.size()<1) )
		{
			LOOGE<<"[get_xml_Hypothese Err!!xmlImageID:]"<<xmlImageID;
			cvReleaseImage(&img);img = 0;
			continue;
		}

		/************************save pos roi data*****************************/
		for( i=0;i<vecOutXmlRect.size();i++)
		{
			cvSetImageROI( img,cvRect(vecOutXmlRect[i].second[0],vecOutXmlRect[i].second[1], 
				(vecOutXmlRect[i].second[2]-vecOutXmlRect[i].second[0]),
				(vecOutXmlRect[i].second[3]-vecOutXmlRect[i].second[1]) ) );	//for imagequality
			IplImage* MutiROI = cvCreateImage(cvGetSize(img),img->depth,img->nChannels);
			cvCopy( img, MutiROI, NULL );
			cvResetImageROI(img);	
			
			/*****************************Resize Img*****************************/
			IplImage *MutiROIResize = cvCreateImage(cvSize(256, 256), img->depth, img->nChannels);
			cvResize( MutiROI, MutiROIResize );

			sprintf( svImgFile, "%s/%s/%s_%d.jpg", svImagePath, vecOutXmlRect[i].first.c_str(), xmlImageID.c_str(), i );
			cvSaveImage( svImgFile, MutiROIResize );

			cvReleaseImage(&MutiROIResize);MutiROIResize = 0;
			cvReleaseImage(&MutiROI);MutiROI = 0;
		}

		nCount++;
		if( nCount%50 == 0 )
		{
			printf("Loaded %ld img...\n",nCount);
		}

		cvReleaseImage(&img);img = 0;
	}

	/*********************************close file*************************************/
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	

	/*********************************Print Info*********************************/
	if ( nCount != 0 ) 
	{
		printf( "nCount:%ld,PredictTime:%.4fms\n", nCount, allPredictTime*1000.0/nCount );
	}
	
	cout<<"Done!! "<<endl;
	
	return nRet;
}	

int Get_Xml_Bing_ROI_for_1_0_0( char *szQueryList, char* loadXmlPath, char* svImagePath, char* KeyFilePath, char *layerName, int binGPU, int deviceID )
{
	char tPath[256];
	char loadImgPath[256];
	char szImgPath[256];
	char svImgFile[256];
	int i, j, label, svImg, width, height, nRet = 0;
	int roiNum, topN=15;
	long inputLabel, nCount, svBingPos, svBingNeg, countIOU_0_7, nNegCount;
	unsigned long long ImageID = 0;
	string strImageID;
	double allPredictTime;
	FILE *fpListFile = 0;

	RunTimer<double> run;
	API_COMMEN api_commen;
	API_MAINBOBY api_mainboby;
	API_XML api_xml;

	vector < pair < string,float > > Res;

	/***********************************Init*************************************/
	vector< pair<float, Vec4i> > boxHypothese;
	const static Scalar colors[] =  { 	CV_RGB(0,0,255),	CV_RGB(0,255,255),	CV_RGB(0,255,0),	
										CV_RGB(255,255,0),	CV_RGB(255,0,0),	CV_RGB(255,0,255),
										CV_RGB(0,128,255), 	CV_RGB(255,128,0)} ;
	/***********************************Init**********************************/
	plog::init(plog::info, "plog.txt"); 

	string xmlImageID;
	vector< pair< string, Vec4i > > vecXmlLabelRect;
	vector< pair< string, Vec4i > > vecOutXmlRect;
	
	/********************************Open Query List*****************************/
	fpListFile = fopen(szQueryList,"r");
	if (!fpListFile) 
	{
		cout << "0.can't open " << szQueryList << endl;
		return TEC_INVALID_PARAM;
	}

	/***********************************Init*************************************/
	nRet = api_mainboby.Init( KeyFilePath, layerName, binGPU, deviceID ); 
	if (nRet != 0)
	{
	   cout<<"Fail to initialization "<<endl;
	   return TEC_INVALID_PARAM;
	}

	nCount = 0;
	allPredictTime = 0.0;
	/*****************************Process one by one*****************************/
	while(EOF != fscanf(fpListFile, "%s", loadImgPath))
	{
		IplImage *img = cvLoadImage(loadImgPath);
		if(!img || (img->width<128) || (img->height<128) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
		{	
			cout<<"Can't open " << loadImgPath << endl;
			continue;
		}	

		/************************getRandomID*****************************/
		strImageID = api_commen.GetStringIDFromFilePath( loadImgPath );

		/************************loadXml*****************************/
		vecXmlLabelRect.clear();
		sprintf(szImgPath, "%s/%s.xml",loadXmlPath,strImageID.c_str());
		nRet = api_xml.load_xml( szImgPath, xmlImageID, vecXmlLabelRect );
		if ( (nRet!=0) || (vecXmlLabelRect.size()<1) )
		{
			LOOGE<<"[loadXml Err!!loadXml:]"<<szImgPath;
			continue;
		}

		/************************get xml Hypothese*****************************/
		vecOutXmlRect.clear();
		int binHypothese = 1; //0-normal,1-for mainbody v1.0.0
		nRet = api_xml.Get_Xml_Hypothese( xmlImageID, img->width, img->height, vecXmlLabelRect, vecOutXmlRect, binHypothese );
		if ( (nRet!=0) || (vecOutXmlRect.size()<1) )
		{
			LOOGE<<"[get_xml_Hypothese Err!!xmlImageID:]"<<xmlImageID;
			continue;
		}

		/************************Get_Bing_Hypothese*****************************/	
		boxHypothese.clear();
		run.start();
		//BinTraining:2-NO Remove Rectfor Training;1-Remove small Rect for Training;0-Remove small Rect for Test
		nRet = api_mainboby.Get_Bing_Hypothese( img, boxHypothese, 2 );
		if ( (nRet!=0) || (boxHypothese.size()<1) )
		{
			LOOGE<<"[Get_Bing_Hypothese Err!!loadImgPath:]"<<loadImgPath;
			continue;
		}
		run.end();
		LOOGI<<"[Get_Bing_Hypothese] time:"<<run.time();
		allPredictTime += run.time();

		/************************Get_iou_cover*****************************/
		vector< pair<Vec4i, pair< double, double > > > vectorIouCover;
		nRet = api_mainboby.Get_iou_cover( boxHypothese, vecOutXmlRect, vectorIouCover );
		if ( (nRet!=0) || (vectorIouCover.size()<1) )
		{
			LOOGE<<"[Get_iou_cover Err!!loadImgPath:]"<<loadImgPath;
			continue;
		}

		/************************save pos roi data*****************************/
/*		for( i=0;i<vecOutXmlRect.size();i++)
		{
			cvSetImageROI( img,cvRect(vecOutXmlRect[i].second[0],vecOutXmlRect[i].second[1], 
				(vecOutXmlRect[i].second[2]-vecOutXmlRect[i].second[0]),
				(vecOutXmlRect[i].second[3]-vecOutXmlRect[i].second[1]) ) );	//for imagequality
			IplImage* MutiROI = cvCreateImage(cvGetSize(img),img->depth,img->nChannels);
			cvCopy( img, MutiROI, NULL );
			cvResetImageROI(img);	
			
			//
			IplImage *MutiROIResize = cvCreateImage(cvSize(256, 256), img->depth, img->nChannels);
			cvResize( MutiROI, MutiROIResize );

			sprintf( svImgFile, "%s/%s/%s_%d.jpg", svImagePath, vecOutXmlRect[i].first.c_str(), xmlImageID.c_str(), i );
			cvSaveImage( svImgFile, MutiROIResize );

			cvReleaseImage(&MutiROIResize);MutiROIResize = 0;
			cvReleaseImage(&MutiROI);MutiROI = 0;
		}*/

		/************************save neg roi data*****************************/
		nNegCount = 0;
		for( i=0;i<vectorIouCover.size();i++)
		{
			width = vectorIouCover[i].first[2]-vectorIouCover[i].first[0];
			height = vectorIouCover[i].first[3]-vectorIouCover[i].first[1];
			if ( ( vectorIouCover[i].second.first<=0.2 ) && (vectorIouCover[i].second.second<0.2) &&
				 ( width>=64) && ( height>=64) && ( width<256) && ( height<256) )
			{
				cvSetImageROI( img,cvRect(vectorIouCover[i].first[0],vectorIouCover[i].first[1], 
					(vectorIouCover[i].first[2]-vectorIouCover[i].first[0]),
					(vectorIouCover[i].first[3]-vectorIouCover[i].first[1]) ) );	//for imagequality
				IplImage* MutiROI = cvCreateImage(cvGetSize(img),img->depth,img->nChannels);
				cvCopy( img, MutiROI, NULL );
				cvResetImageROI(img);	
				
				/*****************************Resize Img*****************************/
				IplImage *MutiROIResize = cvCreateImage(cvSize(256, 256), img->depth, img->nChannels);
				cvResize( MutiROI, MutiROIResize );

				sprintf( svImgFile, "%s/neg/%s_%d.jpg", svImagePath, xmlImageID.c_str(), i );
				cvSaveImage( svImgFile, MutiROIResize );

				cvReleaseImage(&MutiROIResize);MutiROIResize = 0;
				cvReleaseImage(&MutiROI);MutiROI = 0;

				nNegCount++;
			}

			if( nNegCount>4 )
				break;
		}

		nCount++;
		if( nCount%50 == 0 )
		{
			printf("Loaded %ld img...\n",nCount);
		}

		cvReleaseImage(&img);img = 0;
	}

	/*********************************close file*************************************/
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	

	/*********************************Release*************************************/
	api_mainboby.Release();

	/*********************************Print Info*********************************/
	if ( nCount != 0 ) 
	{
		printf( "nCount:%ld,PredictTime:%.4fms\n", nCount, allPredictTime*1000.0/nCount );
	}
	
	cout<<"Done!! "<<endl;
	
	return nRet;
}	


int Get_fineturn_sample( char *szQueryList, char* svImagePath, char* label )
{
	char loadImgPath[256];
	char szImgPath[256];
	char svImgFile[256];
	int i, j, svImg, binHypothese, nRet = 0;
	int roiNum, topN=15;
	long inputLabel, nCount, maxLabelNum, svBingPos, svBingNeg, countIOU_0_7;
	unsigned long long ImageID = 0;
	string strImageID;
	double allPredictTime;
	FILE *fpListFile = 0;

	RunTimer<double> run;
	API_COMMEN api_commen;
	API_XML api_xml;

	/***********************************Init**********************************/
	plog::init(plog::info, "plog.txt"); 
	vector< string > vecImgPath;
	
	/********************************Open Query List*****************************/
	fpListFile = fopen(szQueryList,"r");
	if (!fpListFile) 
	{
		cout << "0.can't open " << szQueryList << endl;
		return TEC_INVALID_PARAM;
	}

	nCount = 0;
	allPredictTime = 0.0;
	vecImgPath.clear();
	/*****************************Process one by one*****************************/
	while(EOF != fscanf(fpListFile, "%s", loadImgPath))
	{
		IplImage *img = cvLoadImage(loadImgPath);
		if(!img || (img->width<128) || (img->height<128) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
		{	
			cout<<"Can't open " << loadImgPath << endl;
			cvReleaseImage(&img);img = 0;
			continue;
		}	

		vecImgPath.push_back( string(loadImgPath) );
		cvReleaseImage(&img);img = 0;
	}

	maxLabelNum = int(0.1*vecImgPath.size()+0.5);

	for(i=0;i<vecImgPath.size();i++)
	{
		IplImage *img = cvLoadImage(vecImgPath[i].c_str());
		if(!img || (img->width<128) || (img->height<128) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
		{	
			cout<<"Can't open " << vecImgPath[i].c_str() << endl;
			cvReleaseImage(&img);img = 0;
			continue;
		}	
		
		/************************getRandomID*****************************/
		strImageID = api_commen.GetStringIDFromFilePath( vecImgPath[i].c_str() );
			
		/*****************************Resize Img*****************************/
		IplImage *imgResize = cvCreateImage(cvSize(256, 256), img->depth, img->nChannels);
		cvResize( img, imgResize );

		if (nCount < maxLabelNum)
			sprintf( svImgFile, "%s/val/%s/%s.jpg", svImagePath, label, strImageID.c_str() );
		else if (nCount < 2*maxLabelNum)
			sprintf( svImgFile, "%s/test/%s/%s.jpg", svImagePath, label, strImageID.c_str() );
		else
			sprintf( svImgFile, "%s/train/%s/%s.jpg", svImagePath, label, strImageID.c_str() );
		cvSaveImage( svImgFile, imgResize );

		cvReleaseImage(&imgResize);imgResize = 0;
		cvReleaseImage(&img);img = 0;
		
		nCount++;
		if( nCount%50 == 0 )
		{
			printf("Loaded %ld img...\n",nCount);
		}

		
	}

	/*********************************close file*************************************/
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	

	/*********************************Print Info*********************************/
	if ( nCount != 0 ) 
	{
		printf( "nCount:%ld,PredictTime:%.4fms\n", nCount, allPredictTime*1000.0/nCount );
	}
	
	cout<<"Done!! "<<endl;
	
	return nRet;
}	

int DL_ExtractFeat( char *szQueryList, char *szFeatResult, char *szKeyFiles, char *layerName, int binGPU, int deviceID)
{
	/*****************************Init*****************************/
	char loadImgPath[256];
	char szImgPath[256];
	int i, j, label, svImg, nRet = 0;
	long inputLabel, nCount;
	unsigned long long ImageID = 0;
	double allGetLabelTime,tGetLabelTime;
	FILE *fpListFile = 0 , *fpFeatOut = 0;
	
	vector< vector<float> > imgFeat;
	vector< vector<float> > vecNormDLFeat;
	vector< pair< int, float > > imgLabel;
	vector<float> normDLFeat;

	API_COMMEN api_commen;
	API_CAFFE api_caffe;
	
	/***********************************Init*************************************/
	char DL_DeployFile[1024] = {0};
	char DL_ModelFile[1024] = {0};
	char DL_Meanfile[1024] = {0};

	sprintf(DL_DeployFile, "%s/vgg_16/deploy_vgg_16.prototxt",szKeyFiles);
	sprintf(DL_ModelFile, "%s/vgg_16/VGG_ILSVRC_16_layers.caffemodel",szKeyFiles);
	sprintf(DL_Meanfile, "%s/vgg_16/imagenet_mean.binaryproto",szKeyFiles);	//vgg:add 2dcode
	nRet = api_caffe.Init( DL_DeployFile, DL_ModelFile, DL_Meanfile, layerName, binGPU, deviceID ); 
	if (nRet != 0)
	{
	   cout<<"Fail to initialization "<<endl;
	   return TEC_INVALID_PARAM;
	}
	
	/********************************Open Query List*****************************/
	fpListFile = fopen(szQueryList,"r");
	if (!fpListFile) 
	{
		cout << "0.can't open " << szQueryList << endl;
		return TEC_INVALID_PARAM;
	}

	fpFeatOut = fopen(szFeatResult, "wt");
	if (!fpFeatOut)
	{
		cout << "Can't open result file " << szFeatResult << endl;
		return TEC_INVALID_PARAM;
	}

	nCount = 0;
	tGetLabelTime = 0.0;
	allGetLabelTime = 0.0;
	/*****************************Process one by one*****************************/
	while(EOF != fscanf(fpListFile, "%s %d", &loadImgPath, &label))
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

		/************************getRandomID*****************************/
		api_commen.getRandomID( ImageID );

		/*****************************GetMutiImg*****************************/
		vector < Mat_ < Vec3f > > img_dl;
		nRet = api_commen.Img_GetMutiRoi( img, ImageID, img_dl );
		if ( ( nRet != 0) || ( img_dl.size() < 1 ) )
		{
			cout<<"Fail to Img_GetMutiRoi!! "<<endl; 
			cvReleaseImage(&img);img = 0;
			continue;
		}	

/*		//save
		for(i=0;i<img_dl.size();i++)
		{
			sprintf(szImgPath, "res/Img_GetMutiRoi_%ld_%d.jpg",ImageID,i);
			imwrite( szImgPath, img_dl[i] );
		}*/
		
		/*****************************GetLabelFeat*****************************/	
		imgLabel.clear();
		imgFeat.clear();	
		int bExtractFeat = 2;		//[In]:Get Label(1),Extract Feat(2),both(3)
		tGetLabelTime = (double)getTickCount();
		nRet = api_caffe.GetLabelFeat( img_dl, ImageID, layerName, bExtractFeat, imgLabel, imgFeat);	
		tGetLabelTime = (double)getTickCount() - tGetLabelTime;
		tGetLabelTime = tGetLabelTime*1000./cv::getTickFrequency();
		allGetLabelTime += tGetLabelTime;
		if ( (nRet != 0) || (imgFeat.size()<1) )
		{
		   	cout<<"Fail to GetFeat!! "<<endl;
			cvReleaseImage(&img);img = 0;
		   	continue;
		}

		/************************Normal Feat*****************************/
		vecNormDLFeat.clear();
		//printf("imgFeat.size()-%d:",imgFeat.size());
		for(i=0;i<imgFeat.size();i++)	//imgFeat.size()=2:1-ads/in36class,2-imagequality;
		{
			normDLFeat.clear();

			/************************DL Feat Normal*****************************/
			nRet = api_commen.Normal_L2(imgFeat[i],normDLFeat);
			if (nRet != 0)
			{
			   	cout<<"Fail to Normal!! "<<endl;
				cvReleaseImage(&img);img = 0;
			   	continue;
			}

			vecNormDLFeat.push_back( normDLFeat );
			
			//for(j=0;j<normDLFeat.size();j++)	//imgFeat.size()=2:1-ads/in36class,2-imagequality;
			//{
			//	printf("%d-%.4f ",j,normDLFeat[j]);
			//}
		}
		//printf("\n\n");

		/************************Save GetFeat*****************************/
		for ( i=0;i<vecNormDLFeat.size();i++ )
		{
			fprintf(fpFeatOut, "%d ", label );
			for ( j=0;j<vecNormDLFeat[i].size();j++ )
			{
				fprintf(fpFeatOut, "%d:%.6f ", j+1, (vecNormDLFeat[i][j]+0.00000001) );
			}
			fprintf(fpFeatOut, "\n");
		}	

		/*********************************Release*************************************/
		cvReleaseImage(&img);img = 0;
	}

	/*********************************Release*************************************/
	api_caffe.Release();

	/*********************************close file*************************************/
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	
	if (fpFeatOut) {fclose(fpFeatOut);fpFeatOut = 0;}

	/*********************************Print Info*********************************/
	if ( nCount != 0 ) 
	{
		printf( "nCount:%ld,GetLabelTime:%.4fms\n", nCount,allGetLabelTime*1.0/nCount );
	}
	
	cout<<"Done!! "<<endl;
	
	return nRet;
}

int SVM_Predict( char *szQueryList, char* KeyFilePath, char *layerName, int binGPU, int deviceID )
{
	char tPath[256];
	char loadImgPath[256];
	char szImgPath[256];
	string text;
	vector<string> vecText;
	int i, j, label, svImg, nRet = 0;
	int topN=5;
	long allLabel, nCount, tmpCount;
	unsigned long long ImageID = 0;
	double allPredictTime;
	FILE *fpListFile = 0;

	RunTimer<double> run;
	API_COMMEN api_commen;
	API_MAINBOBY api_mainboby;

	map<string, long> map_Num_Res_Label;
	map<string, long>::iterator it_Num_Res_Label;

	vector< pair< pair< string, Vec4i >, float > > Res;

	/***********************************Init**********************************/
	plog::init(plog::info, "plog.txt"); 

	const static Scalar colors[] =  { 	CV_RGB(0,0,255),	CV_RGB(0,255,255),	CV_RGB(0,255,0),	
										CV_RGB(255,255,0),	CV_RGB(255,0,0),	CV_RGB(255,0,255),
										CV_RGB(0,128,255), 	CV_RGB(255,128,0)} ;
	
	/********************************Open Query List*****************************/
	fpListFile = fopen(szQueryList,"r");
	if (!fpListFile) 
	{
		cout << "0.can't open " << szQueryList << endl;
		return TEC_INVALID_PARAM;
	}

	/***********************************Init*************************************/
	nRet = api_mainboby.Init( KeyFilePath, layerName, binGPU, deviceID ); 
	if (nRet != 0)
	{
	   cout<<"Fail to initialization "<<endl;
	   return TEC_INVALID_PARAM;
	}

	nCount = 0;
	allPredictTime = 0.0;
	/*****************************Process one by one*****************************/
	while(EOF != fscanf(fpListFile, "%s", loadImgPath))
	{
		IplImage *img = cvLoadImage(loadImgPath);
		if(!img || (img->width<128) || (img->height<128) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
		{	
			cout<<"Can't open " << loadImgPath << endl;
			cvReleaseImage(&img);img = 0;
			continue;
		}	
		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld img...\n",nCount);

		/************************getRandomID*****************************/
		api_commen.getRandomID( ImageID );
		//ImageID = api_commen.GetIDFromFilePath( loadImgPath );

		/************************ResizeImg*****************************/
		IplImage* imgResize = api_mainboby.ResizeImg( img );

		/************************Get_Hypothese*****************************/	
		Res.clear();
		run.start();
		nRet = api_mainboby.Predict( imgResize, ImageID, layerName, Res );
		if (nRet!=0)
		{
			LOOGE<<"[Predict Err!!loadImgPath:]"<<loadImgPath;
			cvReleaseImage(&img);img = 0;
			cvReleaseImage(&imgResize);imgResize = 0;
			continue;
		}
		run.end();
		LOOGI<<"[Predict] time:"<<run.time();
		allPredictTime += run.time();

		/************************save one img with muti label*****************************/
/*		Mat matImg(imgResize);
		sprintf( tPath, "res_predict/%s/", Res[0].first.first.c_str() );
		for(i=0;i<Res.size();i++)  
		{				
			Scalar color = colors[i%8];
			rectangle( matImg, cvPoint(Res[i].first.second[0], Res[i].first.second[1]),
	                   cvPoint(Res[i].first.second[2], Res[i].first.second[3]), color, 2, 8, 0);

			sprintf(szImgPath, "%.2f %s", Res[i].second, Res[i].first.first.c_str() );
			text = szImgPath;
			//putText(matImg, text, cvPoint(Res[i].first.second[0]+1, Res[i].first.second[1]+20), 
			//	FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
			putText(matImg, text, cvPoint(1, i*20+20), FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
				
			sprintf(tPath, "%s%.2f-%s_", tPath, Res[i].second, Res[i].first.first.c_str() );
		}
		sprintf( tPath, "%s%lld.jpg", tPath, ImageID );
		imwrite( tPath, matImg );*/

		/************************save muti img with one label*****************************/
		for(i=0;i<Res.size();i++)  
		{				
			IplImage* imgReWrite = cvCloneImage( imgResize );
			Mat matImg(imgReWrite);
			Scalar color = colors[i%8];
			rectangle( matImg, cvPoint(Res[i].first.second[0], Res[i].first.second[1]),
	                   cvPoint(Res[i].first.second[2], Res[i].first.second[3]), color, 2, 8, 0);

			sprintf(szImgPath, "%.2f %s", Res[i].second, Res[i].first.first.c_str() );
			text = szImgPath;
			//putText(matImg, text, cvPoint(Res[i].first.second[0]+1, Res[i].first.second[1]+20), 
			//	FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
			putText(matImg, text, cvPoint(1, i*20+20), FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
				
			sprintf(tPath, "res_predict/%s/%.2f-%s_%lld.jpg", 
				Res[i].first.first.c_str(),Res[i].second, Res[i].first.first.c_str(),ImageID );
			imwrite( tPath, matImg );

			cvReleaseImage(&imgReWrite);imgReWrite = 0;
		}

		//check data 
		for(i=0;i<Res.size();i++)  
		{	
			text = Res[i].first.first;
			it_Num_Res_Label = map_Num_Res_Label.find(text);
			if(it_Num_Res_Label == map_Num_Res_Label.end())
			{
			    map_Num_Res_Label[text] = 1;
			}
			else
			{	
				map_Num_Res_Label[text] = it_Num_Res_Label->second+1;
			}
		}

		cvReleaseImage(&imgResize);imgResize = 0;
		cvReleaseImage(&img);img = 0;
	}

	/*********************************close file*************************************/
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	

	/*********************************Release*************************************/
	api_mainboby.Release();

	/*********************************Print Info*********************************/
	if ( nCount != 0 ) 
	{
		printf( "nCount:%ld,PredictTime:%.4fms\n", nCount, allPredictTime*1000.0/nCount );

		//print check data
		long Num_Check[3] = {0};
		printf("Label_Num:%d\n",map_Num_Res_Label.size());
		for(it_Num_Res_Label = map_Num_Res_Label.begin(); it_Num_Res_Label != map_Num_Res_Label.end(); it_Num_Res_Label++)
		{
			text = it_Num_Res_Label->first.c_str();
			printf("%s_%ld\n", text.c_str(), it_Num_Res_Label->second );

			if (text == "other.other.other")
				Num_Check[0] += it_Num_Res_Label->second;
			else if ( (text == "food.food.food") || (text == "goods.goods.goods") ||
				 (text == "people.people.people") ||(text == "pet.pet.pet") || 
				 (text == "scene.scene.scene") )
				Num_Check[1] += it_Num_Res_Label->second;
			else
				Num_Check[2] += it_Num_Res_Label->second;
		}
		printf("\n");

		allLabel = Num_Check[0]+Num_Check[1]+Num_Check[2];
		printf("AllLabel:%ld,2nd-class-label:%ld_%.2f,1rd-class-label:%ld_%.2f,other:%ld_%.2f\n",
			allLabel, 	Num_Check[2], Num_Check[2]*100.0/(Num_Check[1]+Num_Check[2]), 
						Num_Check[1], Num_Check[1]*100.0/(Num_Check[1]+Num_Check[2]), 
						Num_Check[0], Num_Check[0]*100.0/allLabel );
		
	}
	
	cout<<"Done!! "<<endl;
	
	return nRet;
}	

int main(int argc, char* argv[])
{
	int  ret = 0;
	char szKeyFiles[256],szSavePath[256];
	API_COMMEN api_commen;

	if (argc == 7 && strcmp(argv[1],"-get_roi") == 0) {
		ret = Get_Bing_ROI( argv[2], argv[3], argv[4], atol(argv[5]), atol(argv[6]) );
	}
	else if (argc == 7 && strcmp(argv[1],"-predict_roi") == 0) {
		ret = Predict_Hypothese( argv[2], argv[3], argv[4], atol(argv[5]), atol(argv[6]) );
	}
	else if (argc == 5 && strcmp(argv[1],"-get_xml_roi") == 0) {
		ret = Get_Xml_ROI( argv[2], argv[3], argv[4] );
	}
	else if (argc == 9 && strcmp(argv[1],"-get_xml_bing_roi_for_1_0_0") == 0) {
		ret = Get_Xml_Bing_ROI_for_1_0_0( argv[2], argv[3], argv[4], argv[5], argv[6], atol(argv[7]), atol(argv[8]) );
	}
	else if (argc == 5 && strcmp(argv[1],"-get_fineturn_sample") == 0) {
		ret = Get_fineturn_sample( argv[2], argv[3], argv[4] );
	}
	else if (argc == 8 && strcmp(argv[1],"-extract") == 0) {
		strcpy(szKeyFiles, argv[4]);
		api_commen.PadEnd(szKeyFiles);
		ret = DL_ExtractFeat( argv[2], argv[3], szKeyFiles, argv[5], atol(argv[6]), atol(argv[7]) );
	}
	else if (argc == 7 && strcmp(argv[1],"-predict") == 0) {
		ret = SVM_Predict( argv[2], argv[3], argv[4], atol(argv[5]), atol(argv[6]) );
	}
	else
	{
		cout << "usage:\n" << endl;
		cout << "\tDemo_mainboby -get_roi queryList.txt keyFilePath layerName binGPU deviceID\n" << endl;
		cout << "\tDemo_mainboby -predict_roi queryList.txt keyFilePath layerName binGPU deviceID\n" << endl;
		cout << "\tDemo_mainboby -get_xml_roi queryList.txt loadXmlPath svImagePath\n" << endl;
		cout << "\tDemo_mainboby -get_xml_bing_roi_for_1_0_0 queryList.txt loadXmlPath svImagePath keyFilePath layerName binGPU deviceID\n" << endl;
		cout << "\tDemo_mainboby -get_fineturn_sample queryList.txt svImagePath label\n" << endl;
		cout << "\tDemo_mainboby -extract queryList.txt szFeat keyFilePath layerName binGPU deviceID\n" << endl;
		cout << "\tDemo_mainboby -predict queryList.txt keyFilePath layerName binGPU deviceID\n" << endl;
		return ret;
	}
	return ret;
}
