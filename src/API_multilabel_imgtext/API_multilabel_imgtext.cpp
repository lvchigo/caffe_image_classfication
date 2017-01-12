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

#include "API_commen.h"
#include "API_multilabel_imgtext.h"
#include "TErrorCode.h"
#include "plog/Log.h"


using namespace cv;
using namespace std;

static bool SortComp(const pair <int, float> elem1, const pair <int, float> elem2)
{
	return (elem1.second > elem2.second);
}


/***********************************Init*************************************/
/// construct function 
API_MUTILABEL_IMGTEXT::API_MUTILABEL_IMGTEXT()
{
}

/// destruct function 
API_MUTILABEL_IMGTEXT::~API_MUTILABEL_IMGTEXT(void)
{
}

/***********************************Init*************************************/
int API_MUTILABEL_IMGTEXT::Init( 
	const char* 	KeyFilePath,						//[In]:KeyFilePath
	const char* 	layerName,							//[In]:layerName:"fc7"
	const int		binGPU, 							//[In]:USE GPU(1) or not(0)
	const int		deviceID )	 						//[In]:GPU ID
{
	int i,nRet = 0;
	char tPath[1024] = {0};
	char tPath2[1024] = {0};
	char tPath3[1024] = {0};
	
	/***********************************Init api_caffe*************************************/
	sprintf(tPath, "%s/mutilabel_imgtext/deploy_vgg16.prototxt",KeyFilePath);
	sprintf(tPath2, "%s/mutilabel_imgtext/vgg16_iter4_42500_238class.caffemodel",KeyFilePath);
	sprintf(tPath3, "%s/mutilabel_imgtext/imagenet_mean.binaryproto",KeyFilePath);
	nRet = api_caffe.Init( tPath, tPath2, tPath3, layerName, binGPU, deviceID ); 
	if (nRet != 0)
	{
	   LOOGE<<"Fail to initialization ";
	   return TEC_INVALID_PARAM;
	}

	/***********************************Load dic_20Class File**********************************/
	dic_20Class.clear();
	sprintf(tPath, "%s/mutilabel_imgtext/dict_238.txt",KeyFilePath);
	printf("load dic_20Class:%s\n",tPath);
	api_commen.loadWordDict(tPath,dic_20Class);
	printf( "dict:size-%d,tag:", int(dic_20Class.size()) );
	for ( i=0;i<dic_20Class.size();i++ )
	{
		printf( "%d-%s ",i,dic_20Class[i].c_str() );
	}
	printf( "\n" );

	return nRet;
}

/***********************************Extract Feat**********************************/
int API_MUTILABEL_IMGTEXT::ExtractFeat( 
	IplImage* 						image, 								//[In]:image
	UInt64							ImageID,							//[In]:ImageID
	const char* 					layerName,							//[In]:Layer Name by Extract
	vector< vector< float > > 		&vecFeat)				
{
	if(!image || (image->width<16) || (image->height<16) || image->nChannels != 3 || image->depth != IPL_DEPTH_8U) 
	{	
		LOOGE<<"image err!!";
		return TEC_INVALID_PARAM;
	}	

	/*****************************Init*****************************/
	int i,j,nRet = 0;
	double tPredictTime = 0;
	RunTimer<double> run;
	
	vector< pair< int, float > > imgLabel;
	vector< vector<float> > imgFeat;
	vector<float> EncodeFeat;
	vector<float> normFeat;
	
	/*****************************GetMutiImg*****************************/
	vector < Mat_ < Vec3f > > img_dl;
	nRet = api_commen.Img_GetMutiRoi( image, ImageID, img_dl );	
	if ( ( nRet != 0) || ( img_dl.size() < 1 ) )
	{
		LOOGE<<"Fail to Img_GetMutiRoi!!";
		return TEC_BAD_STATE;
	}
	
	/*****************************GetLabelFeat*****************************/	
	imgLabel.clear();
	imgFeat.clear();	
	int bExtractFeat = 2;		//[In]:Get Label(1),Extract Feat(2),both(3)
	//tPredictTime = (double)getTickCount();
	run.start();
	nRet = api_caffe.GetLabelFeat( img_dl, ImageID, layerName, bExtractFeat, imgLabel, imgFeat);
	if ( (nRet != 0) || (imgFeat.size()<1) )
	{
		LOOGE<<"Fail to GetFeat!!";
		return TEC_BAD_STATE;
	}
	run.end();
	LOOGI<<"[ExtractFeat--GetLabelFeat] time:"<<run.time();
	//tPredictTime = (double)getTickCount() - tPredictTime;
	//tPredictTime = tPredictTime*1000./cv::getTickFrequency();
	//printf( "[Time api_caffe.GetLabelFeat]:%.4fms\n", tPredictTime );

	/************************Normal && PCA Feat*****************************/
	vecFeat.clear();
	for(i=0;i<imgFeat.size();i++)
	{
		normFeat.clear();
		EncodeFeat.clear();	

		nRet = api_commen.Normal_MinMax(imgFeat[i],normFeat);
		if (nRet != 0)
		{
		   LOOGE<<"Fail to Normal_MinMax!!";
		   return nRet;
		}

		vecFeat.push_back(normFeat);	//linearSVM, FOR in73Class
	}

	return 0;
}


/***********************************Predict**********************************/
int API_MUTILABEL_IMGTEXT::Predict(
	IplImage						*image, 			//[In]:image
	const char* 					layerName,			//[In]:Layer Name by Extract
	vector< pair< string, float > >	&ResProb,
	vector< pair< string, float > >	&ResScore,
	int								&label_add)			//[Out]:Res:In37Class/ads6Class/imgquality3class
{
	if(!image || (image->width<16) || (image->height<16) || image->nChannels != 3 || image->depth != IPL_DEPTH_8U) 
	{	
		LOOGE<<"image err!!";
		return TEC_INVALID_PARAM;
	}	

	/*****************************Init*****************************/
	int i,j,nRet = 0;
	int n_score_layer, top_score_layer = 5;
	double tPredictTime = 0;
	RunTimer<double> run;
	UInt64 ImageID = 0;
	
	vector< pair< int, float > > imgLabel;
	vector< vector<float> > imgFeat;
	vector< pair< int,float > > 		intLabelInfo;
	vector< pair< int, float > >		tmpResScore;
	
	/*****************************GetMutiImg*****************************/
	vector < Mat_ < Vec3f > > img_dl;
	nRet = api_commen.Img_GetMutiRoi( image, ImageID, img_dl );	
	if ( ( nRet != 0) || ( img_dl.size() < 1 ) )
	{
		LOOGE<<"Fail to Img_GetMutiRoi!!";
		return TEC_BAD_STATE;
	}
	
	/*****************************GetLabelFeat*****************************/	
	imgLabel.clear();
	imgFeat.clear();	
	int bExtractFeat = 2;		//[In]:Get Label(1),Extract Feat(2),both(3)
	//tPredictTime = (double)getTickCount();
	run.start();
	nRet = api_caffe.GetLabelFeat( img_dl, ImageID, layerName, bExtractFeat, imgLabel, imgFeat);
	if ( (nRet != 0) || (imgLabel.size()<1) || (imgFeat.size()<1) )
	{
		LOOGE<<"Fail to GetFeat!!";
		return TEC_BAD_STATE;
	}
	run.end();
	LOOGI<<"[ExtractFeat--GetLabelFeat] time:"<<run.time();
	//tPredictTime = (double)getTickCount() - tPredictTime;
	//tPredictTime = tPredictTime*1000./cv::getTickFrequency();
	//printf( "[Time api_caffe.GetLabelFeat]:%.4fms\n", tPredictTime );

	//score layer>0
	tmpResScore.clear();
	for (i = 0; i < imgFeat.size(); ++i) {
	  	for (j = 0; j < imgFeat[i].size(); ++j) {
			if (imgFeat[i][j]>0)
				tmpResScore.push_back( std::make_pair( j, imgFeat[i][j] ) );
		}
	}

	//sort label result
	sort(tmpResScore.begin(), tmpResScore.end(),SortComp);	

	//top-N
	ResScore.clear();
	n_score_layer = 0;
	label_add = 0;
	for (i = 0; i < tmpResScore.size(); ++i) {
		if (n_score_layer<top_score_layer)
		{
			ResScore.push_back( std::make_pair( dic_20Class[tmpResScore[i].first], tmpResScore[i].second ) );
			n_score_layer++;

			if (tmpResScore[i].first>64)
				label_add = 1;
		}
	}
	
	/************************Merge37classLabel*****************************/	
	ResProb.clear();
	intLabelInfo.clear();
	nRet = Merge20classLabel( imgLabel, ResProb, intLabelInfo );
	if (nRet != 0)
	{	
		LOOGE<<"Fail to Merge Label!!";
	   	return nRet;
	}

	return nRet;
}

int API_MUTILABEL_IMGTEXT::Merge20classLabel(
	vector< pair< int, float > >		inImgLabel, 		//[In]:ImgDetail from GetLabel
	vector< pair< string, float > > 	&LabelInfo,			//[Out]:LabelInfo
	vector< pair< int,float > > 		&intLabelInfo )		//[Out]:intLabelInfo
{
	if ( inImgLabel.size() < 1 ) 
	{ 
		LOOGE<<"MergeLabel[err]:inImgLabel.size()<1!!";
		return TEC_INVALID_PARAM;
	}
	
	int i,label,nRet = 0;
	float score = 0.0;
	LabelInfo.clear();
	intLabelInfo.clear();

	for ( i=0;i<inImgLabel.size();i++ )
	{
		label = inImgLabel[i].first;
		score = inImgLabel[i].second;		

		if (label < dic_20Class.size() )
		{
			LabelInfo.push_back( std::make_pair( dic_20Class[label], score ) );
			intLabelInfo.push_back( std::make_pair( label, score ) );
		}
		else
		{ 
			LOOGE<<"Merge Label[err]!!";
			return TEC_INVALID_PARAM;
		}
	}
	
	return nRet;
}

/***********************************Release**********************************/
void API_MUTILABEL_IMGTEXT::Release()
{
	
	/***********************************net Model**********************************/
	api_caffe.Release();

	/***********************************dict Model**********************************/
	dic_20Class.clear();

}

