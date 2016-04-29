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
#include "API_caffe.h"
#include "API_linearsvm.h"
#include "API_mutilabel.h"
#include "TErrorCode.h"
#include "plog/Log.h"


using namespace cv;
using namespace std;

/***********************************Init*************************************/
/// construct function 
API_MUTILABEL::API_MUTILABEL()
{
}

/// destruct function 
API_MUTILABEL::~API_MUTILABEL(void)
{
}

/***********************************Init*************************************/
int API_MUTILABEL::Init( 
	const char* 	KeyFilePath,						//[In]:KeyFilePath
	const char* 	layerName,							//[In]:layerName:"fc7"
	const int		binGPU, 							//[In]:USE GPU(1) or not(0)
	const int		deviceID )	 						//[In]:GPU ID
{
	int i,nRet = 0;
	char tPath[1024] = {0};
	char tPath2[1024] = {0};
	char tPath3[1024] = {0};

	/***********************************Init**********************************/
	sprintf(tPath, "%s/log/plog.txt",KeyFilePath);	//logout
	plog::init(plog::error, tPath); 
	
	/***********************************Init api_caffe*************************************/
	sprintf(tPath, "%s/vgg_16/deploy_vgg_16.prototxt",KeyFilePath);
	sprintf(tPath2, "%s/vgg_16/VGG_ILSVRC_16_layers.caffemodel",KeyFilePath);
	sprintf(tPath3, "%s/vgg_16/imagenet_mean.binaryproto",KeyFilePath);
	nRet = api_caffe.Init( tPath, tPath2, tPath3, layerName, binGPU, deviceID ); 
	if (nRet != 0)
	{
	   LOOGE<<"Fail to initialization ";
	   return TEC_INVALID_PARAM;
	}

	/***********************************Init Model_SVM_in73class*************************************/
	sprintf(tPath, "%s/in73class/linearsvm_in73class_050702.model",KeyFilePath);	//linearSVM
	printf("load Model_SVM_in73class:%s\n",tPath);
	nRet = api_libsvm_73class.Init(tPath); 
	if (nRet != 0)
	{
	   LOOGE<<"Fail to initialization ";
	   return TEC_INVALID_PARAM;
	}

	/***********************************Init Model_SVM_in6class*************************************/
	sprintf(tPath, "%s/in6class/linearsvm_in6class_050716.model", KeyFilePath);		//linearSVM
	printf("load Model_SVM_in6class:%s\n",tPath);
	nRet = api_libsvm_in6Class.Init(tPath); 
	if (nRet != 0)
	{
	   LOOGE<<"Fail to initialization ";
	   return TEC_INVALID_PARAM;
	}

	/***********************************Init Model_SVM_in95class*************************************/
	sprintf(tPath, "%s/in95class/linearsvm_in95class_151111.model",KeyFilePath);	//linearSVM
	printf("load Model_SVM_in95class:%s\n",tPath);
	nRet = api_libsvm_95class.Init(tPath); 
	if (nRet != 0)
	{
	   LOOGE<<"Fail to initialization ";
	   return TEC_INVALID_PARAM;
	}

	/***********************************Load dic_73Class File**********************************/
	dic_73Class.clear();
	sprintf(tPath, "%s/in73class/dict",KeyFilePath);
	printf("load dic_73Class:%s\n",tPath);
	api_commen.loadWordDict(tPath,dic_73Class);
	printf( "dict:size-%d,tag:", int(dic_73Class.size()) );
	for ( i=0;i<dic_73Class.size();i++ )
	{
		printf( "%d-%s ",i,dic_73Class[i].c_str() );
	}
	printf( "\n" );

	/***********************************Load dic_in37class File**********************************/
	dic_in37class.clear();
	sprintf(tPath, "%s/in73class/dict_online",KeyFilePath);
	printf("load dic_in37class:%s\n",tPath);
	api_commen.loadWordDict(tPath,dic_in37class);
	printf( "dict:size-%d,tag:", int(dic_in37class.size()) );
	for ( i=0;i<dic_in37class.size();i++ )
	{
		printf( "%d-%s ",i,dic_in37class[i].c_str() );
	}
	printf( "\n" );

	/***********************************Load dic_95Class File**********************************/
	dic_95Class.clear();
	sprintf(tPath, "%s/in95class/dict",KeyFilePath);
	printf("load dic_95Class:%s\n",tPath);
	api_commen.loadWordDict(tPath,dic_95Class);
	printf( "dict:size-%d,tag:", int(dic_95Class.size()) );
	for ( i=0;i<dic_95Class.size();i++ )
	{
		printf( "%d-%s ",i,dic_95Class[i].c_str() );
	}
	printf( "\n" );

	return nRet;
}

/***********************************Extract Feat**********************************/
int API_MUTILABEL::ExtractFeat( 
	IplImage* 						image, 								//[In]:image
	UInt64							ImageID,							//[In]:ImageID
	const char* 					layerName,							//[In]:Layer Name by Extract
	vector< vector< float > > 		&vecIn73ClassFeat,					//for in73class
	vector< vector< float > > 		&vecIn6ClassFeat)					//for in6class
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
	vector<float> normIn73ClassFeat;
	vector<float> normIn6ClassFeat;
	
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
	run.start();
	nRet = api_caffe.GetLabelFeat( img_dl, ImageID, layerName, bExtractFeat, imgLabel, imgFeat);
	if ( (nRet != 0) || (imgFeat.size()<1) )
	{
		LOOGE<<"Fail to GetFeat!!";
		return TEC_BAD_STATE;
	}
	run.end();
	LOOGI<<"[ExtractFeat--GetLabelFeat] time:"<<run.time();

	/************************Normal && PCA Feat*****************************/
	vecIn73ClassFeat.clear();
	vecIn6ClassFeat.clear();
	for(i=0;i<imgFeat.size();i++)
	{
		if ( i == 0 )
		{
			normIn73ClassFeat.clear();
			normIn6ClassFeat.clear();
			EncodeFeat.clear();	

			/************************DL Ads6ClassFeat Normal*****************************/
			nRet = api_commen.Normal_L2(imgFeat[i],normIn73ClassFeat);
			if (nRet != 0)
			{
			   LOOGE<<"Fail to GetFeat!!";
			   return nRet;
			}

			/************************DL In6ClassFeat Normal*****************************/
			nRet = api_commen.Normal_MinMax(imgFeat[i],normIn6ClassFeat);	//NormDLFeat FOR in5class
			if (nRet != 0)
			{
			   LOOGE<<"Fail to Normal_MinMax!!";
			   return nRet;
			}

			vecIn73ClassFeat.push_back(normIn73ClassFeat);	//linearSVM, FOR in73Class
			vecIn6ClassFeat.push_back(normIn6ClassFeat);	// FOR in6class
		}
	}

	return 0;
}


/***********************************Predict**********************************/
int API_MUTILABEL::Predict(
	IplImage						*image, 			//[In]:image
	UInt64							ImageID,			//[In]:ImageID
	const char* 					layerName,			//[In]:Layer Name by Extract
	vector< pair< string, float > >	&Res)				//[Out]:Res:In37Class/ads6Class/imgquality3class
{
	if(!image || (image->width<16) || (image->height<16) || image->nChannels != 3 || image->depth != IPL_DEPTH_8U) 
	{	
		LOOGE<<"image err!!";
		return TEC_INVALID_PARAM;
	}	

	/*****************************Init*****************************/
	int i,nRet = 0;
	RunTimer<double> run;
	
	vector< pair< int, float > > imgLabel;
	vector< vector<float> > imgFeat;
	vector< vector<float> > vecIn73ClassFeat;
	vector< vector<float> > vecIn6ClassFeat;
	vector < pair < int,float > > tmpRes;
	vector < pair < int,float > > tmpResIn37Class;
	vector< pair< string, float > >	ResIn37Class;

	/*****************************ExtractFeat*****************************/
	run.start();
	nRet = ExtractFeat( image, ImageID, layerName, vecIn73ClassFeat, vecIn6ClassFeat );
	if ( (nRet != 0) || (vecIn73ClassFeat.size()<1) || (vecIn6ClassFeat.size()<1) )
	{
		LOOGE<<"Fail to GetFeat!!";
		return TEC_BAD_STATE;
	}
	run.end();
	LOOGI<<"[Predict--ExtractFeat] time:"<<run.time();

	/************************Predict In37Class*****************************/
	{
		/************************SVM 73class Predict*****************************/	
		tmpRes.clear();
		nRet = api_libsvm_73class.Predict_mutilabel( vecIn73ClassFeat, tmpRes );
		//nRet = api_libsvm_in6Class.Predict_mutilabel( vecIn6ClassFeat, tmpRes );
		//nRet = api_libsvm_95class.Predict_mutilabel( vecIn73ClassFeat, tmpRes );
		if ( (nRet != 0) || (tmpRes.size()<1) )
		{	
			LOOGE<<"Fail to Predict SVM!!ImageID:"<<ImageID;
		   	return nRet;
		}		

		/************************Merge37classLabel*****************************/	
		ResIn37Class.clear();
		tmpResIn37Class.clear();
		nRet = Merge73classLabel( tmpRes, ResIn37Class, tmpResIn37Class );
		//nRet = Merge37classLabel( tmpRes, ResIn37Class, tmpResIn37Class );
		//nRet = Merge95classLabel( tmpRes, ResIn37Class, tmpResIn37Class );
		if ( (nRet != 0) || (ResIn37Class.size()<1) )
		{	
			LOOGE<<"Fail to Merge37classLabel!!ImageID:"<<ImageID;
		   	return nRet;
		}
	}

	/************************Merge Res*****************************/
	Res.clear();
	if ( ResIn37Class.size() > 0 )
	{
		for (i=0;i<ResIn37Class.size();i++)
			Res.push_back( make_pair( ResIn37Class[i].first, ResIn37Class[i].second ) );		
	}
	else
		Res.push_back( make_pair( "null", 0 ) );

	return nRet;
}

int API_MUTILABEL::Merge37classLabel(
	vector< pair< int, float > >		inImgLabel, 		//[In]:ImgDetail from GetLabel
	vector< pair< string, float > > 	&LabelInfo,			//[Out]:LabelInfo
	vector< pair< int,float > > 		&intLabelInfo )		//[Out]:intLabelInfo
{
	if ( inImgLabel.size() < 1 ) 
	{
		LOOGE<<"MergeLabel[err]:inImgLabel.size()<1!!";
		return 1;
	}
	
	int i,index,tmpLabel,topN,nRet = 0;
	float score = 0.0;
	const int onlineLabel[73] = {	1, 8, 2, 0, 8, 3, 8, 8, 8, 4,
			       					0, 5, 6, 8, 8, 8, 8, 7, 19,9,
			       					15,19,15,0, 19,10,11,0, 12,13,
			       					0, 0, 0, 11,16,15,0, 0, 15,19,
			       					18,0, 17,25,20,14,21,25,22,23,
			        				24,28,26,27,28,0, 33,33,33,29,
			        				0, 0, 33,33,33,30,31,34,32,33,
			        				33,35,36 };
	LabelInfo.clear();
	intLabelInfo.clear();

	topN = 0;
	for ( i=0;i<inImgLabel.size();i++ )
	{
		tmpLabel = 0;
		index = inImgLabel[i].first;
		score = inImgLabel[i].second;
		//printf( "MergeLabel:i-%d,index-%d,score-%.4f\n",i,index, score );

		tmpLabel = onlineLabel[index];
		if  ( (i == 0) && (													//data need adjust ****
			( ( index>=0  ) && ( index<5  ) && (score>=0.7) ) || 	
			( ( index==5  ) 				&& (score>=0.7) ) || 	//cook
			( ( index>=6  ) && ( index<9  ) && (score>=0.7) ) ||	//food.*
			( ( index==9  ) 				&& (score>=0.6) ) || 	//fruit
			( ( index>=10 ) && ( index<28 ) && (score>=0.7) ) ||	//good.*
			( ( index==28 ) 				&& (score>=0.5) ) || 	//flower
			( ( index>=29 ) && ( index<40 ) && (score>=0.7) ) ||	//good.*
			( ( index==40 )					&& (score>=0.8) ) ||	//goods.shoe.shoe
			( ( index>=41 ) && ( index<43 ) && (score>=0.7) ) ||	//good.*
			( ( index>=43 ) && ( index<48 ) && (score>=0.6) ) ||	//people.*
			( ( index>=48 ) && ( index<51 ) && (score> 0.6) ) ||	//people.self.* && people.street
			( ( index>=51 ) && ( index<71 ) && (score>=0.7) ) || 
			( ( index==71 ) 				&& (score>=0.6) ) || 	//text
			( ( index>=72 ) && ( index<73 ) && (score>=0.7) )
			 ) )
		{
			topN = 1;
		}
		LabelInfo.push_back( std::make_pair( dic_in37class[tmpLabel], score ) );
		intLabelInfo.push_back( std::make_pair( tmpLabel, score ) );
		//printf("index:%d,label:%s,score:%.4f\n",tmpLabel,dic_in37class[tmpLabel].c_str(),score);

//		if (topN == 1)
//			break;
	}
	
	return nRet;
}

int API_MUTILABEL::Merge73classLabel(
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

		if (label < dic_73Class.size() )
		{
			LabelInfo.push_back( std::make_pair( dic_73Class[label], score ) );
			intLabelInfo.push_back( std::make_pair( label, score ) );
		}
		else
		{ 
			LOOGE<<"Merge72classLabel[err]!!label:"<<label;
			return TEC_INVALID_PARAM;
		}
	}
	
	return nRet;
}

int API_MUTILABEL::MergeIn6ClassLabel(
	vector< pair< int, float > > 		inImgLabel, 		//[In]:inImgLabel
	vector< pair< string, float > > 	&LabelInfo,			//[Out]:outImgLabel
	vector< pair< int,float > > 		&intLabelInfo )		//[Out]:intLabelInfo
{
	if ( inImgLabel.size() < 1 ) 
	{
		LOOGE<<"MergeLabel[err]:inImgLabel.size()<1!!";
		return TEC_INVALID_PARAM;
	}

	int i,index,tmpLabel,nRet = 0;
	float score = 0.0;
	const int onlineLabel[6] = { 8, 0, 25, 28, 19, 33 };
	LabelInfo.clear();
	intLabelInfo.clear();
	
	for ( i=0;i<inImgLabel.size();i++ )
	{
		tmpLabel = 0;
		index = inImgLabel[i].first;
		score = inImgLabel[i].second;
		//printf( "MergeLabel:i-%d,index-%d,score-%.4f\n",i,index, score );
		if ( 0 == i )
		{
			if (
				( ( index>=0  ) && ( index<2  ) && (score>=0.8) ) || 	//food other
				( ( index==2  ) 				&& (score>=0.6) ) || 	//people
				( ( index>=3  ) && ( index<6  ) && (score>=0.8) ) 		//pet puppet scene
				 )
			{
				tmpLabel = onlineLabel[index];
			}
			else
			{
				tmpLabel = 0;
			}
			LabelInfo.push_back( std::make_pair( dic_in37class[tmpLabel], score ) );
			intLabelInfo.push_back( std::make_pair( tmpLabel, score ) );
			//printf("index:%d,label:%s,score:%.4f\n",tmpLabel,dic_in37class[tmpLabel].c_str(),score);
		}
	}
	
	return nRet;
}

int API_MUTILABEL::Merge95classLabel(
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

		if (label < dic_95Class.size() )
		{
			LabelInfo.push_back( std::make_pair( dic_95Class[label], score ) );
			intLabelInfo.push_back( std::make_pair( label, score ) );
		}
		else
		{ 
			LOOGE<<"Merge95classLabel[err]!!label:"<<label;
			return TEC_INVALID_PARAM;
		}
	}
	
	return nRet;
}

/***********************************Release**********************************/
void API_MUTILABEL::Release()
{
	/***********************************net Model**********************************/
	api_caffe.Release();

	/***********************************dict Model**********************************/
	dic_73Class.clear();
	dic_in37class.clear();
	dic_95Class.clear();

	/***********************************SVM Model******libsvm3.2.0********************/
	api_libsvm_73class.Release();
	api_libsvm_in6Class.Release();
	api_libsvm_95class.Release();

}





