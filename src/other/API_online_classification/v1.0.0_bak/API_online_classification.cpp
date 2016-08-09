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

#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <ml.h>
#include <cvaux.h>

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
#include "API_pca.h"
#include "API_libsvm.h"
#include "API_linearsvm.h"
#include "similardetect/SD_global.h"
#include "API_online_classification.h"
#include "TErrorCode.h"

using namespace cv;
using namespace std;

/***********************************Init*************************************/
/// construct function 
API_ONLINE_CLASSIFICATION::API_ONLINE_CLASSIFICATION()
{
}

/// destruct function 
API_ONLINE_CLASSIFICATION::~API_ONLINE_CLASSIFICATION(void)
{
}

/***********************************Init*************************************/
int API_ONLINE_CLASSIFICATION::Init( 
	const char* 	KeyFilePath,						//[In]:KeyFilePath
	const char* 	layerName,							//[In]:layerName:"fc7"
	const int		binGPU, 							//[In]:USE GPU(1) or not(0)
	const int		deviceID )	 						//[In]:GPU ID
{
	int i,nRet = 0;
	char tPath[1024] = {0};
	char tPath2[1024] = {0};
	char tPath3[1024] = {0};

	/***********************************Load similardetect File**********************************/
	SD_GLOBAL_1_1_0::ClassInfo ci;
	vector < SD_GLOBAL_1_1_0::ClassInfo > classList;
	
	sprintf(tPath, "%s/SimilarDetect/keyfile",KeyFilePath);
	printf("load similardetect File:%s\n",tPath);
	nRet  = SD_GLOBAL_1_1_0::Init(tPath);
	if (nRet != 0)
	{
	   cout<<"similardetect init err :can't open"<< tPath<<endl;
	   return TEC_INVALID_PARAM;
	}
	
	ci.ClassID = 70020;
	ci.SubClassID = 70020;
	classList.push_back(ci);
	nRet = SD_GLOBAL_1_1_0::LoadClassData(classList);
	if (nRet != 0)
	{
	   cout<<"similardetect init err : Fail to LoadClassData. Error code:"<< nRet << endl;
	   return TEC_INVALID_PARAM;
	}
	
	/***********************************Init api_caffe*************************************/
	sprintf(tPath, "%s/vgg_16/deploy_vgg_16.prototxt",KeyFilePath);
	sprintf(tPath2, "%s/vgg_16/VGG_ILSVRC_16_layers.caffemodel",KeyFilePath);
	sprintf(tPath3, "%s/vgg_16/imagenet_mean.binaryproto",KeyFilePath);
	nRet = api_caffe.Init( tPath, tPath2, tPath3, layerName, binGPU, deviceID ); 
	if (nRet != 0)
	{
	   cout<<"Fail to initialization "<<endl;
	   return TEC_INVALID_PARAM;
	}

	/***********************************Init api_pca*************************************/
#ifndef USE_LINEARSVM
	sprintf(tPath, "%s/in73class/PCA500_L2_in73class_05052801.model",KeyFilePath);			//vgg:add 2dcode
	printf("load api_pca:%s\n",tPath);
	api_pca.PCA_Feat_Init(tPath); 
#endif

	/***********************************Init Model_SVM_in73class*************************************/
#ifdef USE_LINEARSVM
	sprintf(tPath, "%s/in73class/linearsvm_in73class_050702.model",KeyFilePath);	//linearSVM
#else
	sprintf(tPath, "%s/in73class/in73class_VGG16_L2Norm_PCA500Data_05052801_SVM.model",KeyFilePath);	//vgg:add 2dcode
#endif
	printf("load Model_SVM_in73class:%s\n",tPath);
	nRet = api_libsvm_73class.Init(tPath); 
	if (nRet != 0)
	{
	   cout<<"Fail to initialization "<<endl;
	   return TEC_INVALID_PARAM;
	}

	/***********************************Init Model_SVM_in6class*************************************/
#ifdef USE_LINEARSVM
	sprintf(tPath, "%s/in6class/linearsvm_in6class_050702.model", KeyFilePath);		//linearSVM
#else
	sprintf(tPath, "%s/in6class/in6class_AllData_051401_VGG16_SVM.model", KeyFilePath);
#endif
	printf("load Model_SVM_in6class:%s\n",tPath);
	nRet = api_libsvm_in6Class.Init(tPath); 
	if (nRet != 0)
	{
	   cout<<"Fail to initialization "<<endl;
	   return TEC_INVALID_PARAM;
	}

	/***********************************Init Model_SVM_ads6Class*************************************/
#ifdef USE_LINEARSVM
	sprintf(tPath, "%s/ads6class/linearsvm_ads6class_050702.model", KeyFilePath);	//linearSVM
#else
	sprintf(tPath, "%s/ads6class/ads6class_vgg_050625_svm.model", KeyFilePath);	//vgg:add 2dcode
#endif
	printf("load Model_SVM_ads6Class:%s\n",tPath);
	nRet = api_libsvm_ads6Class.Init(tPath); 
	if (nRet != 0)
	{
	   cout<<"Fail to initialization "<<endl;
	   return TEC_INVALID_PARAM;
	}
	
	/***********************************Init Model_imgquality3class*************************************/
#ifdef USE_LINEARSVM
	sprintf(tPath, "%s/imagequality/linearsvm_imagequality_food_scene_pet_street_050702.model",KeyFilePath);	//linearSVM
#else
	sprintf(tPath, "%s/imagequality/imagequality_food_scene_pet_street_05062302_SVM.model",KeyFilePath);	//libsvm-4feat
#endif
	printf("load Model_imgquality3class:%s\n",tPath);
	nRet = api_libsvm_imgquality3class.Init(tPath); 
	if (nRet != 0)
	{
	   cout<<"Fail to initialization "<<endl;
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

	/***********************************Load dic_ads6Class File**********************************/
	dic_ads6Class.clear();
	sprintf(tPath, "%s/ads6class/dict",KeyFilePath);
	printf("load dic_ads6Class:%s\n",tPath);
	api_commen.loadWordDict(tPath,dic_ads6Class);
	printf( "dict:size-%d,tag:", int(dic_ads6Class.size()) );
	for ( i=0;i<dic_ads6Class.size();i++ )
	{
		printf( "%d-%s ",i,dic_ads6Class[i].c_str() );
	}
	printf( "\n" );

	/***********************************Load dic_imgquality3class File**********************************/
	dic_imgquality3class.clear();
	sprintf(tPath, "%s/imagequality/dict_online",KeyFilePath);
	api_commen.loadWordDict(tPath,dic_imgquality3class);
	printf( "dict:size-%d,tag:", int(dic_imgquality3class.size()) );
	for ( i=0;i<dic_imgquality3class.size();i++ )
	{
		printf( "%d-%s ",i,dic_imgquality3class[i].c_str() );
	}
	printf( "\n" );

	return nRet;
}

/***********************************Extract Feat**********************************/
int API_ONLINE_CLASSIFICATION::ExtractFeat( 
	IplImage* 						image, 								//[In]:image
	UInt64							ImageID,							//[In]:ImageID
	const char* 					layerName,							//[In]:Layer Name by Extract
	vector< vector< float > > 		&vecIn73ClassFeat,					//for in73class
	vector< vector< float > > 		&vecIn6ClassFeat,					//for in6class
	vector< vector< float > > 		&vecAds6ClassFeat,					//for ads6class
	vector< vector< float > > 		&vecImageQuality3ClassFeat )		//for imagequality
{
	if(!image || (image->width<16) || (image->height<16) || image->nChannels != 3 || image->depth != IPL_DEPTH_8U) 
	{	
		cout<<"image err!!" << endl;
		return TEC_INVALID_PARAM;
	}	

	/*****************************Init*****************************/
	int i,nRet = 0;
	vector< pair< int, float > > imgLabel;
	vector< vector<float> > imgFeat;
	vector<float> EncodeFeat;
	vector<float> normIn73ClassFeat;
	vector<float> normIn6ClassFeat;
	vector<float> normAds6ClassFeat;
	vector<float> normImageQuality3ClassFeat;

	/*****************************GetMutiImg*****************************/
	vector < Mat_ < Vec3f > > img_dl;
	api_commen.Img_GetMutiRoi( image, ImageID, img_dl );	
	if ( img_dl.size() < 1 )
	{
		cout<<"Fail to Img_GetMutiRoi!! "<<endl; 
		return TEC_BAD_STATE;
	}
	
	/*****************************GetLabelFeat*****************************/	
	imgLabel.clear();
	imgFeat.clear();	
	int bExtractFeat = 3;		//[In]:Get Label(1),Extract Feat(2),both(3)
	nRet = api_caffe.GetLabelFeat( img_dl, ImageID, layerName, bExtractFeat, imgLabel, imgFeat);
	if ( (nRet != 0) || (imgFeat.size()<1) )
	{
		cout<<"Fail to GetFeat!! "<<endl; 
		return TEC_BAD_STATE;
	}

	/************************Normal && PCA Feat*****************************/
	vecIn73ClassFeat.clear();
	vecAds6ClassFeat.clear();
	vecIn6ClassFeat.clear();
	for(i=0;i<imgFeat.size();i++)
	{
		if ( i == 0 )
		{
			normAds6ClassFeat.clear();
			normIn6ClassFeat.clear();
			EncodeFeat.clear();
			normIn73ClassFeat.clear();

			/************************DL Ads6ClassFeat Normal*****************************/
			nRet = api_commen.Normal_L2(imgFeat[i],normAds6ClassFeat);
			if (nRet != 0)
			{
			   cout<<"Fail to Normal_L2!! "<<endl; 
			   return nRet;
			}

			/************************DL In6ClassFeat Normal*****************************/
			nRet = api_commen.Normal_MinMax(imgFeat[i],normIn6ClassFeat);	//NormDLFeat FOR in5class
			if (nRet != 0)
			{
			   cout<<"Fail to Normal_MinMax!! "<<endl;
			   return nRet;
			}

#ifndef USE_LINEARSVM
			/************************PCA_Feat_Encode*****************************/
			nRet = api_pca.PCA_Feat_Encode(normAds6ClassFeat,EncodeFeat);
			if (nRet != 0)
			{
			   cout<<"Fail to PCA_Feat_Encode!! "<<endl; 
			   return nRet;
			}

			/************************PCA Feat Normal*****************************/
			nRet = api_commen.Normal_L2(EncodeFeat,normIn73ClassFeat);
			if (nRet != 0)
			{
			   cout<<"Fail to Normal_L2!! "<<endl; 
			   return nRet;
			}
#endif

#ifdef USE_LINEARSVM
			vecIn73ClassFeat.push_back(normAds6ClassFeat);	//linearSVM, FOR in73Class
#else
			vecIn73ClassFeat.push_back(normIn73ClassFeat);	// FOR in73Class
#endif
			vecIn6ClassFeat.push_back(normIn6ClassFeat);	// FOR in6class
			vecAds6ClassFeat.push_back(normAds6ClassFeat);	// FOR ads6class	
		}
		else if ( i == 1 )
		{
			/************************DL Feat Normal*****************************/
			nRet = api_commen.Normal_MinMax(imgFeat[i],normImageQuality3ClassFeat);
			if (nRet != 0)
			{
			   cout<<"Fail to Normal_MinMax!! "<<endl;
			   return nRet;
			}

			vecImageQuality3ClassFeat.push_back(normImageQuality3ClassFeat);	// FOR in73Class
		}
	}

	return 0;
}


/***********************************Predict**********************************/
int API_ONLINE_CLASSIFICATION::Predict(
	IplImage						*image, 			//[In]:image
	UInt64							ImageID,			//[In]:ImageID
	const char* 					layerName,			//[In]:Layer Name by Extract
	vector< pair< string, float > >	&Res)				//[Out]:Res:In37Class/ads6Class/imgquality3class
{
	if(!image || (image->width<16) || (image->height<16) || image->nChannels != 3 || image->depth != IPL_DEPTH_8U) 
	{	
		cout<<"image err!!" << endl;
		return TEC_INVALID_PARAM;
	}	

	/*****************************Init*****************************/
	int i,nRet = 0;
	vector< pair< int, float > > imgLabel;
	vector< vector<float> > imgFeat;
	vector< vector<float> > vecIn73ClassFeat;
	vector< vector<float> > vecIn6ClassFeat;
	vector< vector<float> > vecAds6ClassFeat;
	vector< vector<float> > vecImageQuality3ClassFeat;
	vector < pair < int,float > > tmpRes;
	vector < pair < int,float > > tmpResIn37Class;
	vector< pair< string, float > >	ResIn37Class;
	vector< pair< string, float > >	ResAds6Class;
	vector< pair< string, float > >	ResImageQuality3Class;

	/*****************************ExtractFeat*****************************/
	nRet = ExtractFeat( image, ImageID, layerName, vecIn73ClassFeat, vecIn6ClassFeat, vecAds6ClassFeat, vecImageQuality3ClassFeat );
	if ( (nRet != 0) || (vecIn73ClassFeat.size()<1) || (vecIn6ClassFeat.size()<1) || 
		 				(vecAds6ClassFeat.size()<1) || (vecImageQuality3ClassFeat.size()<1) )
	{
		cout<<"Fail to GetFeat!! "<<endl;
		return TEC_BAD_STATE;
	}

	/************************Predict In37Class*****************************/
	{
		/************************SVM 73class Predict*****************************/	
		tmpRes.clear();
		nRet = api_libsvm_73class.Predict( vecIn73ClassFeat, tmpRes );	//PCA_FEAT FOR 72Class
		if (nRet != 0)
		{	
		   	cout<<"Fail to Predict SVM 72class !! "<<endl; 
		   	return nRet;
		}		

		/************************Merge37classLabel*****************************/	
		ResIn37Class.clear();
		tmpResIn37Class.clear();
		//nRet = Merge73classLabel( tmpRes, ResIn37Class, tmpResIn37Class );
		nRet = Merge37classLabel( tmpRes, ResIn37Class, tmpResIn37Class );
		if (nRet != 0)
		{	
		   	cout<<"Fail to Merge37classLabel!! "<<endl; 
		   	return nRet;
		}
		
		//Merge37classLabel
		if (ResIn37Class[0].first == "other.other.other")	//recall
		{ 
			/************************SVM in6Class Predict*****************************/	
			tmpRes.clear();
			//printf("api_libsvm_in6Class.Predict!\n");
			nRet = api_libsvm_in6Class.Predict( vecIn6ClassFeat, tmpRes );	// FOR in6class
			if (nRet != 0)
			{	
			   	cout<<"Fail to Predict SVM in5class !! "<<endl; 
			   	return nRet;
			}

			/************************MergeIn6ClassLabel*****************************/	
			ResIn37Class.clear();
			tmpResIn37Class.clear();
			//printf("api_libsvm_in6Class.MergeIn6ClassLabel!\n");
			nRet = MergeIn6ClassLabel( tmpRes, ResIn37Class, tmpResIn37Class );
			if (nRet != 0)
			{	
			   	cout<<"Fail to MergeIn6ClassLabel!! "<<endl; 
			   	return nRet;
			}
		}
	}

	/************************Predict Ads6Class*****************************/
	{
		/***********************************Resize Img**********************************/
		IplImage *imageResize = cvCreateImage(cvSize(WIDTH, HEIGHT), image->depth, image->nChannels);
		cvResize( image, imageResize );

		/***********************************SimilarDetect**********************************/
		//printf( "Start GetAdsLabel[1]:SimilarDetect...\n");
		SD_RES result_SD;
		nRet = SD_GLOBAL_1_1_0::SimilarDetect( imageResize, ImageID, 70020, 70020, &result_SD, 0 );//search SD
		if ( nRet!=TOK )
		{
			cout<<"SimilarDetect err!!" << endl;
			cvReleaseImage(&imageResize);imageResize = 0;
			return nRet;
		}

		/************************Get Ads6Class*****************************/	
		if ( result_SD.sMode == SD_eSame )//detect similar image
			ResAds6Class.push_back( std::make_pair( "ads.ads.ads", 1.0 ) );
		else if ( ( result_SD.sMode != SD_eSame ) && ( ResIn37Class[0].first == "other.2dcode.2dcode" ) )
			ResAds6Class.push_back( make_pair( "ads.2dcode.2dcode", ResIn37Class[0].second ) );
		else if ( ( result_SD.sMode != SD_eSame ) && ( ResIn37Class[0].first != "other.2dcode.2dcode" ) && 
				  ( ResIn37Class[0].first != "other.text.text" ) && (ResIn37Class[0].first != "other.other.other") )
			ResAds6Class.push_back( make_pair( "ads.norm.norm", ResIn37Class[0].second ) );		 //in73class:text/other
		else
		{
			/************************SVM ads5class Predict*****************************/	
			tmpRes.clear();
			nRet = api_libsvm_ads6Class.Predict( vecAds6ClassFeat, tmpRes );	// FOR ads6class
			if (nRet != 0)
			{	
			   	cout<<"Fail to Predict SVM ads5class !! "<<endl; 
			   	return nRet;
			}

			/************************Merge5ClassLabel*****************************/	
			ResAds6Class.clear();
			nRet = MergeAds6ClassLabel( tmpRes, ResAds6Class );
			if (nRet != 0)
			{	
			   	cout<<"Fail to Merge5ClassLabel!! "<<endl; 
			   	return nRet;
			}
		}
		cvReleaseImage(&imageResize);imageResize = 0;
	}

	/************************Predict ImageQuality*****************************/
	{
		/************************SVM imgquality3class Predict*****************************/	
		tmpRes.clear();
		nRet = api_libsvm_imgquality3class.Predict( vecImageQuality3ClassFeat, tmpRes );	// FOR ads6class
		if (nRet != 0)
		{	
		   	cout<<"Fail to Predict SVM imgquality3class !! "<<endl; 
		   	return nRet;
		}

		/************************Merge5ClassLabel*****************************/	
		ResImageQuality3Class.clear();
		nRet = MergeImageQuality3ClassLabel( tmpRes, tmpResIn37Class, ResImageQuality3Class );	//err
		if (nRet != 0)
		{	
		   	cout<<"Fail to Merge5ClassLabel!! "<<endl; 
		   	return nRet;
		}	
	}

	/************************Merge Res*****************************/
	Res.clear();
	if ( ResIn37Class.size() > 0 )
		Res.push_back( make_pair( ResIn37Class[0].first, ResIn37Class[0].second ) );
	else
		Res.push_back( make_pair( "null", 0 ) );
	if ( ResAds6Class.size() > 0 )
		Res.push_back( make_pair( ResAds6Class[0].first, ResAds6Class[0].second ) );
	else
		Res.push_back( make_pair( "null", 0 ) );
	if ( ResImageQuality3Class.size() > 0 )
		Res.push_back( make_pair( ResImageQuality3Class[0].first, ResImageQuality3Class[0].second ) );
	else
		Res.push_back( make_pair( "null", 0 ) );

	return nRet;
}

int API_ONLINE_CLASSIFICATION::Merge37classLabel(
	vector< pair< int, float > >		inImgLabel, 		//[In]:ImgDetail from GetLabel
	vector< pair< string, float > > 	&LabelInfo,			//[Out]:LabelInfo
	vector< pair< int,float > > 		&intLabelInfo )		//[Out]:intLabelInfo
{
	if ( inImgLabel.size() < 1 ) 
	{
		//printf( "MergeLabel[err]:imgDetail.label.size():%d\n", imgDetail.label.size() );
		return 1;
	}
	
	int i,index,tmpLabel,nRet = 0;
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
	
	for ( i=0;i<inImgLabel.size();i++ )
	{
		tmpLabel = 0;
		index = inImgLabel[i].first;
		score = inImgLabel[i].second;
		//printf( "MergeLabel:i-%d,index-%d,score-%.4f\n",i,index, score );
		if ( i == 0 )
		{
			if (
#ifdef USE_LINEARSVM
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
#else
				( ( index>=0  ) && ( index<5  ) && (score>=0.9) ) || 	
				( ( index==5  ) 				&& (score>=0.8) ) || 	//cook
				( ( index>=6  ) && ( index<9  ) && (score>=0.9) ) ||	//food.*
				( ( index==9  ) 				&& (score>=0.8) ) || 	//fruit
				( ( index>=10 ) && ( index<28 ) && (score>=0.9) ) ||	//good.*
				( ( index==28 ) 				&& (score>=0.8) ) || 	//flower
				( ( index>=29 ) && ( index<43 ) && (score>=0.9) ) ||	//good.*
				( ( index>=43 ) && ( index<48 ) && (score>=0.8) ) ||	//people.*
				( ( index>=48 ) && ( index<51 ) && (score> 0.7) ) ||	//people.self.* && people.street
				( ( index>=51 ) && ( index<71 ) && (score>=0.9) ) || 
				( ( index==71 ) 				&& (score>=0.8) ) || 	//text
				( ( index>=72 ) && ( index<73 ) && (score>=0.9) )
#endif
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


int API_ONLINE_CLASSIFICATION::Merge73classLabel(
	vector< pair< int, float > >		inImgLabel, 		//[In]:ImgDetail from GetLabel
	vector< pair< string, float > > 	&LabelInfo,			//[Out]:LabelInfo
	vector< pair< int,float > > 		&intLabelInfo )		//[Out]:intLabelInfo
{
	if ( inImgLabel.size() < 1 ) 
	{
		printf( "MergeLabel[err]:inImgLabel.size()<1!!\n", inImgLabel.size() );
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
			printf( "Merge72classLabel[err]:label:%d out of size:%d!!\n", label,dic_73Class.size() );
			return TEC_INVALID_PARAM;
		}
	}
	
	return nRet;
}

int API_ONLINE_CLASSIFICATION::MergeIn6ClassLabel(
	vector< pair< int, float > > 		inImgLabel, 		//[In]:inImgLabel
	vector< pair< string, float > > 	&LabelInfo,			//[Out]:outImgLabel
	vector< pair< int,float > > 		&intLabelInfo )		//[Out]:intLabelInfo
{
	if ( inImgLabel.size() < 1 ) 
	{
		printf( "MergeLabel[err]:inImgLabel.size()<1!!\n" );
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
#ifdef USE_LINEARSVM
				( ( index>=0  ) && ( index<2  ) && (score>=0.8) ) || 	//food other
				( ( index==2  ) 				&& (score>=0.6) ) || 	//people
				( ( index>=3  ) && ( index<6  ) && (score>=0.8) ) 		//pet puppet scene
#else
				( ( index>=0  ) && ( index<2  ) && (score>=0.8) ) || 	//food other
				( ( index==2  ) 				&& (score>=0.6) ) || 	//people
				( ( index>=3  ) && ( index<6  ) && (score>=0.8) )		//pet puppet scene*/
#endif
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


int API_ONLINE_CLASSIFICATION::MergeAds6ClassLabel(
	vector< pair< int, float > > 		inImgLabel, 		//[In]:inImgLabel
	vector< pair< string, float > > 	&LabelInfo)			//[Out]:outImgLabel
{
	if ( inImgLabel.size() < 1 ) 
	{
		printf( "MergeLabel[err]:inImgLabel.size()<1!!\n", inImgLabel.size() );
		return TEC_INVALID_PARAM;
	}
	
	int i,label,nRet = 0;
	float score = 0.0;
	LabelInfo.clear();

	for ( i=0;i<inImgLabel.size();i++ )
	{
		label = inImgLabel[i].first;
		score = inImgLabel[i].second;		
			
		if ( label < dic_ads6Class.size() )
		{
			if ( dic_ads6Class[label] == "ads" ) //ads
				LabelInfo.push_back( make_pair( "ads.ads.ads", score ) );
			else if ( dic_ads6Class[label] == "2dcode" ) //2dcode
				LabelInfo.push_back( make_pair( "ads.2dcode.2dcode", score ) );
			else if ( dic_ads6Class[label] == "text_qq" ) //text_qq
				LabelInfo.push_back( make_pair( "ads.text.text", score ) );
			else
				LabelInfo.push_back( make_pair( "ads.norm.norm", score ) );
		}
		else
		{
			printf( "MergeAds6ClassLabel[err]:label out of size:%d!!\n", dic_ads6Class.size() );
			return TEC_INVALID_PARAM;
		}
	}
	
	return nRet;
}

int API_ONLINE_CLASSIFICATION::MergeImageQuality3ClassLabel(
	vector< pair< int, float > >		inImgQualityLabel, 		//[In]:inImgQualityLabel
	vector< pair< int, float > >		inImgClassLabel, 		//[In]:inImgClassLabel
	vector< pair< string, float > > 	&LabelInfo) 			//[Out]:outImgLabel
{
	if ( inImgQualityLabel.size() < 1 ) 
	{
		printf( "MergeLabel[err]:inImgLabel.size()<1!!\n", inImgQualityLabel.size() );
		return TEC_INVALID_PARAM;
	}
	
	int i,label,imgQualityLabel,nRet = 0;
	float score = 0.0;
	const int filterImageQualityLabel[37] = {	
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 0, 0, 0 };	//	all 37 class except: sticker/text/2dcode
/*	const int filterImageQualityLabel[36] = {	
		0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  
		0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 
		1, 1, 1, 1, 0, 0 };	//	scene class
	LabelInfo.clear();*/

#ifndef USE_CLASSLABEL
	inImgClassLabel.clear();
#endif

	if ( inImgClassLabel.size() < 1 )	//no class label
	{
		for ( i=0;i<inImgQualityLabel.size();i++ )
		{
			label = inImgQualityLabel[i].first;
			score = inImgQualityLabel[i].second;		

			if ( label < dic_imgquality3class.size() )
				LabelInfo.push_back( make_pair( dic_imgquality3class[label], score ) );
			else
			{
				printf( "dic_imgquality3class[err]:label out of size:%d!!\n", dic_imgquality3class.size() );
				return TEC_INVALID_PARAM;
			}
		}
	}
	else
	{
		for ( i=0;i<inImgClassLabel.size();i++ )
		{
			label = inImgClassLabel[i].first;		

			if ( label < dic_in37class.size() )
			{
				if ( filterImageQualityLabel[ label ] == 1 )
					LabelInfo.push_back( make_pair( dic_imgquality3class[inImgQualityLabel[0].first], inImgQualityLabel[0].second ) );
				else
					LabelInfo.push_back( make_pair( dic_imgquality3class[1], 2.0 ) );	//medial
			}
			else
			{
				printf( "dic_online[err]:label out of size:%d!!\n", dic_in37class.size() );
				return TEC_INVALID_PARAM;
			}
		}
	}
	
	return nRet;
}



/***********************************Release**********************************/
void API_ONLINE_CLASSIFICATION::Release()
{
	/***********************************Similar Detect******************************/
	SD_GLOBAL_1_1_0::Uninit();

	/***********************************PCA Model**********************************/
#ifndef USE_LINEARSVM
	api_pca.PCA_Feat_Release();
#endif
	
	/***********************************net Model**********************************/
	api_caffe.Release();

	/***********************************dict Model**********************************/
	dic_73Class.clear();
	dic_in37class.clear();
	dic_ads6Class.clear();
	dic_imgquality3class.clear();

	/***********************************SVM Model******libsvm3.2.0********************/
	api_libsvm_73class.Release();
	api_libsvm_in6Class.Release();
	api_libsvm_ads6Class.Release();
	api_libsvm_imgquality3class.Release();
}





