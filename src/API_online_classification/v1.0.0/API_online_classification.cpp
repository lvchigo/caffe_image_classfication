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
#include "SD_global.h"
#include "API_online_classification.h"
#include "TErrorCode.h"
#include "plog/Log.h"


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

	/***********************************Init**********************************/
	sprintf(tPath, "%s/log/plog.txt",KeyFilePath);	//logout
	plog::init(plog::error, tPath); 

	/***********************************Load similardetect File**********************************/
	SD_GLOBAL_1_1_0::ClassInfo ci;
	vector < SD_GLOBAL_1_1_0::ClassInfo > classList;
	
	sprintf(tPath, "%s/SimilarDetect/keyfile",KeyFilePath);
	printf("load similardetect File:%s\n",tPath);
	nRet  = SD_GLOBAL_1_1_0::Init(tPath);
	if (nRet != 0)
	{
	   LOOGE<<"similardetect init err :can't open"<<tPath;
	   return TEC_INVALID_PARAM;
	}
	
	ci.ClassID = 70020;
	ci.SubClassID = 70020;
	classList.push_back(ci);
	nRet = SD_GLOBAL_1_1_0::LoadClassData(classList);
	if (nRet != 0)
	{
	   LOOGE<<"similardetect init err : Fail to LoadClassData. Error code:"<<nRet;
	   return TEC_INVALID_PARAM;
	}
	
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

	/***********************************Init Model_SVM_adsRecall3Class*************************************/
	sprintf(tPath, "%s/ads6class/linearsvm_adsRecall3Class_151104.model", KeyFilePath);	//linearSVM
	printf("load Model_SVM_adsRecall3Class:%s\n",tPath);
	nRet = api_libsvm_adsRecall3Class.Init(tPath); 
	if (nRet != 0)
	{
	   LOOGE<<"Fail to initialization "<<tPath;
	   return TEC_INVALID_PARAM;
	}
	
	/***********************************Init Model_SVM_ads6Class*************************************/
	//sprintf(tPath, "%s/ads6class/linearsvm_ads6class_050702.model", KeyFilePath);	//linearSVM
	sprintf(tPath, "%s/ads6class/linearsvm_ads6class_151106.model", KeyFilePath);	//linearSVM
	printf("load Model_SVM_ads6Class:%s\n",tPath);
	nRet = api_libsvm_ads6Class.Init(tPath); 
	if (nRet != 0)
	{
	   LOOGE<<"Fail to initialization "<<tPath;
	   return TEC_INVALID_PARAM;
	}
	
	/***********************************Init Model_imgquality3class*************************************/
	sprintf(tPath, "%s/imagequality/linearsvm_imagequality3class_food_050819.model",KeyFilePath);	//linearSVM
	printf("load api_libsvm_imgquality3class_food:%s\n",tPath);
	nRet = api_libsvm_imgquality3class_food.Init(tPath); 

	sprintf(tPath, "%s/imagequality/linearsvm_imagequality3class_scene_050819.model",KeyFilePath);	//linearSVM
	printf("load api_libsvm_imgquality3class_scene:%s\n",tPath);
	nRet += api_libsvm_imgquality3class_scene.Init(tPath); 

	sprintf(tPath, "%s/imagequality/linearsvm_imagequality3class_street_050819.model",KeyFilePath);	//linearSVM
	printf("load api_libsvm_imgquality3class_street:%s\n",tPath);
	nRet += api_libsvm_imgquality3class_street.Init(tPath); 

	sprintf(tPath, "%s/imagequality/linearsvm_imagequality3class_pet_050819.model",KeyFilePath);	//linearSVM
	printf("load api_libsvm_imgquality3class_pet:%s\n",tPath);
	nRet = api_libsvm_imgquality3class_pet.Init(tPath); 

	sprintf(tPath, "%s/imagequality/linearsvm_imagequality3class_goods_050814_vgg16.model",KeyFilePath);	//linearSVM
	printf("load api_libsvm_imgquality3class_goods:%s\n",tPath);
	nRet += api_libsvm_imgquality3class_goods.Init(tPath); 

	sprintf(tPath, "%s/imagequality/linearsvm_imagequality3class_people_050817_vgg16.model",KeyFilePath);	//linearSVM
	printf("load api_libsvm_imgquality3class_people:%s\n",tPath);
	nRet = api_libsvm_imgquality3class_people.Init(tPath); 

	sprintf(tPath, "%s/imagequality/linearsvm_imagequality3class_other_050818_vgg16.model",KeyFilePath);	//linearSVM
	printf("load api_libsvm_imgquality3class_other:%s\n",tPath);
	nRet += api_libsvm_imgquality3class_other.Init(tPath); 
	if (nRet != 0)
	{
	   LOOGE<<"Fail to initialization "<<tPath;
	   return TEC_INVALID_PARAM;
	}
	
	/***********************************Init Model_imgqualityblur2class*************************************/
	sprintf(tPath, "%s/imagequalityblur/linearsvm_imagequalityblur2class_050819.model",KeyFilePath);	//linearSVM
	printf("load api_libsvm_imgqualityblur2class:%s\n",tPath);
	nRet = api_libsvm_imgqualityblur2class.Init(tPath); 
	if (nRet != 0)
	{
	   LOOGE<<"Fail to initialization "<<tPath;
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
	vector< vector< float > > 		&vecImageQuality3ClassFeat,			//for imagequality
	vector< vector< float > > 		&vecImageQualityBlurFeat )			//for imagequality Blur
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
	vector<float> normAds6ClassFeat;
	vector<float> normImageQuality3ClassFeat;
	vector<float> normImageQualityBlurFeat;
	vector<float> tmpImageQualityBlurFeat;
	
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
	vecIn73ClassFeat.clear();
	vecIn6ClassFeat.clear();
	vecAds6ClassFeat.clear();
	vecImageQuality3ClassFeat.clear();
	vecImageQualityBlurFeat.clear();
	tmpImageQualityBlurFeat.clear();
	for(i=0;i<imgFeat.size();i++)
	{
		if ( i == 0 )
		{
			normIn73ClassFeat.clear();
			normIn6ClassFeat.clear();
			normAds6ClassFeat.clear();
			EncodeFeat.clear();	

			/************************DL Ads6ClassFeat Normal*****************************/
			nRet = api_commen.Normal_L2(imgFeat[i],normAds6ClassFeat);
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

			vecIn73ClassFeat.push_back(normAds6ClassFeat);	//linearSVM, FOR in73Class
			vecIn6ClassFeat.push_back(normIn6ClassFeat);	// FOR in6class
			vecAds6ClassFeat.push_back(normAds6ClassFeat);	// FOR ads6class	
		}
		else if ( i == 1 )
		{
			normImageQuality3ClassFeat.clear();
			
			/************************DL Feat Normal*****************************/
			nRet = api_commen.Normal_MinMax(imgFeat[i],normImageQuality3ClassFeat);
			if (nRet != 0)
			{
			   LOOGE<<"Fail to Normal_MinMax!!";
			   return nRet;
			}

			vecImageQuality3ClassFeat.push_back(normImageQuality3ClassFeat);	// FOR ImageQuality3Class
		}
		else
		{
			//Merge to one vector
			for( j=0;j<imgFeat[i].size();j++ )
				tmpImageQualityBlurFeat.push_back(imgFeat[i][j]);	// FOR ImageQualityBlur
		}
	}

	/************************DL ImageQualityBlurFeat Normal*****************************/
	if ( tmpImageQualityBlurFeat.size() > 0 )
	{
		normImageQualityBlurFeat.clear();
		nRet = api_commen.Normal_L2(tmpImageQualityBlurFeat,normImageQualityBlurFeat);
		if (nRet != 0)
		{
		   LOOGE<<"Fail to Normal_L2!!";
		   return nRet;
		}

		//push data
		vecImageQualityBlurFeat.push_back( normImageQualityBlurFeat );
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
		LOOGE<<"image err!!";
		return TEC_INVALID_PARAM;
	}	

	/*****************************Init*****************************/
	int i,nRet = 0;
	double tPredictTime = 0;
	RunTimer<double> run;
	
	vector< pair< int, float > > imgLabel;
	vector< vector<float> > imgFeat;
	vector< vector<float> > vecIn73ClassFeat;
	vector< vector<float> > vecIn6ClassFeat;
	vector< vector<float> > vecAds6ClassFeat;
	vector< vector<float> > vecImageQuality3ClassFeat;
	vector< vector<float> > vecImageQualityBlurFeat;
	vector < pair < int,float > > tmpRes;
	vector < pair < int,float > > tmpResIn37Class;
	vector< pair< string, float > >	ResIn37Class;
	vector< pair< string, float > >	ResAds6Class;
	vector< pair< string, float > >	ResImageQuality3Class;
	vector < pair < int,float > > tmpImageQualityBlur2Class;
	vector < pair < int,float > > ResAdsRecall3Class;

	/*****************************ExtractFeat*****************************/
	//tPredictTime = (double)getTickCount();
	run.start();
	nRet = ExtractFeat( image, ImageID, layerName, vecIn73ClassFeat, vecIn6ClassFeat, vecAds6ClassFeat, vecImageQuality3ClassFeat, vecImageQualityBlurFeat );
//	if ( (nRet != 0) || (vecIn73ClassFeat.size()<1) || (vecIn6ClassFeat.size()<1) || 
//		 (vecAds6ClassFeat.size()<1) || (vecImageQuality3ClassFeat.size()<1) || (vecImageQualityBlurFeat.size()<1) )
	if ( (nRet != 0) || (vecIn73ClassFeat.size()<1) || (vecIn6ClassFeat.size()<1) || 
		 (vecAds6ClassFeat.size()<1) )
	{
		LOOGE<<"Fail to GetFeat!!";
		return TEC_BAD_STATE;
	}
	run.end();
	LOOGI<<"[Predict--ExtractFeat] time:"<<run.time();
	//tPredictTime = (double)getTickCount() - tPredictTime;
	//tPredictTime = tPredictTime*1000./cv::getTickFrequency();
	//printf( "[Time ExtractFeat]:%.4fms\n", tPredictTime );

	/************************Predict In37Class*****************************/
	{
		/************************SVM 73class Predict*****************************/	
		tmpRes.clear();
		nRet = api_libsvm_73class.Predict( vecIn73ClassFeat, tmpRes );	//PCA_FEAT FOR 72Class
		if (nRet != 0)
		{	
			LOOGE<<"Fail to Predict SVM 72class !!";
		   	return nRet;
		}		

		/************************Merge37classLabel*****************************/	
		ResIn37Class.clear();
		tmpResIn37Class.clear();
		//nRet = Merge73classLabel( tmpRes, ResIn37Class, tmpResIn37Class );
		nRet = Merge37classLabel( tmpRes, ResIn37Class, tmpResIn37Class );
		if (nRet != 0)
		{	
			LOOGE<<"Fail to Merge37classLabel!!";
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
				LOOGE<<"Fail to Predict SVM in5class!!";
			   	return nRet;
			}

			/************************MergeIn6ClassLabel*****************************/	
			ResIn37Class.clear();
			tmpResIn37Class.clear();
			//printf("api_libsvm_in6Class.MergeIn6ClassLabel!\n");
			nRet = MergeIn6ClassLabel( tmpRes, ResIn37Class, tmpResIn37Class );
			if (nRet != 0)
			{	
				LOOGE<<"Fail to MergeIn6ClassLabel!!";
			   	return nRet;
			}
		}
	}

	/************************Predict AdsRecall3Class*****************************/
	{
		ResAdsRecall3Class.clear();
		nRet = api_libsvm_adsRecall3Class.Predict( vecAds6ClassFeat, ResAdsRecall3Class );
		if (nRet != 0)
		{	
			LOOGE<<"Fail to Predict SVM adsRecall3Class!!";
		   	return nRet;
		}

		/*****************************save img data*****************************/
/*		char szImgPath[1024] = {0};
		for( i=0;i<ResAdsRecall3Class.size();i++ )
		{	
			sprintf(szImgPath, "adsRecall3class/%d/%.2f_%ld.jpg",
				ResAdsRecall3Class[i].first,ResAdsRecall3Class[i].second,ImageID);
			cvSaveImage( szImgPath,image );

			//printf("id:%ld,adsRecall3class:%d-%.2f\n",ImageID,tmpRes[i].first,tmpRes[i].second);
		}*/
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
			LOOGE<<"SimilarDetect err!!";
			cvReleaseImage(&imageResize);imageResize = 0;
			return nRet;
		}

		/************************Get Ads6Class*****************************/	
		if ( result_SD.sMode == SD_eSame )//detect similar image
			ResAds6Class.push_back( std::make_pair( "ads.ads.ads", 1.0 ) );
		else if ( ( result_SD.sMode != SD_eSame ) && ( ResIn37Class[0].first == "other.2dcode.2dcode" ) )
			ResAds6Class.push_back( make_pair( "ads.2dcode.2dcode", ResIn37Class[0].second ) );
//		else if ( ( result_SD.sMode != SD_eSame ) && ( ResIn37Class[0].first != "other.2dcode.2dcode" ) && 
//				  ( ResIn37Class[0].first != "other.text.text" ) && (ResIn37Class[0].first != "other.other.other") )
		else if ( ( result_SD.sMode != SD_eSame ) && ( ResIn37Class[0].first != "other.2dcode.2dcode" ) && 
				  ( ResIn37Class[0].first != "other.text.text" ) && (ResIn37Class[0].first != "other.other.other") &&
				  ( ResAdsRecall3Class[0].first == 0 ) )
			ResAds6Class.push_back( make_pair( "ads.norm.norm", ResIn37Class[0].second ) );		 //in73class:text/other
		else
		{
			/************************SVM ads5class Predict*****************************/	
			tmpRes.clear();
			nRet = api_libsvm_ads6Class.Predict( vecAds6ClassFeat, tmpRes );	// FOR ads6class
			if (nRet != 0)
			{	
				LOOGE<<"Fail to Predict SVM ads5class!!";
			   	return nRet;
			}

			/************************Merge5ClassLabel*****************************/	
			ResAds6Class.clear();
			nRet = MergeAds6ClassLabel( tmpRes, ResAds6Class );
			if (nRet != 0)
			{	
				LOOGE<<"Fail to MergeAds6ClassLabel!!";
			   	return nRet;
			}
		}
		cvReleaseImage(&imageResize);imageResize = 0;
	}

	/************************Predict ImageQuality*****************************/
	{
		/************************SVM imgquality3class Predict*****************************/	
		tmpRes.clear();
		vector< int > ChooseClassification;	//[Out]:LabelInfo:0-other,1-food,2-scene,3-street,4-pet,-1-noClassfication
		nRet = ChooseImageQuality3Classification( tmpResIn37Class, ChooseClassification ); 		
		if ( (nRet != 0) || ( ChooseClassification.size() < 1 ) )
		{	
			LOOGE<<"Fail to ChooseImageQuality3Classification!!";
		   	return nRet;
		}	
		
		for (i=0;i<ChooseClassification.size();i++)
		{	
			if ( ChooseClassification[ i ] == 1 )
				nRet = api_libsvm_imgquality3class_food.Predict( vecImageQuality3ClassFeat, tmpRes );	// FOR food
			else if ( ChooseClassification[ i ] == 2 )
				nRet = api_libsvm_imgquality3class_scene.Predict( vecImageQuality3ClassFeat, tmpRes );	// FOR scene
			else if ( ChooseClassification[ i ] == 3 )
				nRet = api_libsvm_imgquality3class_street.Predict( vecImageQuality3ClassFeat, tmpRes );	// FOR street
			else if ( ChooseClassification[ i ] == 4 )
				nRet = api_libsvm_imgquality3class_pet.Predict( vecImageQuality3ClassFeat, tmpRes );	// FOR pet
			else if ( ChooseClassification[ i ] == 5 )
				nRet = api_libsvm_imgquality3class_goods.Predict( vecImageQuality3ClassFeat, tmpRes );	// FOR goods
			else if ( ChooseClassification[ i ] == 6 )
				nRet = api_libsvm_imgquality3class_people.Predict( vecImageQuality3ClassFeat, tmpRes );	// FOR people
			else if ( ChooseClassification[ i ] == 0 )
				nRet = api_libsvm_imgquality3class_other.Predict( vecImageQuality3ClassFeat, tmpRes );	// FOR other
			else
				tmpRes.push_back( std::make_pair( 1, 2.0 ) );											// FOR noClassfication
		}
		if (nRet != 0)
		{	
			LOOGE<<"Fail to Predict SVM imgquality3class!!";
		   	return nRet;
		}

		/************************SVM imgqualityblur2class Predict*****************************/	
		for(i=0;i<vecImageQualityBlurFeat.size();i++)
		{
			tmpImageQualityBlur2Class.clear();
			if ( (tmpRes[i].first == 2) && (tmpRes[i].second >= 0.8) )
			{
				nRet = api_libsvm_imgqualityblur2class.Predict( vecImageQualityBlurFeat, tmpImageQualityBlur2Class );
				if (nRet != 0)
				{	 
					LOOGE<<"Fail to Predict SVM imgqualityblur2class!!";
				   	return nRet;
				}

				/*****************************save img data*****************************/
/*				char szImgPath[1024] = {0};
				for( i=0;i<tmpImageQualityBlur2Class.size();i++ )
				{	
					sprintf(szImgPath, "blur/%d/%.2f_%ld.jpg",
						tmpImageQualityBlur2Class[i].first,tmpImageQualityBlur2Class[i].second,ImageID);
					cvSaveImage( szImgPath,image );

					//printf("id:%ld,blur:%d-%.2f\n",ImageID,tmpImageQualityBlur2Class[i].first,tmpImageQualityBlur2Class[i].second);
				}*/
			}

			/************************Merge5ClassLabel*****************************/	
			ResImageQuality3Class.clear();
			nRet = MergeImageQuality3ClassLabel( tmpRes, tmpImageQualityBlur2Class, tmpResIn37Class, ResImageQuality3Class );	//err
			if (nRet != 0)
			{	 
				LOOGE<<"Fail to MergeImageQuality3ClassLabel!!";
			   	return nRet;
			}	
		}
	}

	/************************Merge Res*****************************/
/*	Res.clear();
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
*/
	/************************Merge Res*****************************/
	Res.clear();
	if ( ResIn37Class.size() > 0 )
		Res.push_back( make_pair( ResIn37Class[0].first, ResIn37Class[0].second ) );
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
			if (														//data need adjust ****
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
			LOOGE<<"Merge72classLabel[err]!!";
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


int API_ONLINE_CLASSIFICATION::MergeAds6ClassLabel(
	vector< pair< int, float > > 		inImgLabel, 		//[In]:inImgLabel
	vector< pair< string, float > > 	&LabelInfo)			//[Out]:outImgLabel
{
	if ( inImgLabel.size() < 1 ) 
	{
		LOOGE<<"MergeLabel[err]:inImgLabel.size()<1!!";
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
			//else if ( dic_ads6Class[label] == "ads_colorword" ) //ads_colorword
			//	LabelInfo.push_back( make_pair( "ads.ads.colorword", score ) );
			else
				LabelInfo.push_back( make_pair( "ads.norm.norm", score ) );
		}
		else
		{
			LOOGE<<"MergeAds6ClassLabel[err]!!";
			return TEC_INVALID_PARAM;
		}
	}
	
	return nRet;
}

int API_ONLINE_CLASSIFICATION::MergeImageQuality3ClassLabel(
	vector< pair< int, float > >		inImgQualityLabel, 		//[In]:inImgQualityLabel
	vector< pair< int, float > >		inImgQualityBlurLabel, 	//[In]:inImgQualityBlurLabel
	vector< pair< int, float > >		inImgClassLabel, 		//[In]:inImgClassLabel
	vector< pair< string, float > > 	&LabelInfo) 			//[Out]:outImgLabel
{
	if ( inImgQualityLabel.size() < 1 )
	{
		LOOGE<<"MergeLabel[err]:inImgLabel.size()<1!!";
		return TEC_INVALID_PARAM;
	}
	
	int i,label,imgQualityLabel,nRet = 0;
	float score = 0.0;
	const int filterImageQualityLabel[37] = {	
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
		1, 1, 1, 1, 0, 0, 0 };	//usual use
		
	LabelInfo.clear();

#ifndef USE_CLASSLABEL
	inImgClassLabel.clear();
#endif

	if ( inImgClassLabel.size() < 1 )	//no class label
	{
		for ( i=0;i<inImgQualityLabel.size();i++ )
		{
			label = inImgQualityLabel[i].first;
			score = inImgQualityLabel[i].second;		

			if ( inImgQualityBlurLabel.size() > 0 ) 
			{
				if ( ( inImgQualityBlurLabel[0].first == 2 ) && ( inImgQualityBlurLabel[0].second >= 0.9 ) )	//blur-good
					LabelInfo.push_back( make_pair( dic_imgquality3class[2], inImgQualityBlurLabel[0].second ) );
				else
					LabelInfo.push_back( make_pair( dic_imgquality3class[0], 2.0 ) );		//blur-bad
			}
			else 
			{	
				if ( label < dic_imgquality3class.size() ) 		//imagequality
					LabelInfo.push_back( make_pair( dic_imgquality3class[label], score ) );		
				else
				{
					LOOGE<<"dic_imgquality3class[err]!!";
					return TEC_INVALID_PARAM;
				}
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
				if ( filterImageQualityLabel[ label ] == 1 )	//USE_CLASSLABEL
				{
					if ( inImgQualityBlurLabel.size() > 0 ) 
					{
						if ( ( inImgQualityBlurLabel[0].first == 2 ) && ( inImgQualityBlurLabel[0].second >= 0.9 ) )	//blur-good
							LabelInfo.push_back( make_pair( dic_imgquality3class[2], inImgQualityBlurLabel[0].second ) );
						else
							LabelInfo.push_back( make_pair( dic_imgquality3class[0], 2.0 ) );	//blur-bad
					}
					else 	//imagequality
					{	
						LabelInfo.push_back( make_pair( dic_imgquality3class[inImgQualityLabel[0].first], inImgQualityLabel[0].second ) );
					}
				}
				else			//Without CLASSLABEL
					LabelInfo.push_back( make_pair( dic_imgquality3class[1], 2.0 ) );	//medial
			}
			else
			{
				LOOGE<<"dic_online[err]!!";
				return TEC_INVALID_PARAM;
			}
		}
	}
	
	return nRet;
}


int API_ONLINE_CLASSIFICATION::ChooseImageQuality3Classification(
	vector< pair< int, float > >		inImgClassLabel, 		//[In]:inImgClassLabel
	vector< int >						&LabelInfo) 			//[Out]:LabelInfo:0-other,1-food,2-scene,3-street,-1-noClassfication
{
	if ( inImgClassLabel.size() < 1 ) 
	{
		LOOGE<<"ChooseImageQuality3Classification[err]:inImgClassLabel.size()<1!!";
		return TEC_INVALID_PARAM;
	}
	
	int i,label,imgQualityLabel,nRet = 0;
	float score = 0.0;

	//NO USE_IMGQUALITY_OTHER	//usual use
	const int filterImageQualityLabel[37] = {	
	    0, 1, 1, 1, 1, 1, 1, 1, 1, 5, 
	    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 
	    6, 6, 6, 6, 3, 6, 4, 4, 4, 2,
		2, 2, 2, 2,-1,-1,-1 };	//	0-other,1-food,2-scene,3-street,4-pet,5-goods,6-people,-1-noClassfication

	LabelInfo.clear();
	for ( i=0;i<inImgClassLabel.size();i++ )
	{
		label = inImgClassLabel[i].first;		

		if ( label < dic_in37class.size() )
		{
			if ( filterImageQualityLabel[ label ] == 0 )
				LabelInfo.push_back( 0 );
			else if ( filterImageQualityLabel[ label ] == 1 )
				LabelInfo.push_back( 1 );
			else if ( filterImageQualityLabel[ label ] == 2 )
				LabelInfo.push_back( 2 );
			else if ( filterImageQualityLabel[ label ] == 3 )
				LabelInfo.push_back( 3 );
			else if ( filterImageQualityLabel[ label ] == 4 )
				LabelInfo.push_back( 4 );
			else if ( filterImageQualityLabel[ label ] == 5 )
				LabelInfo.push_back( 5 );
			else if ( filterImageQualityLabel[ label ] == 6 )
				LabelInfo.push_back( 6 );
			else
				LabelInfo.push_back( -1 );
		}
		else
		{
			LOOGE<<"ChooseImageQuality3Classification[err]!!";
			return TEC_INVALID_PARAM;
		}
	}
	
	return nRet;
}

/***********************************Release**********************************/
void API_ONLINE_CLASSIFICATION::Release()
{
	/***********************************Similar Detect******************************/
	SD_GLOBAL_1_1_0::Uninit();
	
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
	api_libsvm_adsRecall3Class.Release();
	api_libsvm_ads6Class.Release();
	api_libsvm_imgquality3class_food.Release();
	api_libsvm_imgquality3class_scene.Release();
	api_libsvm_imgquality3class_street.Release();
	api_libsvm_imgquality3class_pet.Release();
	api_libsvm_imgquality3class_goods.Release();
	api_libsvm_imgquality3class_people.Release();
	api_libsvm_imgquality3class_other.Release();
	api_libsvm_imgqualityblur2class.Release();

}





