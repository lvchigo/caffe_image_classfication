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

#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/legacy/legacy.hpp"

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
#include "API_pca.h"
#include "API_libsvm.h"
#include "API_in5class_classification.h"
#include "TErrorCode.h"

using namespace cv;
using namespace std;

/***********************************Init*************************************/
/// construct function 
API_IN5CLASS_CLASSIFICATION::API_IN5CLASS_CLASSIFICATION()
{
}

/// destruct function 
API_IN5CLASS_CLASSIFICATION::~API_IN5CLASS_CLASSIFICATION(void)
{
}

/***********************************Init*************************************/
int API_IN5CLASS_CLASSIFICATION::Init( const char* KeyFilePath, char *svVocabulary )
{
	int i,nRet = 0;
	char tPath[1024] = {0};

	/********************************train BOVW SVMs*****************************/
	cout << "read vocabulary form file"<<endl;
	FileStorage fs( svVocabulary, FileStorage::READ);
	fs["vocabulary"] >> vocabulary;
	fs.release();

	/***********************************Init Model_SVM_in5Class*************************************/
	sprintf(tPath, "%s/in6class/in6class_AllData_051401_VGG16_SVM.model", KeyFilePath);
	nRet = api_libsvm_in5Class.Init(tPath); 
	if (nRet != 0)
	{
	   cout<<"Fail to initialization "<<endl;
	   return TEC_INVALID_PARAM;
	}

	/***********************************Load dic_5Class File**********************************/
	dic_5Class.clear();
	sprintf(tPath, "%s/in6class/dict_in6class",KeyFilePath);
	api_commen.loadWordDict(tPath,dic_5Class);
	printf( "dict:size-%d,tag:", int(dic_5Class.size()) );
	for ( i=0;i<dic_5Class.size();i++ )
	{
		printf( "%d-%s ",i,dic_5Class[i].c_str() );
	}
	printf( "\n" );

	return nRet;
}

int API_IN5CLASS_CLASSIFICATION::ExtractFeat( 
	IplImage						*img, 				//[In]:image
	UInt64							ImageID,			//[In]:ImageID
	vector< vector< float > > 		&BOWFeat ) 			//[Out]:BOWFeat
{
	if(!img || (img->width<16) || (img->height<16) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
	{	
		cout<<"image err!!" << endl;
		return TEC_INVALID_PARAM;
	}	
	
	/*****************************Init*****************************/
	int i, j, rWidth, rHeight, svImg, nRet = 0;
	BOWFeat.clear();

	Mat descriptors;
	Mat response_hist;
	vector<KeyPoint> keypoints;
	vector< float > bowFeat;
	vector< float > bowNormFeat;

	/********************************Init BOVW*****************************/
	Ptr<FeatureDetector > detector(new SurfFeatureDetector()); //detector
	//Ptr<DescriptorExtractor > extractor(new SurfDescriptorExtractor());//  extractor;
	Ptr<DescriptorExtractor > extractor(
		new OpponentColorDescriptorExtractor(
			 Ptr<DescriptorExtractor>(new SurfDescriptorExtractor())
		 )
	);
	Ptr<DescriptorMatcher > matcher(new BruteForceMatcher<L2<float> >());
	BOWImgDescriptorExtractor bowide(extractor,matcher);
	bowide.setVocabulary(vocabulary);

	/*****************************Remove White Part*****************************/
	IplImage* ImgRemoveWhite = api_commen.RemoveWhitePart( img, ImageID );

	/***********************************Resize Image width && height*****************************/
	nRet = api_commen.GetReWH( ImgRemoveWhite->width, ImgRemoveWhite->height, 256, rWidth, rHeight );
	if (nRet != 0)
	{
	   	cout<<"Fail to GetReWH!! "<<endl;
		cvReleaseImage(&ImgRemoveWhite);ImgRemoveWhite = 0;
		
	   	return nRet;
	}

	/*****************************Resize Img*****************************/
	IplImage *ImgResize = cvCreateImage(cvSize(rWidth, rHeight), img->depth, img->nChannels);
	cvResize( ImgRemoveWhite, ImgResize );
	
	/*****************************GetLabelFeat*****************************/	
	Mat matResizeImg( ImgResize );	
	detector->detect( matResizeImg, keypoints );
	if ( keypoints.size() == 0 )
	{	
		printf( "ImageID:%ld,keypoints:0!!", ImageID );
		cvReleaseImage(&ImgRemoveWhite);ImgRemoveWhite = 0;
		cvReleaseImage(&ImgResize);ImgResize = 0;
		
		return TEC_INVALID_PARAM;
	}	
	
	/*****************************compute BOW Feat*****************************/	
	bowide.compute(img, keypoints, response_hist);
	for(i=0;i<response_hist.rows;i++)
	{
		for(j=0;j<response_hist.cols;j++)
		{
			 bowFeat.push_back( response_hist.at<float>(i,j) );
		}
	}
			
	/************************DL Feat Normal*****************************/
	nRet = api_commen.Normal_L2( bowFeat, bowNormFeat );
	if (nRet != 0)
	{
		cout<<"Fail to Normal_L2!! "<<endl;
		cvReleaseImage(&ImgRemoveWhite);ImgRemoveWhite = 0;
		cvReleaseImage(&ImgResize);ImgResize = 0;
		return nRet;
	}

	BOWFeat.push_back( bowNormFeat );

	/*********************************Release*************************************/
	cvReleaseImage(&ImgRemoveWhite);ImgRemoveWhite = 0;
	cvReleaseImage(&ImgResize);ImgResize = 0;

	//printf("compute BOW Feat: hist.rows:%d,hist.cols:%d\n",response_hist.rows,response_hist.cols);

	return 0;
}


/***********************************Predict**********************************/
int API_IN5CLASS_CLASSIFICATION::Predict(
	IplImage						*image, 			//[In]:image
	UInt64							ImageID,			//[In]:ImageID
	vector< pair< string, float > >	&Res)				//[Out]:Res
{
	if(!image || (image->width<16) || (image->height<16) || image->nChannels != 3 || image->depth != IPL_DEPTH_8U) 
	{	
		cout<<"image err!!" << endl;
		return TEC_INVALID_PARAM;
	}	

	/*****************************Init*****************************/
	int i,nRet = 0;
	vector< vector<float> > imgFeat;
	vector < pair < int,float > > tmpRes;
	Res.clear();

	/************************Feat Extract*****************************/	
	nRet = ExtractFeat( image, ImageID, imgFeat );
	if (nRet != 0)
	{	
	   	cout<<"Fail to ExtractFeat !! "<<endl;
	   	return nRet;
	}

	/************************SVM in5class Predict*****************************/	
	tmpRes.clear();
	nRet = api_libsvm_in5Class.Predict( imgFeat, tmpRes );	//NormDLFeat FOR in5class
	if (nRet != 0)
	{	
	   	cout<<"Fail to Predict SVM in5class !! "<<endl;
	   	return nRet;
	}

	/************************MergeIn5ClassLabel*****************************/	
	Res.clear();
	nRet = MergeIn5ClassLabel( tmpRes, Res );
	if (nRet != 0)
	{	
	   	cout<<"Fail to MergeIn5ClassLabel!! "<<endl;
	   	return nRet;
	}

	return nRet;
}

int API_IN5CLASS_CLASSIFICATION::MergeIn5ClassLabel(
	vector< pair< int, float > > 		inImgLabel, 		//[In]:inImgLabel
	vector< pair< string, float > > 	&LabelInfo)			//[Out]:outImgLabel
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
	
	for ( i=0;i<inImgLabel.size();i++ )
	{
		tmpLabel = 0;
		index = inImgLabel[i].first;
		score = inImgLabel[i].second;
		//printf( "MergeLabel:i-%d,index-%d,score-%.4f\n",i,index, score );
		if ( 0 == i )
		{
			if (( ( index>=0  ) && ( index<2  ) && (score>=0.8) ) || 	//food other
				( ( index==2  ) 				&& (score>=0.6) ) || 	//people
				( ( index>=3  ) && ( index<6  ) && (score>=0.8) ) )		//pet puppet scene
			{
				tmpLabel = onlineLabel[index];
			}
			else
			{
				tmpLabel = 0;
			}
			LabelInfo.push_back( std::make_pair( dic_5Class[tmpLabel], score ) );
			//printf("index:%d,label:%s,score:%.4f\n",tmpLabel,dic_online[tmpLabel].c_str(),score);
		}
	}
	
	return nRet;
}

/***********************************Release**********************************/
void API_IN5CLASS_CLASSIFICATION::Release()
{
	/***********************************dict Model**********************************/
	dic_5Class.clear();

	/***********************************SVM Model******libsvm3.2.0********************/
	api_libsvm_in5Class.Release();
}




