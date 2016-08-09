#pragma once
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

#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>
#include <opencv/ml.h>
#include <opencv/cvaux.h>

#include "API_commen.h"
#include "plog/Log.h"
#include "API_mutilabel.h"
#include "TErrorCode.h"

using namespace cv;
using namespace std;

//#define MutiLabel_T 0.8
//#define ZF 1

static bool Sort_Info(const MutiLabelInfo& elem1, const MutiLabelInfo& elem2)
{
    return (elem1.score > elem2.score);
}

/***********************************Init*************************************/
/// construct function 
API_MUTI_LABEL::API_MUTI_LABEL()
{
}

/// destruct function 
API_MUTI_LABEL::~API_MUTI_LABEL(void)
{
}

/***********************************Init*************************************/
int API_MUTI_LABEL::Init( 
	const char* 	KeyFilePath,						//[In]:KeyFilePath
	const int		binGPU, 							//[In]:USE GPU(1) or not(0)
	const int		deviceID )	 						//[In]:GPU ID
{
	char tPath[1024] = {0};
	char tPath2[1024] = {0};
	char tPath3[1024] = {0};

	string strLayerName = "fc7";
	nRet = 0;
	
	/***********************************Init**********************************/
#ifdef ZF	//zf
	sprintf(tPath, "%s/mutilabel/v2.0.0/faster_rcnn_test_zf.pt",KeyFilePath);
	sprintf(tPath2, "%s/mutilabel/v2.0.0/VGG16_faster_rcnn_final.caffemodel",KeyFilePath); //0.8
#else	//vgg-16
	sprintf(tPath, "%s/mutilabel/v2.0.0/faster_rcnn_test.pt",KeyFilePath);
	sprintf(tPath2, "%s/mutilabel/v2.0.0/VGG16_faster_rcnn_final_mutilabel_65class_8w.caffemodel",KeyFilePath); //0.8
#endif
	nRet = api_caffe_FasterRCNN.Init( tPath, tPath2, strLayerName.c_str(), binGPU, deviceID ); 
	if (nRet != 0)
	{
	   LOOGE<<"Fail to initialization ";
	   return TEC_INVALID_PARAM;
	}
	
	/***********************************Load dic File**********************************/
	dic.clear();
	sprintf(tPath, "%s/mutilabel/v2.0.0/Dict_mutilabel.txt",KeyFilePath);
	printf("load dic:%s\n",tPath);
	api_commen.loadWordDict(tPath,dic);
	printf( "dict:size-%d,tag:", int(dic.size()) );
	for ( i=0;i<dic.size();i++ )
	{
		printf( "%d-%s ",i,dic[i].c_str() );
	}
	printf( "\n" );

	return nRet;
}

int API_MUTI_LABEL::Predict(
	IplImage					*image, 			//[In]:image
	string						imageURL,			//[In]:image URL for CheckData
	unsigned long long			imageID,			//[In]:image ID for CheckData
	unsigned long long			childID,			//[In]:image child ID for CheckData
	float						MutiLabel_T,		//[In]:Prdict_0.8,ReSample_0.9
	vector< MutiLabelInfo >		&Res)				//[In]:Layer Name by Extract
{	
	if(!image || (image->width<16) || (image->height<16) || image->nChannels != 3 || image->depth != IPL_DEPTH_8U ) 
	{	
		LOOGE<<"input err!!";
		return TEC_INVALID_PARAM;
	}

	int i,x1,y1,x2,y2,topN = 100;
	nRet = 0;
	vector<FasterRCNNInfo>		vecLabel;
	vector<MutiLabelInfo> 		MergeRes;
	Res.clear();

	//Send Err Info
	{
		Vec4i errRect(0,0,image->width-1,image->height-1);
		MutiLabelInfo errInfo;
		errInfo.label = "other";
		errInfo.score = 1.0;
		errInfo.rect = errRect;
		errInfo.feat.clear();
		Res.push_back( errInfo );
	}

	/************************ResizeImg*****************************/
	float ratio = 1.0;
	IplImage* imgResize = api_commen.ResizeImg( image, ratio, IMAGE_SIZE );	//320:30ms,512:50ms,720:80ms
	if(!imgResize || imgResize->nChannels != 3 || imgResize->depth != IPL_DEPTH_8U) 
	{	
		LOOGE<<"Fail to ResizeImg";
		cvReleaseImage(&imgResize);imgResize = 0;
		return TEC_INVALID_PARAM;
	}
	
	/***********************************Predict**********************************/
	vecLabel.clear();
	nRet = api_caffe_FasterRCNN.Predict( imgResize, MutiLabel_T, vecLabel );
	if ( (nRet!=0) || (vecLabel.size()<1) )
	{
	   LOOGE<<"Fail to Predict";
	   cvReleaseImage(&imgResize);imgResize = 0;
	   return TEC_INVALID_PARAM;
	}

	/************************Merge*****************************/
	MergeRes.clear();
	nRet = Merge( imgResize, MutiLabel_T, vecLabel, MergeRes );
	if ( (nRet!=0) || (MergeRes.size()<1) )
	{
		LOOGE<<"[Merge Err!!]";
		cvReleaseImage(&imgResize);imgResize = 0;
		return TEC_INVALID_PARAM;
	}

	/************************Merge Res*****************************/
	Res.clear();
	topN = (MergeRes.size()>topN)?topN:MergeRes.size();
	if ( ratio == 1.0 )
	{
		Res.assign( MergeRes.begin(), MergeRes.begin()+topN );
	}
	else
	{
		for(i=0;i<topN;i++)
		{
			x1 = int(MergeRes[i].rect[0]*1.0/ratio + 0.5);
			y1 = int(MergeRes[i].rect[1]*1.0/ratio + 0.5);
			x2 = int(MergeRes[i].rect[2]*1.0/ratio + 0.5);
			y2 = int(MergeRes[i].rect[3]*1.0/ratio + 0.5);
			Vec4i ratioRect(x1,y1,x2,y2);
			MutiLabelInfo ratioInfo;
			ratioInfo.label = MergeRes[i].label;
			ratioInfo.score = MergeRes[i].score;
			ratioInfo.rect = ratioRect;
			std::copy(MergeRes[i].feat.begin(),MergeRes[i].feat.end(), std::back_inserter(ratioInfo.feat)); 
			Res.push_back( ratioInfo );
		}
	}

	cvReleaseImage(&imgResize);imgResize = 0;

	/************************Write Tmp Data*****************************/
	char tPath[4096];
	sprintf( tPath, "%lld,%lld,%s,%d", imageID, childID, imageURL.c_str(), Res.size() );
	for(i=0;i<Res.size();i++)
	{
		sprintf( tPath, "%s,%s_%.2f_%d_%d_%d_%d", tPath, Res[i].label.c_str(), Res[i].score, 
			Res[i].rect[0], Res[i].rect[1], Res[i].rect[2], Res[i].rect[3] );
	}
	LOOGI_(enum_module_in_logo)<<tPath;

	return nRet;
}

int API_MUTI_LABEL::Merge(
		IplImage						*image,			//[In]:image
		float							MutiLabel_T,	//[In]:Prdict_0.8,ReSample_0.9
		vector< FasterRCNNInfo >		inImgLabel, 	//[In]:ImgDetail from inImgLabel
		vector< MutiLabelInfo >			&LabelInfo)		//[Out]:LabelInfo
{
	if ( inImgLabel.size() < 1 ) 
	{ 
		LOOGE<<"MergeLabel[err]:inImgLabel.size()<1!!";
		return TEC_INVALID_PARAM;
	}
	
	int i,k,label,BinMode,bin_sv_Filter_Face;
	float score = 0.0;
	nRet=0;
	LabelInfo.clear();

	for ( i=0;i<inImgLabel.size();i++ )
	{
		label = inImgLabel[i].label;
		score = inImgLabel[i].score;		

		if ( (label<dic.size()) && (score>=MutiLabel_T) )
		{
			MutiLabelInfo mutiLabelInfo;
			mutiLabelInfo.label = dic[label];
			mutiLabelInfo.score = score;
			mutiLabelInfo.rect = inImgLabel[i].rect;
			std::copy(inImgLabel[i].feat.begin(),inImgLabel[i].feat.end(), std::back_inserter(mutiLabelInfo.feat)); 
			
			LabelInfo.push_back( mutiLabelInfo );
		}
	}

	//Send Err Info
	if (LabelInfo.size()<1)
	{
		Vec4i errRect(0,0,image->width-1,image->height-1);
		MutiLabelInfo errInfo;
		errInfo.label = "other.other";
		errInfo.score = 0;
		errInfo.rect = errRect;
		errInfo.feat.clear();
		LabelInfo.push_back( errInfo );
	}
	else
	{
		std::sort(LabelInfo.begin(),LabelInfo.end(),Sort_Info);
	}
	
	return TOK;
}

/***********************************Release**********************************/
void API_MUTI_LABEL::Release()
{
	/***********************************net Model**********************************/
	api_caffe_FasterRCNN.Release();
}


