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

#include "plog/Log.h"
#include "API_caffe_test/API_caffe.h"

using namespace cv;
using namespace caffe;  // NOLINT(build/namespaces)
using namespace std;

static bool SortComp(const NmsInfo& input1, const NmsInfo& input2)
{
    return (input1.score > input2.score);
}

/***********************************Init*************************************/
/// construct function 
API_CAFFE::API_CAFFE()
{
}

/// destruct function 
API_CAFFE::~API_CAFFE(void)
{
}

/***********************************Init*************************************/
int API_CAFFE::Init( 
		const char* DL_DeployFile,							//[In]:DL_DeployFile
		const char* DL_ModelFile,							//[In]:DL_ModelFile
		const char* layerName,								//[In]:layerName:"fc7"
		const int binGPU,									//[In]:USE GPU(1) or not(0)
		const int deviceID )								//[In]:GPU ID
{
	int i, device_id;
	string strLayerName = layerName;
	nRet = 0;

	if ( (!DL_DeployFile) || (!DL_ModelFile) )
		return TEC_INVALID_PARAM;
		
	/***********************************Init GPU**********************************/
	if ( 1 == binGPU ) 
	{
		if ( deviceID != 0 ) 
			device_id = deviceID;
		else
			device_id = 0;
		Caffe::SetDevice(device_id);
		Caffe::set_mode(Caffe::GPU);
		cout << "Using GPU:" << device_id << "!!" << endl;
	} 
	else 
	{
		Caffe::set_mode(Caffe::CPU);
		cout << "Using CPU!!" << endl;
	}

	/***********************************Init**********************************/
	net_dl.reset(new caffe::Net<float>(DL_DeployFile, caffe::TEST));
	net_dl->CopyTrainedLayersFrom( DL_ModelFile );			// load model
	if ( !net_dl->has_blob( strLayerName ) ) {
		LOOGE<<"[Unknown label name:%s in the network!!]";
		return TEC_INVALID_PARAM;
	}

	return nRet;
}

/***********************************Predict**********************************/
int API_CAFFE::Predict(
	IplImage					*image, 			//[In]:image
	const float					Min_T,				//[In]:Min_T
	vector<Info_Label> 			&vecResInfo)		//[Out]:Res
{
	if(!image || (image->width<16) || (image->height<16) || image->nChannels != 3 || image->depth != IPL_DEPTH_8U ) 
	{	
		LOOGE<<"[Predict]:input err!!";
		return TEC_INVALID_PARAM;
	}

	//init
	float tmpScore;
	float MaxLabel_THRESH = 0.5;
    float NMS_THRESH = 0.3;			//[0.2,0.3,0.4]
	float CONF_THRESH = Min_T;		//0.8
	float iter_loss = 0.0;

	float im_info[4] = {0};
    int i,j,k,w,h,height,width,DataSize,tmp;
	nRet = 0;

	vector<float> inFeat;

	vector< NmsInfo > vecNMSInfo;
	vector< RectInfo_MULTILABEL > vecRectInfo;
	vector<FasterRCNNInfo_MULTILABEL> vecRectRegressionInfo;
	vector<FasterRCNNInfo_MULTILABEL> vecBBoxNMSInput;
	vector<FasterRCNNInfo_MULTILABEL> vecBBoxNMSOutput;
	vecResInfo.clear();

	/*****************************Change Data****************************/
	//printf("API_CAFFE_FasterRCNN_MULTILABEL::Predict Change Data\n");
	height = image->height;
    width = image->width;
	DataSize = height*width*3;
	float* data_buf = new float[DataSize];
    for ( h = 0; h < height; ++h) {
        for ( w = 0; w < width; ++w) {
			data_buf[(         h)*width+w] = (float)((uchar*)(image->imageData + image->widthStep*h))[w*3  ]-float(102.9801);
            data_buf[(1*height+h)*width+w] = (float)((uchar*)(image->imageData + image->widthStep*h))[w*3+1]-float(115.9465);
            data_buf[(2*height+h)*width+w] = (float)((uchar*)(image->imageData + image->widthStep*h))[w*3+2]-float(122.7717);
        }
    }
    im_info[0] = height;
    im_info[1] = width;
    im_info[2] = 1.0;	//scale-height
    im_info[3] = 1.0;	//scale-width
    //printf("im_info:%.2f,%.2f\n",im_info[0],im_info[1]);

	std::vector<Blob<float>*> input_blobs; 
	input_blobs = net_dl->input_blobs();
	//printf("input_blobs.size:%d\n",input_blobs.size());
	if (input_blobs.size()<1 )
	{
		printf("input_blobs err!!\n");
		if(data_buf != NULL){ delete[] data_buf; data_buf = NULL;}
		return TEC_INVALID_PARAM;
	}

	//copy data
	//printf("API_CAFFE_FasterRCNN_MULTILABEL::Predict copy data\n");
	if ( input_blobs[0]->count() != DataSize )
	{
		input_blobs[0]->Reshape(1, 3, height, width);
	}
	caffe_copy(input_blobs[0]->count(), data_buf, input_blobs[0]->mutable_cpu_data());
	caffe_copy(input_blobs[1]->count(), im_info, input_blobs[1]->mutable_cpu_data());

	/*****************************Forward Model****************************/
	//printf("API_CAFFE_FasterRCNN_MULTILABEL::Predict Forward\n");
	std::vector<Blob<float>*> output_blobs;
	try
	{
		output_blobs = net_dl->Forward(input_blobs,&iter_loss);
		if (output_blobs.size()<1)
		{
			LOOGE<<"[net_dl->ForwardPrefilled() err~~!!]";
			throw 1;
		}
	}
	catch(...)
	{
		if(data_buf != NULL){ delete[] data_buf; data_buf = NULL;}
		return TEC_BAD_STATE;
	}

	//delete
	if(data_buf != NULL){ delete[] data_buf; data_buf = NULL;}

	/*****************************Get Output Data*****************************/
	//printf("API_CAFFE_FasterRCNN_MULTILABEL::Predict Get Output Data\n");
	caffe::shared_ptr<caffe::Blob<float> > cls_blob;
	cls_blob = net_dl->blob_by_name("cls_prob");

	caffe::shared_ptr<caffe::Blob<float> > bbox_blob;
	bbox_blob = net_dl->blob_by_name("bbox_pred");

	caffe::shared_ptr<caffe::Blob<float> > roi_blob;
	roi_blob = net_dl->blob_by_name("rois");

	int cls_batch_size = cls_blob->num();
	int bbox_batch_size = bbox_blob->num();
	int roi_batch_size = roi_blob->num();
	if( ( cls_batch_size<1 )||( bbox_batch_size<1 )||( roi_batch_size<1 )||
		( cls_batch_size!=bbox_batch_size )||( cls_batch_size!=roi_batch_size ) )
	{
		printf("[Predict]:cls_batch_size = %d, bbox_batch_size = %d, roi_batch_size = %d, err!!\n", 
				cls_batch_size, bbox_batch_size, roi_batch_size );	
		return TEC_BAD_STATE;
	}
	
	int cls_dim_features = cls_blob->count() / cls_batch_size;
	int bbox_dim_features = bbox_blob->count() / bbox_batch_size;
	int roi_dim_features = roi_blob->count() / roi_batch_size;
	if( ( cls_dim_features<1 )||( bbox_dim_features<1 ) )
	{
		printf("[Predict]:cls_dim_features = %d, bbox_dim_features = %d, roi_dim_features = %d, err!!\n", 
				cls_dim_features, bbox_dim_features,roi_dim_features );	
		return TEC_BAD_STATE;
	}

	//printf("[Predict]:cls_batch_size = %d, cls_dim_features = %d!!\n", cls_batch_size, cls_dim_features);
	//printf("[Predict]:bbox_batch_size = %d, bbox_dim_features = %d!!\n", bbox_batch_size, bbox_dim_features);
	//printf("[Predict]:roi_batch_size = %d, roi_dim_features = %d!!\n", roi_batch_size, roi_dim_features);

	/*****************************Get Label && Score && Rect && Bias*****************************/
	const float* cls_blob_data;
	const float* bbox_blob_data;
	const float* roi_blob_data;
	
	vecRectInfo.clear();
	for (i = 0; i < cls_batch_size; ++i) {
		cls_blob_data = cls_blob->cpu_data() + cls_blob->offset(i);
		bbox_blob_data = bbox_blob->cpu_data() + bbox_blob->offset(i);
		roi_blob_data = roi_blob->cpu_data() + roi_blob->offset(i);

		vecNMSInfo.clear();
		tmpScore = 0;
		for (j = 1; j < cls_dim_features; ++j) {
		  	tmpScore = cls_blob_data[j];

			//Get Max Label
			if ( tmpScore<MaxLabel_THRESH )
				continue;

			//add data
			NmsInfo nmsInfo;
			nmsInfo.label = j-1;	//remove background
		  	nmsInfo.score = tmpScore;
			for(k=0;k<4;++k)
				nmsInfo.bbox[k] = bbox_blob_data[4+k];	//cfg.TEST.AGONISTIC
				//nmsInfo.bbox[k] = bbox_blob_data[k];	//cfg.TEST.normal
				//nmsInfo.bbox[k] = bbox_blob_data[k+j*4];
			vecNMSInfo.push_back( nmsInfo );
		}
		
		if (vecNMSInfo.size()<1)
			continue;

		//sort
		sort(vecNMSInfo.begin(), vecNMSInfo.end(),SortComp);	

		//T<CONF_THRESH
		if (vecNMSInfo[0].score<CONF_THRESH)
			continue;

		//add data
		RectInfo_MULTILABEL rectInfo;
		rectInfo.feat.clear();
		rectInfo.label = vecNMSInfo[0].label;
	  	rectInfo.score = vecNMSInfo[0].score;
		for(k=0;k<4;++k)
			rectInfo.bbox[k] = vecNMSInfo[0].bbox[k];
		for(k=0;k<4;++k)
			rectInfo.rect[k] = roi_blob_data[1+k];

		//if (i<3)
		//	printf("roi_feat:%.2f_%.2f_%.2f!!\n",rectInfo.feat[0],rectInfo.feat[1],rectInfo.feat[2]);

		//push nms data
		vecRectInfo.push_back( rectInfo );
	}

	//judge
	if (vecRectInfo.size()<1)
	{
		vecResInfo.clear();
		Vec4i rect_err(0,0,image->width-1,image->height-1);
		FasterRCNNInfo_MULTILABEL info;
		info.label = 51;	//other,11-6,12-7
		info.score = 0;
		info.rect = rect_err;
		info.feat.clear();
		vecResInfo.push_back( info );
		
		//LOOGE<<"[Predict]:vecRectInfo.size()<1!!";
		return 0;
	}

	/*****************************BBox Regression*****************************/
	vecRectRegressionInfo.clear();
	nRet = bbox_regression( vecRectInfo, image->width, image->height, vecRectRegressionInfo );
	if ( (nRet!=0) || (vecRectRegressionInfo.size()<1) )
	{
		LOOGE<<"[Predict]:bbox_regression Err!!";
		return TEC_BAD_STATE;
	}

	/*****************************BBox NMS*****************************/
	vector<int> suppress;
	vecResInfo.clear();
	for(i=0;i<cls_dim_features-1;i++)	//class-label
	{
		//input data
		vecBBoxNMSInput.clear();
		suppress.clear();
		for(j=0;j<vecRectRegressionInfo.size();j++)
		{
			if( vecRectRegressionInfo[j].label == i )
			{
				vecBBoxNMSInput.push_back( vecRectRegressionInfo[j] );
				suppress.push_back(j);	//index
			}
		}
		
		if ( vecBBoxNMSInput.size()<1 )
			continue;
		else if ( vecBBoxNMSInput.size()==1 )
		{
			//send data
			vecResInfo.push_back( vecBBoxNMSInput[0] );
		}
		else
		{
			//bbox_NMS
			vecBBoxNMSOutput.clear();
			nRet = bbox_NMS( vecBBoxNMSInput, vecBBoxNMSOutput, NMS_THRESH );
			if ( (nRet!=0) || (vecBBoxNMSOutput.size()<1) )
			{
				LOOGE<<"[Predict]:bbox_NMS Err!!";
				return TEC_BAD_STATE;
			}

			//send data
			for (j=0;j<vecBBoxNMSOutput.size();j++)
				vecResInfo.push_back( vecBBoxNMSOutput[j] );
		}

		//remove rect
		for (j=suppress.size()-1;j>=0;j--)
		{
			vecRectRegressionInfo.erase(vecRectRegressionInfo.begin()+suppress[j]);
		}
	}

	return nRet;
}

/***********************************Release**********************************/
void API_CAFFE::Release()
{
	if (net_dl) {
    	net_dl.reset();
  	}
}


