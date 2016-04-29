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

#include "caffe/caffe.hpp"
#include "API_caffe_faster_rcnn.h"
#include "plog/Log.h"

using namespace cv;
using namespace caffe;  // NOLINT(build/namespaces)
using namespace std;

struct NmsInfo
{
	int 	label;
    float 	score;
    float 	bbox[4];	//output bbox-bias
};

struct IoUInfo {
	int 	index;
	int 	label;		
	float 	value;
	float 	area;
	float 	x1;
	float 	x2;
	float 	y1;
	float 	y2;
};

static bool SortComp(const NmsInfo& input1, const NmsInfo& input2)
{
    return (input1.score > input2.score);
}

static bool Sort_FasterRCNNInfo(const FasterRCNNInfo& input1, const FasterRCNNInfo& input2)
{
    return (input1.score > input2.score);
}

static bool rindcom(const IoUInfo& t1, const IoUInfo& t2) {
	return t1.value < t2.value;
}

/***********************************Init*************************************/
/// construct function 
API_CAFFE_FasterRCNN::API_CAFFE_FasterRCNN()
{
}

/// destruct function 
API_CAFFE_FasterRCNN::~API_CAFFE_FasterRCNN(void)
{
}

/***********************************Init*************************************/
int API_CAFFE_FasterRCNN::Init( 
		const char* DL_DeployFile,							//[In]:DL_DeployFile
		const char* DL_ModelFile,							//[In]:DL_ModelFile
		const char* layerName,								//[In]:layerName:"fc7"
		const int binGPU,									//[In]:USE GPU(1) or not(0)
		const int deviceID )								//[In]:GPU ID
{
	int i, device_id, nRet = 0;	
	string strLayerName = layerName;

	if ( (!DL_DeployFile) || (!DL_ModelFile) )
		return TEC_INVALID_PARAM;
		
	/***********************************Init GPU**********************************/
	if ( 1 == binGPU ) 
	{
		if ( 1 != deviceID ) 
			device_id = 0;		
		else
			device_id = deviceID;
		Caffe::set_mode(Caffe::GPU);
		Caffe::SetDevice(device_id);
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
		printf("Unknown label name:%s in the network.\n",strLayerName.c_str());
		return TEC_INVALID_PARAM;
	}
	
	return nRet;
}

/***********************************Predict**********************************/
int API_CAFFE_FasterRCNN::Predict(
	IplImage					*image, 			//[In]:image
	const float					Min_T,				//[In]:Min_T
	vector<FasterRCNNInfo> 		&vecResInfo)		//[Out]:Res
{
	if(!image || (image->width<16) || (image->height<16) || image->nChannels != 3 || image->depth != IPL_DEPTH_8U ) 
	{	
		LOOGE<<"[Predict]:input err!!";
		return TEC_INVALID_PARAM;
	}

	//load image
    cv::Mat cv_img(image);
    if(cv_img.empty())
    {
        LOOGE<<"[Predict]:input err!!";
		return TEC_INVALID_PARAM;
    }

	//init
	float tmpScore;
	float MaxLabel_THRESH = 0.5;
    float NMS_THRESH = 0.3;			//[0.2,0.3,0.4]
	float CONF_THRESH = Min_T;		//0.8

	float im_info[3] = {0};
    int i,j,k,height,width,tmp,nRet = 0;

	vector< NmsInfo > vecNMSInfo;
	vector< RectInfo > vecRectInfo;
	vector<FasterRCNNInfo> vecRectRegressionInfo;
	vector<FasterRCNNInfo> vecBBoxNMSInput;
	vector<FasterRCNNInfo> vecBBoxNMSOutput;
	vecResInfo.clear();

	/*****************************Change Data****************************/
    height = cv_img.rows;
    width = cv_img.cols;
	float* data_buf = new float[height*width*3];
	for (int h = 0; h < cv_img.rows; ++h )
    {
        for (int w = 0; w < cv_img.cols; ++w)
        {   
            data_buf[(0*height+h)*width+w] = float(cv_img.at<cv::Vec3b>(cv::Point(w, h))[0])-float(102.9801);
            data_buf[(1*height+h)*width+w] = float(cv_img.at<cv::Vec3b>(cv::Point(w, h))[1])-float(115.9465);
            data_buf[(2*height+h)*width+w] = float(cv_img.at<cv::Vec3b>(cv::Point(w, h))[2])-float(122.7717);

        }
    }   
    im_info[0] = cv_img.rows;
    im_info[1] = cv_img.cols;
    im_info[2] = 1.0;	//scale

	/*****************************Input Data****************************/
    net_dl->blob_by_name("data")->Reshape(1, 3, height, width);
    net_dl->blob_by_name("data")->set_cpu_data(data_buf);
    net_dl->blob_by_name("im_info")->set_cpu_data(im_info);

	/*****************************Forward Model****************************/
	//std::vector<Blob<float>*> input_vec; 
	//net_dl->Forward(input_vec);
	vector<Blob<float>*> output_blobs;
	output_blobs = net_dl->ForwardPrefilled();

	/*****************************Get Output Data*****************************/
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
				cls_batch_size, bbox_batch_size, roi_batch_size);	
		if(data_buf != NULL){ delete[] data_buf; data_buf = NULL;}
		return TEC_BAD_STATE;
	}
	
	int cls_dim_features = cls_blob->count() / cls_batch_size;
	int bbox_dim_features = bbox_blob->count() / bbox_batch_size;
	int roi_dim_features = roi_blob->count() / roi_batch_size;
	if( ( cls_dim_features<1 )||( bbox_dim_features<1 )||( roi_dim_features!=5 )||
		( cls_dim_features*4!=bbox_dim_features ) )
	{
		printf("[Predict]:cls_dim_features = %d, bbox_dim_features = %d, roi_dim_features = %d, err!!\n", 
				cls_dim_features, bbox_dim_features,roi_dim_features);	
		if(data_buf != NULL){ delete[] data_buf; data_buf = NULL;}
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
		//printf("roi:%.2f_%.2f_%.2f_%.2f_%.2f!!\n",roi_blob_data[0],roi_blob_data[1],
		//	roi_blob_data[2],roi_blob_data[3],roi_blob_data[4]);

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
				nmsInfo.bbox[k] = bbox_blob_data[k+j*4];
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
		RectInfo rectInfo;
		rectInfo.label = vecNMSInfo[0].label;
	  	rectInfo.score = vecNMSInfo[0].score;
		for(k=0;k<4;++k)
			rectInfo.bbox[k] = vecNMSInfo[0].bbox[k];
		for(k=0;k<4;++k)
			rectInfo.rect[k] = roi_blob_data[1+k];

		//push nms data
		vecRectInfo.push_back( rectInfo );
	}

	//judge
	if (vecRectInfo.size()<1)
	{
		vecResInfo.clear();
		Vec4i rect_err(0,0,image->width-1,image->height-1);
		FasterRCNNInfo info;
		info.label = 7;	//other,11-6,12-7
		info.score = 1.0;
		info.rect = rect_err;
		vecResInfo.push_back( info );
		
		//LOOGE<<"[Predict]:vecRectInfo.size()<1!!";
		if(data_buf != NULL){ delete[] data_buf; data_buf = NULL;}
		return 0;
	}

	/*****************************BBox Regression*****************************/
	vecRectRegressionInfo.clear();
	nRet = bbox_regression( vecRectInfo, image->width, image->height, vecRectRegressionInfo );
	if ( (nRet!=0) || (vecRectRegressionInfo.size()<1) )
	{
		LOOGE<<"[Predict]:bbox_regression Err!!";
		if(data_buf != NULL){ delete[] data_buf; data_buf = NULL;}
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
				if(data_buf != NULL){ delete[] data_buf; data_buf = NULL;}
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

	//delete
	if(data_buf != NULL){ delete[] data_buf; data_buf = NULL;}

	return nRet;
}

//see python-faster-rcnn:bbox_transform_inv
int API_CAFFE_FasterRCNN::bbox_regression( 
	vector<RectInfo> 			vecRectInfo, 
	int 						width, 
	int 						height, 
	vector<FasterRCNNInfo> 		&vecResInfo )
{
	if( ( vecRectInfo.size()<1 ) || (width<32) || (height<32) )
    {
        LOOGE<<"[bbox_regression]:input err!!";
		return TEC_INVALID_PARAM;
    }
	
	int i,x1,x2,y1,y2,nRet = 0;
	float w, h, ctr_x, ctr_y, dx, dy, dw, dh, pred_ctr_x, pred_ctr_y, pred_w, pred_h;

	vecResInfo.clear();
	for( i=0; i< vecRectInfo.size(); i++)
    {
        w = vecRectInfo[i].rect[2] - vecRectInfo[i].rect[0] + 1.0;
        h = vecRectInfo[i].rect[3] - vecRectInfo[i].rect[1] + 1.0;
        ctr_x = vecRectInfo[i].rect[0] + 0.5 * w;
        ctr_y = vecRectInfo[i].rect[1] + 0.5 * h;

        dx = vecRectInfo[i].bbox[0];
        dy = vecRectInfo[i].bbox[1];
        dw = vecRectInfo[i].bbox[2];
        dh = vecRectInfo[i].bbox[3];
        pred_ctr_x = ctr_x + w*dx;
        pred_ctr_y = ctr_y + h*dy;
        pred_w = w * exp(dw);
        pred_h = h * exp(dh);

        x1 = int(std::max(std::min((pred_ctr_x - 0.5* pred_w), (width-1)* 1.0), 0.0));
        y1 = int(std::max(std::min((pred_ctr_y - 0.5* pred_h), (height-1)*1.0), 0.0));
        x2 = int(std::max(std::min((pred_ctr_x + 0.5* pred_w), (width-1)* 1.0), 0.0));
        y2 = int(std::max(std::min((pred_ctr_y + 0.5* pred_h), (height-1)*1.0), 0.0));

		if ( (x1>=x2) || (y1>=y2) )
			continue;
		
		Vec4i rect(x1,y1,x2,y2);
		FasterRCNNInfo fasterRCNNInfo;
		fasterRCNNInfo.label = vecRectInfo[i].label;
		fasterRCNNInfo.score= vecRectInfo[i].score;
		fasterRCNNInfo.rect = rect;

		vecResInfo.push_back(fasterRCNNInfo);
    }

	if (vecResInfo.size()<1)
	{
		LOOGE<<"[bbox_regression]:vecResInfo.size()<1!!";
		return TEC_BAD_STATE;
	}

	return nRet;
}

int API_CAFFE_FasterRCNN::bbox_NMS(vector<FasterRCNNInfo> src, vector<FasterRCNNInfo> &dst, float overlap) 
{ 
// pick out the most probable boxes
	if (src.empty())
		return TEC_INVALID_PARAM;
	
	int tmp, i, pos, last, srcnum = src.size();
	float xx1,yy1,xx2,yy2,w,h,tmpArea, o;
	vector<IoUInfo> s(srcnum);
	vector<int> pick;
	vector<int> suppress;
	
	for (i=0;i<srcnum;i++)
	{
		s[i].index=i;
		s[i].label=src[i].label;
		s[i].value=src[i].score;
		s[i].x1=src[i].rect[0];
		s[i].y1=src[i].rect[1];
		s[i].x2=src[i].rect[2];
		s[i].y2=src[i].rect[3];
		s[i].area=(src[i].rect[2]-src[i].rect[0]+1)*(src[i].rect[3]-src[i].rect[1]+1);
	}
	std::sort(s.begin(),s.end(),rindcom);

	tmp = 0;
	pick.clear();
	while (!s.empty())
	{
		suppress.clear();
		last=s.size()-1;
		pick.push_back(s[last].index);
		suppress.push_back(last);
		for (pos=0;pos<last;pos++)
		{          
			xx1=max(s[last].x1,s[pos].x1);
			yy1=max(s[last].y1,s[pos].y1);
			xx2=min(s[last].x2,s[pos].x2);
			yy2=min(s[last].y2,s[pos].y2);
			w=xx2-xx1+1;
			h=yy2-yy1+1;
			tmpArea = w*h;
			if ( (w>0)&&(h>0)&&(s[pos].area+s[last].area>tmpArea) )
			{
				//o = tmpArea*1.0/s[pos].area;	//pos.area
				o = tmpArea*1.0/(s[pos].area+s[last].area-tmpArea);	//pos.area
				if (o>overlap)
				{
					suppress.push_back(pos);
				}
			}
		}
		sort(suppress.begin(),suppress.end());

		//remove iou rect
		for (i=suppress.size()-1;i>=0;i--)
		{
			s.erase(s.begin()+suppress[i]);
		}
	}

	if (!dst.empty())
		dst.clear();

	for (i=0;i<pick.size();i++) {
		dst.push_back(src[pick[i]]);
	}

	std::sort(dst.begin(),dst.end(),Sort_FasterRCNNInfo);

	return 0;
}

/***********************************Release**********************************/
void API_CAFFE_FasterRCNN::Release()
{
	if (net_dl) {
    	net_dl.reset();
  	}
}


