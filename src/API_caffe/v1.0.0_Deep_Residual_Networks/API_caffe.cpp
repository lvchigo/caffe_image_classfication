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
#include "API_caffe.h"

using namespace cv;
using namespace caffe;  // NOLINT(build/namespaces)
using namespace std;

/***********************************Init*************************************/
/// construct function 
API_CAFFE::API_CAFFE()
{
}

/// destruct function 
API_CAFFE::~API_CAFFE(void)
{
}

static bool ImgSortComp(const pair <int, float> elem1, const pair <int, float> elem2)
{
	return (elem1.second > elem2.second);
}

static bool ImgSortComp2(const double elem1, const double elem2)
{
	return (elem1 > elem2);
}

/***********************************Init*************************************/
int API_CAFFE::Init( 
		const char* DL_DeployFile,							//[In]:DL_DeployFile
		const char* DL_ModelFile,							//[In]:DL_ModelFile
		const char* DL_Meanfile,							//[In]:DL_Meanfile
		const char* layerName,								//[In]:layerName:"fc7"
		const int binGPU,									//[In]:USE GPU(1) or not(0)
		const int deviceID )								//[In]:GPU ID

{
	int i, device_id, nRet = 0;	
	BlobProto input_mean_dl;
	string strLayerName = layerName;
	char tPath[1024] = {0};

	if ( (!DL_DeployFile) || (!DL_ModelFile) || (!DL_Meanfile) )
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
	
	/***********************************Load 71class File**********************************/
	net_dl.reset(new caffe::Net<float>(DL_DeployFile, caffe::TEST));
	
	net_dl->CopyTrainedLayersFrom( DL_ModelFile );			// load model
	if ( !net_dl->has_blob( strLayerName ) ) {
		printf("Unknown label name:%s in the network.\n",strLayerName.c_str());
		return TEC_INVALID_PARAM;
	}
	
	ReadProtoFromBinaryFile(DL_Meanfile, &input_mean_dl);			// load mean
	mean_dl.FromProto(input_mean_dl);
	
	return nRet;
}

int API_CAFFE::ReadImageToBlob( const vector<Mat_<Vec3f> > img_dl, int ModelMode, Blob<float>& image_blob)
{
	if( 0 == img_dl.size() )
		return 1;
		
	int n_id,c,h,w;
	float tmp;
	float* dto= image_blob.mutable_cpu_data();
    float* d_diff = image_blob.mutable_cpu_diff();
	float* mean;
	if ( ( 1 == ModelMode) || ( 2 == ModelMode) || ( 3 == ModelMode) )
	{
		const float* tmp_mean = mean_dl.cpu_data();
		mean = (float*)tmp_mean;
	}
	
	int w_off = int( (WIDTH - BLOB_WIDTH)*0.5 );
	int h_off = int( (HEIGHT - BLOB_HEIGHT)*0.5 );
	
	for ( n_id = 0; n_id< img_dl.size(); n_id++){	
        for ( c = 0; c < CHANNEL; ++c) {
            for ( h = 0; h < BLOB_HEIGHT; ++h) {
                for ( w = 0; w < BLOB_WIDTH; ++w) {
			// tmp = (float)((uchar*)(img_dl->imageData + img_dl->widthStep*h))[w*3+c];
			tmp = (float)( img_dl[n_id].at<Vec3f>(h,w)[c] );
			*dto = tmp - mean[(c * HEIGHT + h + h_off) * WIDTH + w + w_off];		//mean:256*256
			//*dto = tmp;
			*d_diff++ = 0;
			dto++;
                }
            }
        }
    }
	
    return 0;
}

template <typename Dtype>
bool API_CAFFE::DataMaxMin( const Dtype* d_from, int d_count, float& d_max, float& d_min )
{
  d_max = NMIN;
  d_min = NMAX;
  for (int i = 0; i < d_count; ++i)
  {
    if(d_max<d_from[i]){
      d_max = d_from[i];
    }
    if(d_min>d_from[i]){
      d_min = d_from[i];
    }
  }
  return 1;
}

void API_CAFFE::blob2image(Blob<float>& image_blob, UInt64 imgID){
    const float* dfrom= image_blob.cpu_data();
    int width = image_blob.width();
    int height = image_blob.height();
    int channels = image_blob.channels();
    cv::Mat tmp_img(width,height, CV_32FC3);
    cv::Mat tmp_gray; //(width,height, CV_8UC1);
    char temp_name[NAMELEN];

    float d_max;
    float d_min;
    DataMaxMin( dfrom, image_blob.count(), d_max, d_min );
    LOG(ERROR) << "blob min: " << d_min << "  max: " << d_max;

    for (int img_id=0; img_id<image_blob.num(); img_id++){//image_blob.nums()
        sprintf(temp_name, "%lld_%d.jpg", imgID, img_id);
        for (int c = 0; c < channels; ++c)
        {
            for(int h=0; h<height; h++){
                for(int w=0; w<width; w++){
					tmp_img.at<cv::Vec3f>(h, w)[c] = (*dfrom);
                    dfrom++;
                }
            }
            
        }

        // tmp_img.convertTo(tmp_gray, CV_8UC1);
        cv::imwrite(temp_name, tmp_img);
    }
}

float API_CAFFE::CalcL1Norm(const float* d_from, int d_count) 
{  
  double sum = 0.0;
  for (int i = 0; i < d_count; ++i) {
    sum += fabs(d_from[i]);
  }
  return sum;
}

void API_CAFFE::info(Blob<float>& image_blob)
{
  printf("blob num: %d  channel: %d  width: %d  height: %d\n", 
             image_blob.num(), image_blob.channels(), image_blob.width(), image_blob.height() );
  printf("L1 of blob data: %f\n", CalcL1Norm( image_blob.cpu_data(), image_blob.count() ) );
  printf("L1 of blob diff: %f\n", CalcL1Norm( image_blob.cpu_diff(), image_blob.count() ) );  
}

int API_CAFFE::DL_FileList2LabelFeat( int ModelMode, const vector<Mat_<Vec3f> > img_dl, UInt64 ImageID,
		const long labelNum, const char* layerName, vector< pair< int, float > > &vecLabel, vector< vector<float> > &imgFeat)		
{	
	int i,j,k,getLabelNum,batch_size,dim_features,nRet=0;
	long netLayerSize = 0;
	float iter_loss = 0.0;
	vector<float> singleImgFeat;
	string strLayerName = layerName ;
	singleImgFeat.clear();
	imgFeat.clear();
	
	vector<Mat_<Vec3f> > vecInputImg;
	vecInputImg.clear();
	int loadImageNum = 1;//1,7;	//load image num:1-all image;2-center image;3~11-3*3 block
	if ( img_dl.size() < loadImageNum )
	{
		printf( "DL_FileList2LabelFeat err:img_dl.size()=%d,<loadImageNum=%d!!\n", img_dl.size(), loadImageNum);
		return 1;
	}
	vecInputImg.assign( img_dl.begin(), img_dl.begin()+loadImageNum );
		
	//printf("DL_FileList2Label[1]:ReadImageToBlob\n");
	/*****************************load source*****************************/
	Blob<float> image_blob( vecInputImg.size(), CHANNEL, BLOB_HEIGHT, BLOB_WIDTH ); // input 
	if ( ReadImageToBlob( vecInputImg, ModelMode, image_blob  ) )
	{
		cout<< "ReadImageToBlob Err!!" << endl;
		return 1;
	}
	//printf("DL_FileList2Label[1.1]:vecInputImg.size:%ld,CHANNEL:%d,BLOB_HEIGHT:%d,BLOB_WIDTH:%d\n",
	//	vecInputImg.size(),CHANNEL, BLOB_HEIGHT, BLOB_WIDTH);
	//blob2image(image_blob,ImageID);
	//info(image_blob);
	
	/*****************************Data Change*****************************/
	//printf("DL_FileList2Label[2]:CopyLayer\n");		
	vector<Blob<float>*> input_blobs;
	if ( ( 1 == ModelMode) || ( 2 == ModelMode) || ( 3 == ModelMode) )
	{		
		input_blobs = net_dl->input_blobs();
		getLabelNum = 2;
	}
	else
	{
		cout<< "ModelMode Err!!" << endl;
		return 1;
	}
	
	//printf(" input_blobs.size():%d\n",input_blobs.size());
	for ( i = 0; i < input_blobs.size(); ++i) {
		//printf(" input_blobs[%d]->count():%d\n",i,input_blobs[i]->count());
		caffe_copy(input_blobs[i]->count(), image_blob.mutable_cpu_data(),input_blobs[i]->mutable_cpu_data());
		//blob2image(input_blobs[i][0],ImageID);
		//info(input_blobs[i][0]);
	}

	/*****************************Forward*****************************/
	//printf("DL_FileList2Label[3]:Forward\n");
	vector<Blob<float>*> output_blobs;
	if ( ( 1 == ModelMode) || ( 2 == ModelMode) || ( 3 == ModelMode)  )
	{	
		output_blobs = net_dl->Forward(input_blobs, &iter_loss);	
	}
	//cout<< "ImageID:"<<ImageID<<" net_dl->layers().size():"<< netLayerSize << endl;
	//printf("iter_loss:%.4f\n",iter_loss);
	//info(net_dl->output_blobs()[0][0]);
	
	/*****************************get label*****************************/
	//printf("DL_FileList2Label[4]:get label\n");	
	vector<float> label_blob_data;
	label_blob_data.clear();
	if ( ( 1 == ModelMode ) || ( 3 == ModelMode) )
	{
		for (j = 0; j < output_blobs.size(); ++j) {
			for (k = 0; k < output_blobs[j]->count(); ++k) {
				label_blob_data.push_back(output_blobs[j]->cpu_data()[k]);
			}
		}

		vector< pair< int,float> > vecRes;
		vecRes.clear();
		for ( i = 0; i < vecInputImg.size(); ++i ) 
		{
			//get label result
			//printf("ImageID:%ld,ModelMode:%d,Img:%d,imgSize:%d:",ImageID,ModelMode,i,vecInputImg.size());
			for ( k=0; k < labelNum; ++k ) 
			{
				if ( i == 0 )
				{
					vecRes.push_back( std::make_pair( k, label_blob_data[k+i*labelNum] ) );		//write
				}
				else
				{
					vecRes[k].second += label_blob_data[k+i*labelNum];		//get mean
				}
				//printf("%d-%.4f ",k, label_blob_data[k+i*labelNum]);
			}
			//printf("\n");
		}
		//printf("\n");
		
		for ( k = 0; k < labelNum; ++k ) 
		{
			vecRes[k].second = vecRes[k].second *1.0/ vecInputImg.size() ;	//get mean 
		}

		//sort label result
		sort(vecRes.begin(), vecRes.end(),ImgSortComp);		
		
		//push info
		for ( k = 0; k < getLabelNum; ++k)
		{
			vecLabel.push_back( std::make_pair( vecRes[k].first, vecRes[k].second ) );
			//printf( "push info:k-%d,label-%d,score-%.4f\n",vecLabel.size(),vecRes[k].first, vecRes[k].second );
		}
		
		//printf( "NOTE:vecLabel.size:%d\n",vecLabel.size() );
	}
	
	/*****************************get Feat*****************************/
	//printf("DL_FileList2Feat[4]:get Feat\n");	
	caffe::shared_ptr<caffe::Blob<float> > feature_blob;
	if ( ( 2 == ModelMode) || ( 3 == ModelMode) )
	{	
		feature_blob = net_dl->blob_by_name(strLayerName);
		batch_size = feature_blob->num();
		if( batch_size < 1 )
		{
			printf("DL_FileList2Feat[4.1]:batch_size = %d, err!!\n", batch_size);	
			return 1;
		}
		
		dim_features = feature_blob->count() / batch_size;
		//printf("DL_FileList2Feat[4.2]:batch_size = %d, dim_features = %d!!\n", batch_size, dim_features);

		/*****************************get Orignal Feat && Normalize*****************************/
		const float* feature_blob_data;
		for (i = 0; i < batch_size; ++i) {
			singleImgFeat.clear();
			
			feature_blob_data = feature_blob->cpu_data() + feature_blob->offset(i);
			for (j = 0; j < dim_features; ++j) {
			  	singleImgFeat.push_back( feature_blob_data[j] );
			}
			imgFeat.push_back( singleImgFeat );
		}
	}

	return 0;
}

/***********************************Process**********************************/
int API_CAFFE::GetLabelFeat(
	const vector<Mat_<Vec3f> > 		img_dl,				//[In]:source image(s)
	UInt64							imgID,				//[In]:source imageID
	const char* 					layerName,			//[In]:Layer Name by Extract
	const int						bExtractFeat,		//[In]:Get Label(1),Extract Feat(2),both(3)
	vector< pair< int, float > >	&label, 			//[Out]:ImgDetail
	vector< vector<float> >			&imgFeat)			//[Out]:imgFeat 
{
	if( img_dl.size() == 0 ) 
	{	
		cout<<"InputImg err!!" << endl;
		return TEC_INVALID_PARAM;
	}

	/*****************************Init*****************************/
	int nRet = 0;

	/**********************FileList2Label-5Class***************************/
	label.clear();
	imgFeat.clear();
	nRet = DL_FileList2LabelFeat( bExtractFeat, img_dl, imgID, LABEL_NUM[0], layerName,label,imgFeat );
	if ( nRet )
	{
		cout << "DL_FileList2Label Err " << endl;
		return TEC_BAD_STATE;
	}	
	
	return nRet;
}

/***********************************Process**********************************/
int API_CAFFE::GetLabelFeat_GoogleNet(
	const vector<Mat_<Vec3f> > 		img_dl,				//[In]:source image(s)
	UInt64							imgID,				//[In]:source imageID
	const int 						labelNum,			//[In]:label Num
	const char* 					layerName,			//[In]:Layer Name by Extract
	const int						bExtractFeat,		//[In]:Get Label(1),Extract Feat(2),both(3)
	vector< pair< int, float > >	&label, 			//[Out]:ImgDetail
	vector< vector<float> >			&imgFeat)			//[Out]:imgFeat 
{
	if( img_dl.size() == 0 ) 
	{	
		cout<<"InputImg err!!" << endl;
		return TEC_INVALID_PARAM;
	}

	/*****************************Init*****************************/
	int nRet = 0;

	/**********************FileList2Label-5Class***************************/
	label.clear();
	imgFeat.clear();
	nRet = DL_FileList2LabelFeat( bExtractFeat, img_dl, imgID, labelNum, layerName,label,imgFeat );
	if ( nRet )
	{
		cout << "DL_FileList2Label Err " << endl;
		return TEC_BAD_STATE;
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


