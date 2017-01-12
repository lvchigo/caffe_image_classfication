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
#include <algorithm>	//min/max
#include <math.h>		//atan

#include "API_commen.h"
#include "plog/Log.h"
#include "TErrorCode.h"
#include "caffe/caffe.hpp"
#include "API_faceannoation.h"

using namespace cv;
using namespace caffe;  // NOLINT(build/namespaces)
using namespace std;

/***********************************Init*************************************/
/// construct function 
API_FACE_ANNOATION::API_FACE_ANNOATION()
{
}

/// destruct function 
API_FACE_ANNOATION::~API_FACE_ANNOATION(void)
{
}

/***********************************Init*************************************/
int API_FACE_ANNOATION::Init( 
		const char* 	KeyFilePath,						//[In]:KeyFilePath
		const int 		binGPU, 							//[In]:USE GPU(1) or not(0)
		const int 		deviceID )							//[In]:GPU ID
{
	int i, device_id, nRet = 0;	
	string strLayerName = "Dense2";
	char tPath[1024] = {0};

	char tPath1[1024] = {0};	//prototxt
	char tPath2[1024] = {0};	//caffemodel
	char tPath3[1024] = {0};	//mean
	char tPath4[1024] = {0};	//std
	
	/***********************************Init**********************************/
	sprintf(tPath1, "%s/face_annoation/VanillaCNN/vanilla_deploy.prototxt",KeyFilePath);
	sprintf(tPath2, "%s/face_annoation/VanillaCNN/vanillaCNN.caffemodel",KeyFilePath);
	sprintf(tPath3, "%s/face_annoation/VanillaCNN/trainMean.png",KeyFilePath);
	sprintf(tPath4, "%s/face_annoation/VanillaCNN/trainSTD.png",KeyFilePath);

	if ( (!tPath1) || (!tPath2) || (!tPath3) || (!tPath4) )
	{
		LOOGE<<"[API_FACE_ANNOATION::Init:Path Err]";
		return TEC_INVALID_PARAM;
	}
		
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
	
	/***********************************Load Model File**********************************/
	net_dl.reset(new caffe::Net<float>(tPath1, caffe::TEST));
	net_dl->CopyTrainedLayersFrom( tPath2 );
	if ( !net_dl->has_blob( strLayerName ) ) {
		LOOGE<<"[API_FACE_ANNOATION::Init:Unknown label name in the network]";
		return TEC_INVALID_PARAM;
	}
	
	/***********************************Load Face Detect Model File**********************************/
/*	nRet = api_face_detect.Init( KeyFilePath, binGPU, deviceID ); 
	if (nRet != 0)
	{
	   LOOGE<<"[API_FACE_ANNOATION::Init: api_face_detect.Init Err]";
	   return TEC_INVALID_PARAM;
	}*/

	/***********************************Load image**********************************/
	imgMean = cvLoadImage(tPath3);
	if(!imgMean || (imgMean->width!=FACE_BLOB_WIDTH) || (imgMean->height!=FACE_BLOB_HEIGHT) || imgMean->nChannels != CHANNEL ) 
	{	
		LOOGE<<"[API_FACE_ANNOATION::Init:imgMean Err]";
	   	return TEC_INVALID_PARAM;
	}	

	imgSTD = cvLoadImage(tPath4);
	if(!imgSTD || (imgSTD->width!=FACE_BLOB_WIDTH) || (imgSTD->height!=FACE_BLOB_HEIGHT) || imgSTD->nChannels != CHANNEL ) 
	{	
		LOOGE<<"[API_FACE_ANNOATION::Init:imgSTD Err]";
	   	return TEC_INVALID_PARAM;
	}	
	
	return nRet;
}

int API_FACE_ANNOATION::ReadImageToBlob( IplImage *image, Blob<float>& image_blob)
{
	if( !image )
	{
		LOOGE<<"[API_FACE_ANNOATION::ReadImageToBlob:image Err]";
		return TEC_INVALID_PARAM;
	}
		
	int c,h,w;
	float tmp,mean,std;
	float* dto= image_blob.mutable_cpu_data();
    float* d_diff = image_blob.mutable_cpu_diff();

    for ( c = 0; c < CHANNEL; ++c) {
        for ( h = 0; h < FACE_BLOB_HEIGHT; ++h) {
            for ( w = 0; w < FACE_BLOB_WIDTH; ++w) {
				tmp = (float)((uchar*)(image->imageData + image->widthStep*h))[w*3+c];
				mean = (float)((uchar*)(imgMean->imageData + imgMean->widthStep*h))[w*3+c];
				std = (float)((uchar*)(imgSTD->imageData + imgSTD->widthStep*h))[w*3+c];
				*dto = (tmp-mean)*1.0/(std+0.00000001);
				*d_diff++ = 0;
				dto++;
            }
        }
    }
	
    return 0;
}

/***********************************Predict**********************************/
int API_FACE_ANNOATION::Predict(
	IplImage							*image, 	//[In]:image
	vector< MutiLabelInfo > 			Res_MultiLabel,		//[In]:res of MultiLabel
	vector< FaceAnnoationInfo >			&Res_Face)		//[Out]:res of face annoation 
{	
	int i,j,batch_size,dim_features,nRet=0;
	int x,y,w,h,eye_w,eye_h;
	float iter_loss = 0.0;
	Res_Face.clear();
	vector< FaceAnnoationInfo > tmpRes;

	//Send Err Info
	{
		Vec4i errRect(0,0,image->width-1,image->height-1);
		FaceAnnoationInfo errInfo;
		errInfo.label = "other.other";
		errInfo.score = 1.0;
		errInfo.rect = errRect;
		errInfo.angle = 0.0;
		for(i=0;i<10;i++)
			errInfo.annoation[i] = 0;
		Res_Face.push_back( errInfo );
	}

	/*****************************Predict:face detect*****************************/
/*	vector< FaceDetectInfo > Res_FaceDetect;
	nRet = api_face_detect.Predict( image, Res_FaceDetect );
	if ( (nRet!=0) || (Res_FaceDetect.size()<1) )
	{
		LOOGE<<"[Predict:face detect Err]";
		return TEC_BAD_STATE;
	}*/
		
	vector< MutiLabelInfo > Res_FaceDetect;
	for(i=0;i<Res_MultiLabel.size();i++) 
	{
		if (Res_MultiLabel[i].label == "person.face" )
		{
			Res_FaceDetect.push_back( Res_MultiLabel[i] );
		}
	}
	if (Res_FaceDetect.size()<1)
	{
		LOOGE<<"[Predict: no face detect!!]";
		return TOK;
	}

	/*****************************Add_Face_Rect*****************************/
/*	vector< FaceDetectInfo > AddFaceInfo;
	nRet = Add_Face_Rect( image, Res_FaceDetect, AddFaceInfo );
	if ( (nRet!=0) || (AddFaceInfo.size()<1) )
	{
		LOOGE<<"[Add_Face_Rect Err]";
		return TEC_BAD_STATE;
	}*/

	/*****************************Predict*****************************/
	tmpRes.clear();
	for (i=0;i<Res_FaceDetect.size();i++)
	{
		x = Res_FaceDetect[i].rect[0];
		y = Res_FaceDetect[i].rect[1];
		w = Res_FaceDetect[i].rect[2]-Res_FaceDetect[i].rect[0];
		h = Res_FaceDetect[i].rect[3]-Res_FaceDetect[i].rect[1];
		if ( (x<0)||(y<0)||(w<0)||(h<0)||(x>image->width)||(y>image->height)||(w>image->width)||(h>image->height) )
			continue;
		
		cvSetImageROI( image,cvRect(x,y,w,h) );		
		IplImage* MutiROI = cvCreateImage(cvGetSize(image),image->depth,image->nChannels);
		cvCopy( image, MutiROI, NULL );
		cvResetImageROI(image);	
		
		/*****************************Resize Img*****************************/
		IplImage *MutiROIResize = cvCreateImage(cvSize(FACE_BLOB_WIDTH, FACE_BLOB_HEIGHT), MutiROI->depth, MutiROI->nChannels);
		cvResize( MutiROI, MutiROIResize );

		/*****************************load source*****************************/
		Blob<float> image_blob( 1, CHANNEL, FACE_BLOB_WIDTH, FACE_BLOB_HEIGHT ); // input 
		if ( ReadImageToBlob( MutiROIResize, image_blob  ) )
		{
			LOOGE<<"[API_FACE_ANNOATION::Predict:ReadImageToBlob Err!!]";
			if (MutiROI) {cvReleaseImage(&MutiROI);MutiROI = 0;}
			if (MutiROIResize) {cvReleaseImage(&MutiROIResize);MutiROIResize = 0;}
			return TEC_BAD_STATE;
		}
		
		/*****************************Data Change*****************************/	
		vector<Blob<float>*> input_blobs;	
		input_blobs = net_dl->input_blobs();
		for ( j = 0; j < input_blobs.size(); ++j) {
			caffe_copy(input_blobs[j]->count(), image_blob.mutable_cpu_data(),input_blobs[j]->mutable_cpu_data());
		}

		/*****************************Forward*****************************/
		vector<Blob<float>*> output_blobs;	
		output_blobs = net_dl->Forward(input_blobs, &iter_loss);	
		
		/*****************************get Feat*****************************/
		caffe::shared_ptr<caffe::Blob<float> > feature_blob;	
		feature_blob = net_dl->blob_by_name("Dense2");
		batch_size = feature_blob->num();
		if( batch_size < 1 )
		{
			LOOGE<<"[API_FACE_ANNOATION::Predict:batch_size err!!]";
			if (MutiROI) {cvReleaseImage(&MutiROI);MutiROI = 0;}
			if (MutiROIResize) {cvReleaseImage(&MutiROIResize);MutiROIResize = 0;}
			return TEC_BAD_STATE;
		}
		
		dim_features = feature_blob->count() / batch_size;
		//printf("batch_size = %d, dim_features = %d!!\n", batch_size, dim_features);

		/*****************************get Orignal Feat && Normalize*****************************/
		const float* feature_blob_data;
		for (j = 0; j < batch_size; ++j) {
			FaceAnnoationInfo faceAnnoationInfo;
			faceAnnoationInfo.label = Res_FaceDetect[i].label;
			faceAnnoationInfo.score = Res_FaceDetect[i].score;
			faceAnnoationInfo.rect = Res_FaceDetect[i].rect;

			//annoation
			feature_blob_data = feature_blob->cpu_data() + feature_blob->offset(j);
			for (j = 0; j < dim_features; ++j) {
				if ( j%2 == 1 )	//y
			  		faceAnnoationInfo.annoation[j] = y+int((feature_blob_data[j]+0.5)*h+0.5);
				else			//x
					faceAnnoationInfo.annoation[j] = x+int((feature_blob_data[j]+0.5)*w+0.5);
			}

			//angle
			eye_w = faceAnnoationInfo.annoation[2]-faceAnnoationInfo.annoation[0];	//eye-w
			eye_h = faceAnnoationInfo.annoation[1]-faceAnnoationInfo.annoation[3];	//eye-h
			faceAnnoationInfo.angle = atan(eye_h*1.0/eye_w)*180.0/CV_PI;

			tmpRes.push_back( faceAnnoationInfo );
		}

		if (MutiROI) {cvReleaseImage(&MutiROI);MutiROI = 0;}
		if (MutiROIResize) {cvReleaseImage(&MutiROIResize);MutiROIResize = 0;}
	}

	/************************Merge Res_Face*****************************/
	if ( tmpRes.size()>0 )
	{
		Res_Face.clear();
		Res_Face.assign( tmpRes.begin(), tmpRes.end() );
	}
	else if ( ( tmpRes.size()==0 ) && ( Res_FaceDetect.size()>0 ) )
	{
		Res_Face.clear();
		for(i=0;i<Res_FaceDetect.size();i++)
		{
			FaceAnnoationInfo errInfo;
			errInfo.label = Res_FaceDetect[i].label;
			errInfo.score = Res_FaceDetect[i].score;
			errInfo.rect = Res_FaceDetect[i].rect;
			errInfo.angle = 0.0;
			for(i=0;i<10;i++)
				errInfo.annoation[i] = 0;
			Res_Face.push_back( errInfo );
		}
	}
		
	return TOK;
}

/***********************************Add_Face_Rect**********************************/
/*int API_FACE_ANNOATION::Add_Face_Rect(
		IplImage					*image,
		vector< FaceDetectInfo > 	inputFaceInfo,
		vector< FaceDetectInfo > 	&AddFaceInfo)
{
	if ( inputFaceInfo.size() < 1 ) 
	{ 
		LOOGE<<"[API_FACE_ANNOATION::Add_Face_Rect:inputFaceInfo.size()<1!!]";
		return TEC_INVALID_PARAM;
	}

	int i,x1,y1,x2,y2,tmpx,tmpy,tmpw,tmph,nRet=0;
	
	//get face_add
	AddFaceInfo.clear();
	for(i=0;i<inputFaceInfo.size();i++)
	{
		tmpx = inputFaceInfo[i].rect[0];
		tmpy = inputFaceInfo[i].rect[1];
		tmpw = inputFaceInfo[i].rect[2]-inputFaceInfo[i].rect[0];	//rect w
		tmph = inputFaceInfo[i].rect[3]-inputFaceInfo[i].rect[1];	//rect h
		if ( (tmpx<0) || (tmpy<0) || (tmpw<1) || (tmph<1) )
			continue;
		
		//get face_add
		x1 = min( max( int(tmpx-tmpw*0.05+0.5), 0), (image->width-1) );
		y1 = min( max( int(tmpy-tmph*0.05+0.5), 0), (image->height-1) );
		x2 = min( max( int(tmpx+tmpw*1.05+0.5), 0), (image->width-1) );
		y2 = min( max( int(tmpy+tmph*1.05+0.5), 0), (image->height-1) );
		Vec4i addVect(x1,y1,x2,y2);
		FaceDetectInfo faceDetectInfo;
		faceDetectInfo.label = inputFaceInfo[i].label;
		faceDetectInfo.score= inputFaceInfo[i].score;
		faceDetectInfo.rect= addVect;
		AddFaceInfo.push_back( faceDetectInfo );
	}

	return nRet;
}
*/
/***********************************Get_Rotate_Image:Rotate with Center**********************************/
IplImage* API_FACE_ANNOATION::Get_Rotate_ROI( IplImage *image, FaceAnnoationInfo faceAnnoation )
{
	int i,j,batch_size,dim_features,nRet=0;
	int x,y,w,h,img_w,img_h;
	float min_x,min_y,max_x,max_y;

	img_w = image->width;
	img_h = image->height;
	x = faceAnnoation.rect[0];
	y = faceAnnoation.rect[1];
	w = faceAnnoation.rect[2]-faceAnnoation.rect[0];
	h = faceAnnoation.rect[3]-faceAnnoation.rect[1];
	if ( (x<0)||(y<0)||(w<0)||(h<0)||(x>image->width)||(y>image->height)||(w>image->width)||(h>image->height) )
		return NULL;

	CvPoint2D32f point[4];
	for ( i=0; i<4; i++)
	{
	  point[i].x = 0;
	  point[i].y = 0;
	}
	BoxPoints(faceAnnoation, point);

	for ( i=0; i<4; i++)
	{
		if ( i==0 )
		{
			min_x = point[i].x;
			max_x = point[i].x;
			min_y = point[i].y;
			max_y = point[i].y;
		}
		else
		{
			min_x = (point[i].x<min_x)?point[i].x:min_x;
			max_x = (point[i].x>max_x)?point[i].x:max_x;
			min_y = (point[i].y<min_y)?point[i].y:min_y;
			max_y = (point[i].y>max_y)?point[i].y:max_y;
		}
	}
	min_x = std::min( std::max( int(min_x+0.5), 0), (img_w-1) );
	min_y = std::min( std::max( int(min_y+0.5), 0), (img_h-1) );
	max_x = std::min( std::max( int(max_x+0.5), 0), (img_w-1) );
	max_y = std::min( std::max( int(max_y+0.5), 0), (img_h-1) );

	//Get Rotate ROI Point
	CvPoint2D32f roi_center;
	roi_center.x = x + w*0.5 - min_x;
	roi_center.y = y + h*0.5 - min_y;

	cvSetImageROI( image,cvRect(int(min_x),int(min_y),int(max_x-min_x),int(max_y-min_y)) );	
	IplImage* MutiROI = cvCreateImage(cvGetSize(image),image->depth,image->nChannels);
	cvCopy( image, MutiROI, NULL );
	cvResetImageROI(image);	

	IplImage *img_rotate = cvCreateImage(cvSize(int(max_x-min_x), int(max_y-min_y)), MutiROI->depth, MutiROI->nChannels);
	rotateImage( MutiROI, img_rotate, roi_center, faceAnnoation.angle*(-1.0) );
	if (MutiROI) {cvReleaseImage(&MutiROI);MutiROI = 0;}

	//
	CvPoint2D32f roi_point[2];
	roi_point[0].x = std::min( std::max( int(roi_center.x-w*0.5+0.5), 0), img_rotate->width );
	roi_point[0].y = std::min( std::max( int(roi_center.y-h*0.5+0.5), 0), img_rotate->height );
	roi_point[1].x = std::min( std::max( int(roi_center.x+w*0.5+0.5), 0), img_rotate->width );
	roi_point[1].y = std::min( std::max( int(roi_center.y+h*0.5+0.5), 0), img_rotate->height );

	cvSetImageROI( img_rotate,cvRect(int(roi_point[0].x),int(roi_point[0].y),
					int(roi_point[1].x-roi_point[0].x),int(roi_point[1].y-roi_point[0].y)) );	
	IplImage* img_rotate_roi = cvCreateImage(cvGetSize(img_rotate),img_rotate->depth,img_rotate->nChannels);
	cvCopy( img_rotate, img_rotate_roi, NULL );
	cvResetImageROI(img_rotate);	
	if (img_rotate) {cvReleaseImage(&img_rotate);img_rotate = 0;}

	return img_rotate_roi;
}

void API_FACE_ANNOATION::BoxPoints( FaceAnnoationInfo faceAnnoation, CvPoint2D32f pt[4] )
{
	int width, height;
	double angle;
	CvPoint center;
	
	width = faceAnnoation.rect[2]-faceAnnoation.rect[0];
	height = faceAnnoation.rect[3]-faceAnnoation.rect[1];
	center.x = int( faceAnnoation.rect[0] + width*0.5 + 0.5 );
	center.y = int( faceAnnoation.rect[1] + height*0.5 + 0.5 );
	angle = faceAnnoation.angle;
	cvBoxPoints( center, width, height, angle, pt );
}

void API_FACE_ANNOATION::cvBoxPoints( CvPoint center, int width, int height, float angle, CvPoint2D32f pt[4] )
{
	float a,b;

	double angle_rotate = angle*CV_PI/180.0;
	a = (float)cos(angle_rotate)*0.5f;
	b = (float)sin(angle_rotate)*0.5f;

	pt[0].x = center.x - a*width - b*height;
	pt[0].y = center.y + b*width - a*height;
	pt[1].x = center.x + a*width - b*height;
	pt[1].y = center.y - b*width - a*height;
	pt[2].x = 2*center.x - pt[0].x;
	pt[2].y = 2*center.y - pt[0].y;
	pt[3].x = 2*center.x - pt[1].x;
	pt[3].y = 2*center.y - pt[1].y;
}

void API_FACE_ANNOATION::Draw_Rotate_Box( IplImage* img, FaceAnnoationInfo faceAnnoation, Scalar color )
{
	CvPoint2D32f point[4];
	int i;
	
	for ( i=0; i<4; i++)
	{
	  point[i].x = 0;
	  point[i].y = 0;
	}
	BoxPoints(faceAnnoation, point);
	
	CvPoint pt[4];
	for ( i=0; i<4; i++)
	{
		pt[i].x = (int)(point[i].x+0.5);
		pt[i].y = (int)(point[i].y+0.5);
	}
	
	cvLine( img, pt[0], pt[1], color, 2, 8, 0 );
	cvLine( img, pt[1], pt[2], color, 2, 8, 0 );
	cvLine( img, pt[2], pt[3], color, 2, 8, 0 );
	cvLine( img, pt[3], pt[0], color, 2, 8, 0 );
}

void API_FACE_ANNOATION::rotateImage(IplImage* img, IplImage *img_rotate, CvPoint2D32f center, float degree)  
{  
    //CvPoint2D32f center;    
    //center.x = float(img->width*0.5+0.5);  
    //center.y = float(img->height*0.5+0.5);  
    
    float m[6];              
    CvMat M = cvMat( 2, 3, CV_32F, m );  
    cv2DRotationMatrix( center, degree,1, &M);  
    cvWarpAffine(img,img_rotate, &M,CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS,cvScalarAll(0) );  
}

/***********************************Release**********************************/
void API_FACE_ANNOATION::Release()
{
	//api_face_detect.Release();
	
	if (net_dl) {
    	net_dl.reset();
  	}

	if (imgMean) {cvReleaseImage(&imgMean);imgMean = 0;}
	if (imgSTD) {cvReleaseImage(&imgSTD);imgSTD = 0;}
}


