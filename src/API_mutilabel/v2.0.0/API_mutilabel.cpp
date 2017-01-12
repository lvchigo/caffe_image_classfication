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

#include "plog/Log.h"
#include "API_mutilabel/v2.0.0/API_mutilabel.h"
#include "API_commen/TErrorCode.h"

using namespace cv;
using namespace std;

//#define MutiLabel_T 0.8
//#define FRCN 1

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

	string strLayerName;
	nRet = 0;
	
	/***********************************Init**********************************/
#ifdef FRCN	//VGG-16
	strLayerName = "fc7";
	sprintf(tPath, "%s/mutilabel/v2.0.0/faster_rcnn_test.pt",KeyFilePath);
	sprintf(tPath2, "%s/mutilabel/v2.0.0/VGG16_faster_rcnn_final_mutilabel_65class_8w.caffemodel",KeyFilePath); //0.8
#else	//ResNet-101
	strLayerName = "cls_prob";
	sprintf(tPath, "%s/mutilabel/v2.0.0/test_agonistic_c.prototxt",KeyFilePath);
	sprintf(tPath2, "%s/mutilabel/v2.0.0/resnet101_rfcn_ohem_iter_110000.caffemodel",KeyFilePath); //0.8
#endif
	nRet = api_caffe_FasterRCNN_multilabel.Init( tPath, tPath2, strLayerName.c_str(), binGPU, deviceID ); 
	if (nRet != 0)
	{
	   LOOGE<<"Fail to initialization ";
	   return TEC_INVALID_PARAM;
	}

	/***********************************Init*************************************/
	nRet = Face_Init( KeyFilePath, binGPU, deviceID ); 
	if (nRet != 0)
	{
	   cout<<"Fail to initialization "<<endl;
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

/***********************************Init*************************************/
int API_MUTI_LABEL::Face_Init( 
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

int API_MUTI_LABEL::Face_ReadImageToBlob( IplImage *image, Blob<float>& image_blob)
{
	if( !image )
	{
		LOOGE<<"[API_FACE_ANNOATION::Face_ReadImageToBlob:image Err]";
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
	vector<FasterRCNNInfo_MULTILABEL>		vecLabel;
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
	nRet = api_caffe_FasterRCNN_multilabel.Predict( imgResize, MutiLabel_T, vecLabel );
	if ( (nRet!=0) || (vecLabel.size()<1) )
	{
	   LOOGE<<"Fail to Predict";
	   cvReleaseImage(&imgResize);imgResize = 0;
	   return TEC_INVALID_PARAM;
	}

	/************************MutiLabel_Merge*****************************/
	MergeRes.clear();
	nRet = MutiLabel_Merge( imgResize, MutiLabel_T, vecLabel, MergeRes );
	if ( (nRet!=0) || (MergeRes.size()<1) )
	{
		LOOGE<<"[MutiLabel_Merge Err!!]";
		cvReleaseImage(&imgResize);imgResize = 0;
		return TEC_INVALID_PARAM;
	}

	/************************MutiLabel_Merge Res*****************************/
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
	sprintf( tPath, "%lld %lld %s %d", imageID, childID, imageURL.c_str(), Res.size() );
	for(i=0;i<Res.size();i++)
	{
		sprintf( tPath, "%s %s %.2f %d %d %d %d", tPath, Res[i].label.c_str(), Res[i].score, 
			Res[i].rect[0], Res[i].rect[1], Res[i].rect[2], Res[i].rect[3] );
	}
	LOOGI_(enum_module_in_logo)<<tPath;

	return nRet;
}

int API_MUTI_LABEL::MutiLabel_Merge(
		IplImage								*image,			//[In]:image
		float									MutiLabel_T,	//[In]:Prdict_0.8,ReSample_0.9
		vector< FasterRCNNInfo_MULTILABEL >		inImgLabel, 	//[In]:ImgDetail from inImgLabel
		vector< MutiLabelInfo >					&LabelInfo)		//[Out]:LabelInfo
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

/***********************************Predict**********************************/
int API_MUTI_LABEL::Face_Predict(
	IplImage							*image, 	//[In]:image
	vector< MutiLabelInfo > 			Res_MultiLabel,		//[In]:res of MultiLabel
	vector< FaceAnnoationInfo >			&Res_FaceAnnoation)		//[Out]:res of face annoation 
{	
	int i,j,batch_size,dim_features,nRet=0;
	int x,y,w,h,eye_w,eye_h;
	float iter_loss = 0.0;
	Res_FaceAnnoation.clear();
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
		Res_FaceAnnoation.push_back( errInfo );
	}

	/*****************************Predict:face detect*****************************/
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
		if ( Face_ReadImageToBlob( MutiROIResize, image_blob  ) )
		{
			LOOGE<<"[API_FACE_ANNOATION::Predict:Face_ReadImageToBlob Err!!]";
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

	/************************MutiLabel_Merge Res_FaceAnnoation*****************************/
	if ( tmpRes.size()>0 )
	{
		Res_FaceAnnoation.clear();
		Res_FaceAnnoation.assign( tmpRes.begin(), tmpRes.end() );
	}
	else if ( ( tmpRes.size()==0 ) && ( Res_FaceDetect.size()>0 ) )
	{
		Res_FaceAnnoation.clear();
		for(i=0;i<Res_FaceDetect.size();i++)
		{
			FaceAnnoationInfo errInfo;
			errInfo.label = Res_FaceDetect[i].label;
			errInfo.score = Res_FaceDetect[i].score;
			errInfo.rect = Res_FaceDetect[i].rect;
			errInfo.angle = 0.0;
			for(i=0;i<10;i++)
				errInfo.annoation[i] = 0;
			Res_FaceAnnoation.push_back( errInfo );
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
IplImage* API_MUTI_LABEL::Face_Get_Rotate_ROI( IplImage *image, FaceAnnoationInfo faceAnnoation )
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
	Face_BoxPoints(faceAnnoation, point);

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
	Face_rotateImage( MutiROI, img_rotate, roi_center, faceAnnoation.angle*(-1.0) );
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

void API_MUTI_LABEL::Face_BoxPoints( FaceAnnoationInfo faceAnnoation, CvPoint2D32f pt[4] )
{
	int width, height;
	double angle;
	CvPoint center;
	
	width = faceAnnoation.rect[2]-faceAnnoation.rect[0];
	height = faceAnnoation.rect[3]-faceAnnoation.rect[1];
	center.x = int( faceAnnoation.rect[0] + width*0.5 + 0.5 );
	center.y = int( faceAnnoation.rect[1] + height*0.5 + 0.5 );
	angle = faceAnnoation.angle;
	Face_cvBoxPoints( center, width, height, angle, pt );
}

void API_MUTI_LABEL::Face_cvBoxPoints( CvPoint center, int width, int height, float angle, CvPoint2D32f pt[4] )
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

void API_MUTI_LABEL::Face_Draw_Rotate_Box( IplImage* img, FaceAnnoationInfo faceAnnoation, Scalar color )
{
	CvPoint2D32f point[4];
	int i;
	
	for ( i=0; i<4; i++)
	{
	  point[i].x = 0;
	  point[i].y = 0;
	}
	Face_BoxPoints(faceAnnoation, point);
	
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

void API_MUTI_LABEL::Face_rotateImage(IplImage* img, IplImage *img_rotate, CvPoint2D32f center, float degree)  
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
void API_MUTI_LABEL::Release()
{
	/***********************************net Model**********************************/
	api_caffe_FasterRCNN_multilabel.Release();

	/*********************************Release*************************************/
	if (net_dl) {
    	net_dl.reset();
  	}

	if (imgMean) {cvReleaseImage(&imgMean);imgMean = 0;}
	if (imgSTD) {cvReleaseImage(&imgSTD);imgSTD = 0;}
}


