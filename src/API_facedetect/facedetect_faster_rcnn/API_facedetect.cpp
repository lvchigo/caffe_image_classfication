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
#include "API_facedetect.h"
#include "TErrorCode.h"

using namespace cv;
using namespace std;

#define RESIZE 1
#define Min_T 0.8
#define Max_T 0.95


static bool Sort_FaceDetectInfo(const FaceDetectInfo& elem1, const FaceDetectInfo& elem2)
{
    return (elem1.score > elem2.score);
}

static bool ImgSortComp(const pair< vector< int >, float > elem1, const pair< vector< int >, float > elem2)
{
	return (elem1.second > elem2.second);
}

static bool IOUSortComp(const pair<FaceDetectInfo,float> elem1, const pair<FaceDetectInfo,float> elem2)
{
	return (elem1.second > elem2.second);
}


/***********************************Init*************************************/
/// construct function 
API_FACE_DETECT::API_FACE_DETECT()
{
}

/// destruct function 
API_FACE_DETECT::~API_FACE_DETECT(void)
{
}

/***********************************Init*************************************/
int API_FACE_DETECT::Init( 
	const char* 	KeyFilePath,						//[In]:KeyFilePath
	const int		binGPU, 							//[In]:USE GPU(1) or not(0)
	const int		deviceID )	 						//[In]:GPU ID
{
	char tPath[1024] = {0};
	char tPath2[1024] = {0};
	char tPath3[1024] = {0};
	
	/***********************************Init**********************************/
	//vgg-16
	sprintf(tPath, "%s/face_detect/faster_rcnn/faster_rcnn_test_face7class.pt",KeyFilePath);
	//sprintf(tPath, "%s/face_detect/faster_rcnn/faster_rcnn_test.pt",KeyFilePath);
	//sprintf(tPath2, "%s/face_detect/faster_rcnn/VGG16_faster_rcnn_final_40wsamples_8witer_face7class_20160415.caffemodel",KeyFilePath); //0.985
	sprintf(tPath2, "%s/face_detect/faster_rcnn/VGG16_faster_rcnn_final_40wsamples_nowider_8witer_face7class_20160417.caffemodel",KeyFilePath); //0.95
	nRet = api_caffe_FasterRCNN.Init( tPath, tPath2, "fc7", binGPU, deviceID ); 
	if (nRet != 0)
	{
	   LOOGE<<"Fail to initialization ";
	   return TEC_INVALID_PARAM;
	}
	
	/***********************************Load dic File**********************************/
	dic.clear();
	sprintf(tPath, "%s/face_detect/Dict_FAST_RCNN_FACEDETECT11label.txt",KeyFilePath);
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

int API_FACE_DETECT::Predict(
	IplImage					*image, 			//[In]:image
	vector< FaceDetectInfo >	&Res)				//[In]:Layer Name by Extract
{	
	if(!image || (image->width<16) || (image->height<16) || image->nChannels != 3 || image->depth != IPL_DEPTH_8U ) 
	{	
		LOOGE<<"input err!!";
		return TEC_INVALID_PARAM;
	}

	int i,x1,y1,x2,y2,topN = 100;
	nRet = 0;
	vector<FasterRCNNInfo>		vecLabel;
	vector<FaceDetectInfo> 		MergeRes;
	Res.clear();

	//Send Err Info
	{
		Vec4i errRect(0,0,image->width-1,image->height-1);
		FaceDetectInfo errInfo;
		errInfo.label = "other";
		errInfo.score = 1.0;
		errInfo.rect = errRect;
		Res.push_back( errInfo );
	}

	/************************ResizeImg*****************************/
	float ratio = 1.0;
	IplImage* imgResize = api_commen.ResizeImg( image, ratio, IMAGE_SIZE );	//320:30ms,512:50ms,720:80ms
	if(!imgResize || imgResize->nChannels != 3 || imgResize->depth != IPL_DEPTH_8U) 
	{	
		cvReleaseImage(&imgResize);imgResize = 0;
		return TEC_INVALID_PARAM;
	}
	
	/***********************************Predict**********************************/
	vecLabel.clear();
	nRet = api_caffe_FasterRCNN.Predict( imgResize, Min_T, vecLabel );
	if ( (nRet!=0) || (vecLabel.size()<1) )
	{
	   LOOGE<<"Fail to Predict";
	   cvReleaseImage(&imgResize);imgResize = 0;
	   return TEC_INVALID_PARAM;
	}

	/************************Merge*****************************/
	MergeRes.clear();
	nRet = Merge( imgResize, vecLabel, MergeRes );
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
			FaceDetectInfo ratioInfo;
			ratioInfo.label = MergeRes[i].label;
			ratioInfo.score = MergeRes[i].score;
			ratioInfo.rect = ratioRect;
			Res.push_back( ratioInfo );
		}
	}

	cvReleaseImage(&imgResize);imgResize = 0;

	return nRet;
}

int API_FACE_DETECT::Merge(
		IplImage						*image,			//[In]:image
		vector< FasterRCNNInfo >		inImgLabel, 	//[In]:ImgDetail from inImgLabel
		vector< FaceDetectInfo >		&LabelInfo)		//[Out]:LabelInfo
{
	if ( inImgLabel.size() < 1 ) 
	{ 
		LOOGE<<"MergeLabel[err]:inImgLabel.size()<1!!";
		return TEC_INVALID_PARAM;
	}
	
	int i,label,BinMode,bin_sv_Filter_Face,nRet=0;
	float score = 0.0;
	LabelInfo.clear();
	vector< FaceDetectInfo > inFaceInfo;
	vector< FaceDetectInfo > vecPartFace;
	vector< FaceDetectInfo > noFaceInfo;
	vector< FaceDetectInfo > checkInfo;

	for ( i=0;i<inImgLabel.size();i++ )
	{
		label = inImgLabel[i].label;
		score = inImgLabel[i].score;		

		if ( (label<7) && (score>=Min_T) )
		{
			FaceDetectInfo facedetectInfo;
			facedetectInfo.label = dic[label];
			facedetectInfo.score = score;
			facedetectInfo.rect = inImgLabel[i].rect;
			
			inFaceInfo.push_back( facedetectInfo );

			if (label>0)	//"face"
				vecPartFace.push_back( facedetectInfo );
		}
		else if ( (label>=7) && (label<dic.size()) )
		{
			FaceDetectInfo facedetectInfo;
			facedetectInfo.label = dic[label];
			facedetectInfo.score = score;
			facedetectInfo.rect = inImgLabel[i].rect;
			
			noFaceInfo.push_back( facedetectInfo );
		}
		else if (label>=dic.size())
		{ 
			LOOGE<<"Merge[err]!!";
			return TEC_INVALID_PARAM;
		}
	}

	/***********************************Fix_PartFace**********************************/
	vector<FaceDetectInfo>	vecFixFace;
	{
		BinMode = 1;	//1-muti part rect,2-only eye/mouse
		nRet = Fix_PartFace( image, BinMode, inFaceInfo, vecFixFace );	
		if (nRet!=0)
		{
		   LOOGE<<"Fail to Process_FixFace";
		   return TEC_INVALID_PARAM;
		}
	}

	/***********************************Fix_HalfFace**********************************/
	if ( vecFixFace.size()<1 )
	{
		nRet = Fix_HalfFace( image, inFaceInfo, vecFixFace );
		if (nRet!=0)
		{
		   LOOGE<<"Fail to Process_FixFace";
		   return TEC_INVALID_PARAM;
		}
	}

	/***********************************Fix_PartFace**********************************/
	if ( vecFixFace.size()<1 )
	{
		BinMode = 2;	//1-muti part rect,2-only eye/mouse
		nRet = Fix_PartFace( image, BinMode, inFaceInfo, vecFixFace );	
		if (nRet!=0)
		{
		   LOOGE<<"Fail to Process_FixFace";
		   return TEC_INVALID_PARAM;
		}
	}

	//push data--vecFixFace
	LabelInfo.clear();
	for(i=0;i<vecFixFace.size();i++)
	{
		if ( (vecFixFace[i].label=="face") && (vecFixFace[i].score>=Min_T) )
			LabelInfo.push_back( vecFixFace[i] );
	}

	//push data--inFaceInfo
	for(i=0;i<inFaceInfo.size();i++)
	{
		if ( (inFaceInfo[i].label=="face") && (inFaceInfo[i].score>=Max_T) )
			LabelInfo.push_back( inFaceInfo[i] );
		else if ( (inFaceInfo[i].label=="face") && (inFaceInfo[i].score>=Min_T) && (inFaceInfo[i].score<Max_T) )
		{
			bin_sv_Filter_Face = 0;
			nRet = Filter_Face( inFaceInfo[i], vecPartFace, bin_sv_Filter_Face );	
			if (nRet!=0)
			{
			   LOOGE<<"Fail to Process_FixFace";
			   return TEC_INVALID_PARAM;
			}

			if ( bin_sv_Filter_Face == 1 )
				LabelInfo.push_back( inFaceInfo[i] );
		}
	}

	//Send Err Info
	if (LabelInfo.size()<1)
	{
		Vec4i errRect(0,0,image->width-1,image->height-1);
		FaceDetectInfo errInfo;
		errInfo.label = "other";
		errInfo.score = 1.0;
		errInfo.rect = errRect;
		LabelInfo.push_back( errInfo );
	}
	else
	{
		std::sort(LabelInfo.begin(),LabelInfo.end(),Sort_FaceDetectInfo);
	}
	
	return TOK;
}

int API_FACE_DETECT::Filter_Face(
		FaceDetectInfo					tmpFaceInfo,
		vector< FaceDetectInfo > 		vecPartFace,
		int 							&bin_sv)
{
	bin_sv = 0;
	if ( vecPartFace.size() < 2 ) 
		return TOK;

	//init
	vector< pair<FaceDetectInfo,float> > vecPartFaceIOU;

	float iou;
	int i,j,nRet=0;

	//judge Part face belong to full face or not
	vecPartFaceIOU.clear();
	for(j=0;j<vecPartFace.size();j++)
	{
		nRet = api_commen.Rect_IOU(tmpFaceInfo.rect, vecPartFace[j].rect, iou); 
		if (nRet!=0)
		{
		   LOOGE<<"Fail to Fix_HalfFace:Rect_IOU";
		   continue;
		}

		vecPartFaceIOU.push_back( make_pair(vecPartFace[j],iou) );
	}

	if (vecPartFaceIOU.size()<2)
		return TOK;

	//get max iou
	sort( vecPartFaceIOU.begin(), vecPartFaceIOU.end(), IOUSortComp );
	if (vecPartFaceIOU[1].second>=0.05)	//max iou
		bin_sv = 1;
	else
		bin_sv = 0;

	return nRet;
}


int API_FACE_DETECT::Fix_AddRect_Face(
		IplImage						*image,
		int								BinMode,		//1-Half Face,2-only Eye Face,3-only mouse Face
		int								inX1,			//only for BinMode=2-Part Face
		int								inWidth,		//only for BinMode=2-Part Face
		vector< FaceDetectInfo > 		inFaceInfo,
		vector< FaceDetectInfo > 		&outFaceInfo)
{
	int i,j,m,n,rWidth,rHeight,tmp,start;
	float x1,x2,y1,y2,rw,rh,Sw,Sh,cx1,cx2,cy1,cy2;
	float Entropy,Entropy_last,stride = 0.1;
	float roi_wh = 0.4;
	float ratio = 1.0f;
	int maxLen = 256;
	int numColorBlock = 6;

	vector< float > inCH;
	outFaceInfo.clear();
	
	//Resize
	if (image->width > image->height) {
		if (image->width > maxLen)
			ratio = maxLen*1.0 / image->width;
	} 
	else 
	{	
		if (image->height > maxLen)
			ratio = maxLen*1.0 / image->height;
	}
	rWidth =  (int )image->width * ratio;
	rHeight = (int )image->height * ratio;
	IplImage *img_resize = cvCreateImage(cvSize(rWidth, rHeight), image->depth, image->nChannels);
	cvResize(image, img_resize);

	//color 2 gray
	IplImage* gray=cvCreateImage(cvGetSize(img_resize),8,1);  
	cvCvtColor(img_resize,gray,CV_BGR2GRAY);

	//norm
	IplImage* imgNorm = cvCreateImage(cvGetSize(img_resize),8,1); 
	cvNormalize(gray, imgNorm, 255, 0, CV_MINMAX, NULL);

	//smooth
	IplImage* roiBlur=cvCreateImage(cvGetSize(img_resize),8,1);  
	cvSmooth(imgNorm,roiBlur,CV_GAUSSIAN,5,5,0,0);    //3x3

	for(i=0;i<inFaceInfo.size();i++)
	{
		//img resized
		y1 = inFaceInfo[i].rect[1]*ratio;	
		y2 = inFaceInfo[i].rect[3]*ratio;
		rh = y2 - y1;

		if ( BinMode == 1 )			//1-Half Face,2-only Eye Face,3-only mouse Face
		{
			x1 = inFaceInfo[i].rect[0]*ratio;
			x2 = inFaceInfo[i].rect[2]*ratio;
			rw = x2 - x1;
			
			Sw = int(rw*roi_wh+0.5);
			Sh = int(rh*0.75+0.5);
			start = 1;
		}
		else if ( ( BinMode == 2 ) || ( BinMode == 3 ) ) //1-Half Face,2-only Eye Face,3-only mouse Face
		{
			x1 = inX1*ratio;
			x2 = (inX1+inWidth)*ratio;
			rw = x2 - x1;

			Sw = int(rw+0.5);
			Sh = int(rh*roi_wh+0.5);
			start = 0;
		}

		//out rect
		cx1 = x1;
		cy1 = y1;
		cx2 = x2;
		cy2 = y2;

		for(j=start;j<3;j++)
		{
			tmp = 0;
			if ( (j==0) && ( BinMode == 3 ) )
				tmp = int(y1*1.0/(stride*rh));	//mouse for up stride
			else if ( (j==0) && ( BinMode == 2 ) )
				tmp = int((roiBlur->height-y2)*1.0/(stride*rw));	//eye for down stride
			else if ( (j==1) && ( BinMode == 1 ) )
				tmp = int(x1*1.0/(stride*rw));	//left stride
			else if ( (j==2) && ( BinMode == 1 ) )
				tmp = int((roiBlur->width-x2)*1.0/(stride*rw));	//right stride
			
			if ( tmp < 1 )
				continue;

			for (m=0; m<tmp+1; m++ ) 
			{
				if ( (j==0) && ( BinMode == 3 ) )
					cvSetImageROI( roiBlur,cvRect(x1, y1-m*rh*stride, Sw, Sh ) );	//mouse for up stride
				else if ( (j==0) && ( BinMode == 2 ) )
					cvSetImageROI( roiBlur,cvRect(x1, y2-Sh+m*rh*stride, Sw, Sh ) );	//eye for down stride
				else if ( (j==1) && ( BinMode == 1 ) )
					cvSetImageROI( roiBlur,cvRect(x1-m*rw*stride, y1, Sw, Sh ) );	//left stride
				else if ( (j==2) && ( BinMode == 1 ) )
					cvSetImageROI( roiBlur,cvRect(x2-Sw+m*rw*stride, y1, Sw, Sh ) );	//right stride
				IplImage* sROI = cvCreateImage(cvGetSize(roiBlur),roiBlur->depth,roiBlur->nChannels);
				cvCopy( roiBlur, sROI, NULL );
				cvResetImageROI(roiBlur);	
				
				inCH.clear();
				nRet = ColorHistogram( sROI, numColorBlock, inCH );
				if ( (nRet!=0) || (inCH.size()<1) )
				{
					LOOGE<<"Fail to ColorHistogram";
					cvReleaseImage(&sROI);sROI = 0;
					cvReleaseImage(&roiBlur);roiBlur = 0;
					cvReleaseImage(&imgNorm);imgNorm = 0;
					cvReleaseImage(&gray);gray = 0;
					cvReleaseImage(&img_resize);img_resize = 0;
					return TEC_INVALID_PARAM;
				}
				cvReleaseImage(&sROI);sROI = 0;

				Entropy = 0;
				nRet = api_commen.ExtractFeat_Entropy( inCH, Entropy );
				if (nRet!=0)
				{
					LOOGE<<"Fail to ExtractFeat_Entropy";
					cvReleaseImage(&roiBlur);roiBlur = 0;
					cvReleaseImage(&imgNorm);imgNorm = 0;
					cvReleaseImage(&gray);gray = 0;
					cvReleaseImage(&img_resize);img_resize = 0;
					return TEC_INVALID_PARAM;
				}

				if( m == 0 )
					Entropy_last = Entropy;
				else if ( (m>0)&&(Entropy*1.0/Entropy_last<1.2)&&(Entropy*1.0/Entropy_last>0.8333) )
				{
					if ( (j==0) && ( BinMode == 3 ) )
						cy1 = y1-m*rh*stride;	//mouse for up stride
					else if ( (j==0) && ( BinMode == 2 ) )
						cy2 = y2+m*rh*stride;	//eye for down stride
					else if ( (j==1) && ( BinMode == 1 ) )
						cx1 = x1-m*rw*stride;	//left stride
					else if ( (j==2) && ( BinMode == 1 ) )
						cx2 = x2+m*rw*stride;	//right stride
				}
				else
					break;
			}
		}

		//save
		Vec4i tmpRect(int(cx1*1.0/ratio+0.5), int(cy1*1.0/ratio+0.5), 
			int(cx2*1.0/ratio+0.5), int(cy2*1.0/ratio+0.5));
		FaceDetectInfo faceDetectInfo;
		faceDetectInfo.label = "face";
		faceDetectInfo.rect = tmpRect;
		faceDetectInfo.score = inFaceInfo[i].score;
		outFaceInfo.push_back( faceDetectInfo );
	}

	//Release
	cvReleaseImage(&roiBlur);roiBlur = 0;
	cvReleaseImage(&imgNorm);imgNorm = 0;
	cvReleaseImage(&gray);gray = 0;
	cvReleaseImage(&img_resize);img_resize = 0;

	return nRet;
}

int API_FACE_DETECT::Fix_AddRect_PartFace(
		IplImage						*image,
		int								BinMode,		//1-muti part rect,2-only eye/mouse
		vector< FaceDetectInfo > 		inFaceInfo,
		vector< FaceDetectInfo > 		&outFaceInfo)
{
	int i,j,m,n,rWidth,rHeight,tmp;
	float x1,x2,y1,y2,rw,rh,score=0;
	int nPartCount[3] = {0};//eye,nose,mouse
	int svFaceInfo = 0;

	vector< float > inCH;
	vector< FaceDetectInfo > vecEyeFace;
	vector< FaceDetectInfo > vecNoseFace;
	vector< FaceDetectInfo > vecMouseFace;
	vector< FaceDetectInfo > vecHairFace;
	vector< FaceDetectInfo > vecBeardFace;
	outFaceInfo.clear();

	//unnormal face
	if (inFaceInfo.size()<2)
	{
		return TOK;
	}

	vecEyeFace.clear();
	vecNoseFace.clear();
	vecMouseFace.clear();
	vecHairFace.clear();
	vecBeardFace.clear();
	for (i=0;i<inFaceInfo.size();i++)
	{
		if (inFaceInfo[i].label == "eye")
		{
			nPartCount[0]++ ;
			vecEyeFace.push_back( inFaceInfo[i] );
		}
		else if (inFaceInfo[i].label == "nose")
		{
			nPartCount[1]++ ;
			vecNoseFace.push_back( inFaceInfo[i] );
		}
		else if (inFaceInfo[i].label == "mouse")
		{
			nPartCount[2]++ ;
			vecMouseFace.push_back( inFaceInfo[i] );
		}
		else if (inFaceInfo[i].label == "hair")
			vecHairFace.push_back( inFaceInfo[i] );
		else if (inFaceInfo[i].label == "beard")
			vecBeardFace.push_back( inFaceInfo[i] );
	}

	//unnormal face
	if ( (nPartCount[0]+nPartCount[1]+nPartCount[2]<1) ||							//only head/beard
		 ( ( nPartCount[0]==0 ) && ( nPartCount[1]>0 ) && ( nPartCount[2]==0 ) ) )	//only nose
	{
		return TOK;
	}

	svFaceInfo = 0;
	if		( (BinMode==1) && ( nPartCount[0]>0 ) && ( nPartCount[1]>0 ) && ( nPartCount[2]>0 ) )	//eye,nose,mouse
	{
		rw = vecEyeFace[0].rect[2]-vecEyeFace[0].rect[0];	//eye-width
		x1 = std::max(vecEyeFace[0].rect[0]-rw*0.125,0.0);
		y1 = vecEyeFace[0].rect[1];
		x2 = std::min(vecEyeFace[0].rect[2]+rw*0.125,image->width*1.0);
		y2 = vecMouseFace[0].rect[3];
		score = vecEyeFace[0].score;
		svFaceInfo = 1;
	}
	else if ( (BinMode==1) && ( nPartCount[0]>0 ) && ( nPartCount[1]>0 ) && ( nPartCount[2]==0 ) )	//eye,nose
	{
		rw = vecEyeFace[0].rect[2]-vecEyeFace[0].rect[0];	//eye-width
		x1 = std::max(vecEyeFace[0].rect[0]-rw*0.125,0.0);
		y1 = vecEyeFace[0].rect[1];
		x2 = std::min(vecEyeFace[0].rect[2]+rw*0.125,image->width*1.0);
		y2 = std::min(vecNoseFace[0].rect[3]+vecEyeFace[0].rect[3]-vecEyeFace[0].rect[0],image->height*1);
		score = vecEyeFace[0].score;
		svFaceInfo = 1;
	}
	else if ( (BinMode==1) && ( nPartCount[0]==0 ) && ( nPartCount[1]>0 ) && ( nPartCount[2]>0 ) )	//nose,mouse
	{
		rw = vecMouseFace[0].rect[2]-vecMouseFace[0].rect[0];	//eye-width
		x1 = std::max(vecMouseFace[0].rect[0]-rw*0.6,0.0);
		y1 = std::max(vecNoseFace[0].rect[1]*2-vecNoseFace[0].rect[3],0);
		x2 = std::min(vecMouseFace[0].rect[2]+rw*0.6,image->width*1.0);
		y2 = vecMouseFace[0].rect[3];
		score = vecMouseFace[0].score;
		svFaceInfo = 1;
	}
	else if ( BinMode == 2 )	//other
	{
		int binHalfFaceMode;	//1-Half Face,2-only Eye Face,3-only mouse Face
		int	inX1,inX2;			//only for BinMode=2-Part Face
		int	inWidth;			//only for BinMode=2-Part Face
	
		inFaceInfo.clear();
		if ( ( nPartCount[0]>0 ) && ( nPartCount[1]==0 ) && ( nPartCount[2]==0 ) ) //only eye
		{
			binHalfFaceMode = 2;	//1-Half Face,2-only Eye Face,3-only mouse Face
			inX1 = std::min(vecHairFace[0].rect[0],vecEyeFace[0].rect[0]);
			inX2 = std::max(vecHairFace[0].rect[2],vecEyeFace[0].rect[2]);
			inWidth = abs(inX2-inX1);
			inFaceInfo.assign( vecEyeFace.begin(), vecEyeFace.end() );
		}
		else if ( ( nPartCount[0]==0 ) && ( nPartCount[1]==0 ) && ( nPartCount[2]>0 ) ) //only mouse
		{
			binHalfFaceMode = 3;	//1-Half Face,2-only Eye Face,3-only mouse Face
			inX1 = std::min(vecBeardFace[0].rect[0],vecMouseFace[0].rect[0]);
			inX2 = std::max(vecBeardFace[0].rect[2],vecMouseFace[0].rect[2]);
			inWidth = abs(inX2-inX1);
			inFaceInfo.assign( vecMouseFace.begin(), vecMouseFace.end() );
		}

		/***********************************Fix_AddRect**********************************/
		vector<FaceDetectInfo>	vecFixHalfFace;		
		nRet = Fix_AddRect_Face( image, binHalfFaceMode, inX1, inWidth, inFaceInfo, vecFixHalfFace );
		if (nRet!=0)
		{
		   LOOGE<<"Fail to Fix_AddRect";
		   return TEC_INVALID_PARAM;
		}

		if (vecFixHalfFace.size()>0)
		{
			x1 = vecFixHalfFace[0].rect[0];
			y1 = vecFixHalfFace[0].rect[1];
			x2 = vecFixHalfFace[0].rect[2];
			y2 = vecFixHalfFace[0].rect[3];
			score = vecFixHalfFace[0].score;
			svFaceInfo = 1;
		}
		else
		{
			return TOK;
		}
	} 

	if ( svFaceInfo == 1 )
	{
		Vec4i tmpRect(int(x1+0.5), int(y1+0.5), int(x2+0.5), int(y2+0.5) );
		FaceDetectInfo faceDetectInfo;
		faceDetectInfo.label = "face";
		faceDetectInfo.rect = tmpRect;
		faceDetectInfo.score = score;
		outFaceInfo.push_back( faceDetectInfo );
	}

	return TOK;
}

int API_FACE_DETECT::Fix_HalfFace(
		IplImage						*image,
		vector< FaceDetectInfo > 		inputFaceInfo,
		vector< FaceDetectInfo > 		&outputFaceInfo)
{
	outputFaceInfo.clear();
	if ( inputFaceInfo.size() < 1 ) 
	{ 
		return TOK;
	}	

	//init
	vector< FaceDetectInfo > vecFullFace;
	vector< FaceDetectInfo > vecHalfFace;
	vector< pair<FaceDetectInfo,float> > vecHalfFaceIOU;
	vector< int > vecRemoveHalfFace;

	float iou;
	int i,j,nRet=0;

	//get full face && half face
	vecFullFace.clear();
	vecHalfFace.clear();
	for(i=0;i<inputFaceInfo.size();i++)
	{
		if ( inputFaceInfo[i].label=="face" )
			vecFullFace.push_back( inputFaceInfo[i] );
		else if ( inputFaceInfo[i].label=="halfface" )
			vecHalfFace.push_back( inputFaceInfo[i] );
	}

	//no half face
	if (vecHalfFace.size()<1)
		return TOK;

	//judge half face belong to full face or not
	if (vecFullFace.size()>0)	
	{
		vecRemoveHalfFace.clear();
		for(i=0;i<vecHalfFace.size();i++)
		{
			vecHalfFaceIOU.clear();
			for(j=0;j<vecFullFace.size();j++)
			{
				nRet = api_commen.Rect_IOU(vecHalfFace[i].rect, vecFullFace[j].rect, iou); 
				if (nRet!=0)
				{
				   LOOGE<<"Fail to Fix_HalfFace:Rect_IOU";
				   continue;
				}

				vecHalfFaceIOU.push_back( make_pair(vecFullFace[j],iou) );
			}

			if (vecHalfFaceIOU.size()<1)
				continue;

			//get max iou
			sort( vecHalfFaceIOU.begin(), vecHalfFaceIOU.end(), IOUSortComp );
			if (vecHalfFaceIOU[0].second>=0.3)	//max iou>0.3
			{
				vecRemoveHalfFace.push_back( i );
			}
		}

		//remove iou rect
		for (i=vecRemoveHalfFace.size()-1;i>=0;--i)
		{
			vecHalfFace.erase(vecHalfFace.begin()+vecRemoveHalfFace[i]);
		}
	}

	//no single half face
	if (vecHalfFace.size()<1)
		return TOK;

	/***********************************Fix_AddRect**********************************/
	vector<FaceDetectInfo>	vecFixHalfFace;
	int binHalfFaceMode = 1;	//1-Half Face,2-only Eye Face,3-only mouse Face
	int	inX1 = 0;				//only for BinMode=2-Part Face
	int	inWidth = 0;			//only for BinMode=2-Part Face
	nRet = Fix_AddRect_Face( image, binHalfFaceMode, inX1, inWidth, vecHalfFace, vecFixHalfFace );
	if (nRet!=0)
	{
	   LOOGE<<"Fail to Fix_AddRect";
	   return TEC_INVALID_PARAM;
	}

	if (vecFixHalfFace.size()>0)
	{
		outputFaceInfo.assign( vecFixHalfFace.begin(), vecFixHalfFace.end() );
	}

	return nRet;
}

int API_FACE_DETECT::Fix_PartFace(
		IplImage						*image,
		int								BinMode,		//1-muti part rect,2-only eye/mouse
		vector< FaceDetectInfo > 		inputFaceInfo,
		vector< FaceDetectInfo > 		&outputFaceInfo)
{
	outputFaceInfo.clear();
	if ( inputFaceInfo.size() < 1 ) 
	{ 
		return TOK;
	}	

	//init
	vector< FaceDetectInfo > vecFullFace;
	vector< FaceDetectInfo > vecPartFace;
	vector< pair<FaceDetectInfo,float> > vecPartFaceIOU;
	vector< int > vecRemovePartFace;

	float iou;
	int i,j,nRet=0;
	
	//get full face && Part face
	vecFullFace.clear();
	vecPartFace.clear();
	for(i=0;i<inputFaceInfo.size();i++)
	{
		if ( inputFaceInfo[i].label=="face" )
			vecFullFace.push_back( inputFaceInfo[i] );
		else if ( ( inputFaceInfo[i].label=="eye" ) || ( inputFaceInfo[i].label=="mouse" ) 
			   || ( inputFaceInfo[i].label=="nose" ) || ( inputFaceInfo[i].label=="hair" )
			   || ( inputFaceInfo[i].label=="beard" ) )
			vecPartFace.push_back( inputFaceInfo[i] );
	}

	//no two Part face
	if (vecPartFace.size()<2)
		return TOK;

	//judge Part face belong to full face or not
	if (vecFullFace.size()>0)	
	{
		vecRemovePartFace.clear();
		for(i=0;i<vecPartFace.size();i++)
		{
			vecPartFaceIOU.clear();
			for(j=0;j<vecFullFace.size();j++)
			{
				nRet = api_commen.Rect_IOU(vecPartFace[i].rect, vecFullFace[j].rect, iou); 
				if (nRet!=0)
				{
				   LOOGE<<"Fail to Fix_HalfFace:Rect_IOU";
				   continue;
				}

				vecPartFaceIOU.push_back( make_pair(vecFullFace[j],iou) );
			}

			if (vecPartFaceIOU.size()<1)
				continue;

			//get max iou
			sort( vecPartFaceIOU.begin(), vecPartFaceIOU.end(), IOUSortComp );
			if (vecPartFaceIOU[0].second>=0.1)	//max iou
			{
				vecRemovePartFace.push_back( i );
			}
		}

		//remove iou rect
		for (i=vecRemovePartFace.size()-1;i>=0;--i)
		{
			vecPartFace.erase(vecPartFace.begin()+vecRemovePartFace[i]);
		}
	}

	//no two Part face
	if (vecPartFace.size()<2)
		return TOK;

	//judge Part face belong to other Part face or not
	vecRemovePartFace.clear();
	for(i=0;i<vecPartFace.size();i++)
	{
		vecPartFaceIOU.clear();
		for(j=0;j<vecPartFace.size();j++)
		{
			if (i==j)
				continue;
			
			nRet = api_commen.Rect_IOU(vecPartFace[i].rect, vecPartFace[j].rect, iou); 
			if (nRet!=0)
			{
			   LOOGE<<"Fail to Fix_HalfFace:Rect_IOU";
			   continue;
			}

			vecPartFaceIOU.push_back( make_pair(vecPartFace[j],iou) );
		}

		if (vecPartFaceIOU.size()<1)
			continue;

		//get max iou
		sort( vecPartFaceIOU.begin(), vecPartFaceIOU.end(), IOUSortComp );
		if (vecPartFaceIOU[0].second<0.005)		//max iou
		{
			vecRemovePartFace.push_back( i );
		}
	}

	//remove iou rect
	for (i=vecRemovePartFace.size()-1;i>=0;--i)
	{
		vecPartFace.erase(vecPartFace.begin()+vecRemovePartFace[i]);
	}

	//no two Part face
	if (vecPartFace.size()<2)
		return TOK;

	/***********************************Fix_AddRect**********************************/
	vector<FaceDetectInfo>	vecFixPartFace;
	nRet = Fix_AddRect_PartFace( image, BinMode, vecPartFace, vecFixPartFace );
	if (nRet!=0)
	{
	   LOOGE<<"Fail to Fix_AddRect";
	   return TEC_INVALID_PARAM;
	}

	if (vecFixPartFace.size()>0)
	{
		outputFaceInfo.assign( vecFixPartFace.begin(), vecFixPartFace.end() );
	}

	return nRet;
}

int API_FACE_DETECT::ColorHistogram(
	IplImage 				*image, 			//[In]:image
	int 					numColorBlock,		//[In]:numColorBlock
	vector< float >			&Res)				//[Out]:Res
{
	int nRet = TOK;
	int i,j,k,count,tmpPixel,index;
	
	int height     = image->height;
	int width      = image->width;
	int step       = image->widthStep;
	int channels   = image->nChannels;
	int blockPixel = int( 255.0/numColorBlock + 0.5 );
	double size = 1.0/ (height * width) ;

	int feat_len = numColorBlock*numColorBlock*numColorBlock;
	float *pCH = new float[feat_len];
	memset(pCH, 0, feat_len * sizeof(float));

	for (i = 0; i < height; i++) 
	{
		for (j = 0; j < width; j++)
		{
			index = 0;
			for(k=0;k<channels;k++)
			{
				tmpPixel = ((uchar *)(image->imageData + i*step))[j*channels + k];
				index = int(tmpPixel*1.0/blockPixel);	//norm	
				index >> 2;
			}
			pCH[index] += 1;
		}
	}

	//norm
	Res.clear();
	for (i = 0; i < feat_len; i++) 
	{
		pCH[i] = pCH[i] * size ;
		Res.push_back(pCH[i]);
	}

	if (pCH) {delete [] pCH;pCH = NULL;}
	
	return nRet;
}

/***********************************Release**********************************/
void API_FACE_DETECT::Release()
{
	/***********************************net Model**********************************/
	api_caffe_FasterRCNN.Release();
}


