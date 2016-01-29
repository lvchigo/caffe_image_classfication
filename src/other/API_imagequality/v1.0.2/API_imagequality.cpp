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
#include "API_libsvm.h"
#include "API_linearsvm.h"
#include "API_imagequality.h"
#include "ColorLayout.h"
#include "EdgeHist.h"
#include "TErrorCode.h"

#include  <iomanip>

using namespace cv;
using namespace std;


#define 	BLOCKNUM 			9
#define     CLD_DIM             72 
#define     EHD_DIM             80 

#define GLCM_DIS 3
#define GLCM_CLASS 16
#define GLCM_ANGLE_HORIZATION 0
#define GLCM_ANGLE_VERTICAL   1
#define GLCM_ANGLE_DIGONAL    2

/***********************************Init*************************************/
/// construct function 
API_IMAGEQUALITY::API_IMAGEQUALITY()
{
}

/// destruct function 
API_IMAGEQUALITY::~API_IMAGEQUALITY(void)
{
}

/***********************************Init*************************************/
static bool ImgSortComp(
	const pair< vector< int >, float > elem1, 
	const pair< vector< int >, float > elem2)
{
	return (elem1.second > elem2.second);
}

/***********************************Extract BasicInfo Feat*************************************/
int API_IMAGEQUALITY::ExtractFeat_BasicInfo( 
	IplImage* 					pSrcImg, 
	vector< float > 			&meanBasicInfo, 				//36D
	vector< float > 			&devBasicInfo, 					//36D
	vector< float > 			&entropyBasicInfo, 				//36D
	vector< float > 			&constractBasicInfo )			//4D
{
	if(pSrcImg->nChannels != 3) {
		printf("preprocess fun ExtractFeat_BasicInfo img is gray\n");
		return -1;
	}

	int i;
	int nWid = pSrcImg->width; int nHei = pSrcImg->height;

	IplImage* pRGBImg[3];
	IplImage* pGrayImg = NULL;
	for( i =0; i < 3; i++)
	{
		pRGBImg[i] = cvCreateImage(cvSize(nWid,nHei),pSrcImg->depth,1);
	}
	pGrayImg = cvCreateImage(cvSize(nWid,nHei),pSrcImg->depth,1);
	cvCvtColor(pSrcImg,pGrayImg,CV_BGR2GRAY);
	cvSplit(pSrcImg,pRGBImg[0],pRGBImg[1],pRGBImg[2],0);

	meanBasicInfo.clear();
	devBasicInfo.clear(); 
	entropyBasicInfo.clear();
	constractBasicInfo.clear();
	for( i =0; i < 4; i++)
	{
		if (i<3)
		{
			ExtractFeat_BasicInfo_Block( pRGBImg[i], meanBasicInfo, devBasicInfo, entropyBasicInfo, constractBasicInfo );
		}
		else
		{
			ExtractFeat_BasicInfo_Block( pGrayImg, meanBasicInfo, devBasicInfo, entropyBasicInfo, constractBasicInfo );
		}
	}

	/*****************************cvReleaseImage*****************************/
	if(pGrayImg){cvReleaseImage(&pGrayImg);pGrayImg = NULL;}
	for(int i = 0; i < 3; i ++)
	{
		cvReleaseImage(&pRGBImg[i]);
		pRGBImg[i] = NULL;
	}
	
	return 0;
}

void API_IMAGEQUALITY::ExtractFeat_BasicInfo_Block(
	IplImage* 					ScaleImg,
	vector< float > 			&meanBasicInfo, 			//9D
	vector< float > 			&devBasicInfo, 				//9D
	vector< float > 			&entropyBasicInfo, 			//9D
	vector< float > 			&constractBasicInfo)		//1D
{
	int Imgwidth = ScaleImg->width;
	int Imgheight = ScaleImg->height;
	int Bl_width = Imgwidth/3, Bl_height = Imgheight/3;
	CvRect BlockImg[BLOCKNUM] = {
		cvRect(0,0,Bl_width,Bl_height),
		cvRect(Imgwidth*1/3,0,Bl_width,Bl_height),
		cvRect(Imgwidth*2/3,0,Bl_width,Bl_height),
		cvRect(0,Imgheight/3,Bl_width,Bl_height),
		cvRect(Imgwidth/3,Imgheight/3,Bl_width,Bl_height),
		cvRect(Imgwidth*2/3,Imgheight/3,Bl_width,Bl_height),
		cvRect(0,Imgheight*2/3,Bl_width,Bl_height),
		cvRect(Imgwidth*1/3,Imgheight*2/3,Bl_width,Bl_height),
		cvRect(Imgwidth*2/3,Imgheight*2/3,Bl_width,Bl_height)
	};
	//	IplImage *blockimage[BLOCKNUM];
	IplImage* blockimage = cvCreateImage(cvSize(Bl_width,Bl_height),IPL_DEPTH_8U,1);
	vector <float> dev;
	CvScalar Bl_mean[BLOCKNUM];
	CvScalar Bl_dev[BLOCKNUM];

	for (int i=0;i<BLOCKNUM;i++)
	{
		cvResetImageROI(ScaleImg);
		cvSetImageROI(ScaleImg,BlockImg[i]);
		cvCopy(ScaleImg,blockimage);

		cvAvgSdv(blockimage,&Bl_mean[i],&Bl_dev[i],NULL);

		Mat img_cell( blockimage );
		double val_Entropy_cell = ExtractFeat_Entropy_cell(img_cell);

		/**********************push data*************************************/
		dev.push_back(Bl_dev[i].val[0]);
		meanBasicInfo.push_back(Bl_mean[i].val[0]);
		devBasicInfo.push_back(Bl_dev[i].val[0]);
		entropyBasicInfo.push_back( val_Entropy_cell );
	}

	cvResetImageROI(ScaleImg);
	cvReleaseImage(&blockimage);

	CvMat *dev_data;
	dev_data = cvCreateMatHeader(BLOCKNUM,1,CV_32FC1);
	cvSetData(dev_data,&dev[0],sizeof(float));

	CvScalar Bl_Ldev;   //对比度名明显度参数；
	cvAvgSdv(dev_data,NULL,&Bl_Ldev,NULL);

	/**********************push data*************************************/
	constractBasicInfo.push_back( Bl_Ldev.val[0] );

//	cvReleaseMatHeader(&dev_data);
	cvReleaseMat(&dev_data);
}

// calculate entropy of an image
double API_IMAGEQUALITY::ExtractFeat_Entropy_cell(Mat img)
{
	double temp[256];
	for(int i=0;i<256;i++)
	{
		temp[i] = 0.0;
	}

	for(int m=0;m<img.rows;m++)
	{
		const uchar* t = img.ptr<uchar>(m);
		for(int n=0;n<img.cols;n++)
		{
			int i = t[n];
			temp[i] = temp[i]+1;
		}
	}

	for(int i=0;i<256;i++)
	{
		temp[i] = temp[i]/(img.rows*img.cols);
	}

	double result = 0;
	for(int i =0;i<256;i++)
	{
		if(temp[i]==0.0)
			result = result;
		else
			result = result-temp[i]*(log(temp[i])/log(2.0));
	}
	
	return result; 
}

/***********************************ExtractFeat*************************************/
//blur 
//reference No-reference Image Quality Assessment using blur and noisy
//write by Min Goo Choi, Jung Hoon Jung  and so on 

int API_IMAGEQUALITY::ExtractFeat_Blur(
	IplImage* 					pSrcImg,
	vector< float > 			&fBlur )		//45D=5*9D
{
	if( !pSrcImg || pSrcImg->nChannels != 3 )  {
		printf("ExtractFeat_Blur err!!\n");
		return TEC_INVALID_PARAM;
	}
	
	int i,j,nRet=0;
	int Imgwidth = pSrcImg->width;
	int Imgheight = pSrcImg->height;
	int Bl_width = Imgwidth/3, Bl_height = Imgheight/3;
	CvRect BlockImg[BLOCKNUM] = {
		cvRect(0,0,Bl_width,Bl_height),
		cvRect(Imgwidth*1/3,0,Bl_width,Bl_height),
		cvRect(Imgwidth*2/3,0,Bl_width,Bl_height),
		cvRect(0,Imgheight/3,Bl_width,Bl_height),
		cvRect(Imgwidth/3,Imgheight/3,Bl_width,Bl_height),
		cvRect(Imgwidth*2/3,Imgheight/3,Bl_width,Bl_height),
		cvRect(0,Imgheight*2/3,Bl_width,Bl_height),
		cvRect(Imgwidth*1/3,Imgheight*2/3,Bl_width,Bl_height),
		cvRect(Imgwidth*2/3,Imgheight*2/3,Bl_width,Bl_height)
	};

	IplImage* pGrayImg = cvCreateImage(cvGetSize(pSrcImg),pSrcImg->depth,1);
	cvCvtColor(pSrcImg,pGrayImg,CV_BGR2GRAY);

	IplImage* blockimage = cvCreateImage(cvSize(Bl_width,Bl_height),IPL_DEPTH_8U,1);

	vector< float > blockBlur;
	fBlur.clear();
	for ( i=0;i<BLOCKNUM;i++)
	{
		cvResetImageROI(pGrayImg);
		cvSetImageROI(pGrayImg,BlockImg[i]);
		cvCopy(pGrayImg,blockimage);

		blockBlur.clear();
		nRet = ExtractFeat_Blur_Block( blockimage, blockBlur );
		if (nRet != 0)
		{
		   	cout<<"ExtractFeat_Blur_Block err!!" << endl;
			cvReleaseImage(&blockimage);blockimage = 0;
			cvReleaseImage(&pGrayImg);pGrayImg = 0;
			
			return TEC_INVALID_PARAM;
		}

		/**********************Check Data*************************************/
		//char szImgPath[256];
		//sprintf(szImgPath, "res_block/%d.jpg", i );
		//cvSaveImage( szImgPath,blockimage );

		/**********************push data*************************************/
		for( j=0;j<blockBlur.size();j++ )
			fBlur.push_back( blockBlur[j] );
	}
	
	/**********************cvReleaseImage*************************************/
	cvResetImageROI(pGrayImg);
	cvReleaseImage(&blockimage);blockimage = 0;
	cvReleaseImage(&pGrayImg);pGrayImg = 0;

	return 0;
}

int API_IMAGEQUALITY::ExtractFeat_Blur_Block(
	IplImage* 							pSrcImg,
	vector< float > 					&fBlur)		//5D
{
	if(!pSrcImg) 
		return -1;

	IplImage* pGrayImg = NULL;
	pGrayImg = cvCreateImage(cvGetSize(pSrcImg),pSrcImg->depth,1);
	//for mean filter
	IplImage* pNoisyImg = cvCreateImage(cvGetSize(pSrcImg),pSrcImg->depth,1);

	if(pSrcImg->nChannels == 3) cvCvtColor(pSrcImg,pGrayImg,CV_BGR2GRAY);
	else cvCopy(pSrcImg,pGrayImg);

	//something different form paper i use opencv median filter here
	cvSmooth(pGrayImg,pNoisyImg,CV_MEDIAN);

	int nHei = pGrayImg->height; int nWid = pGrayImg->width;

	int total = (nWid)*(nHei);

	int iLineBytes = pGrayImg->widthStep;
	uchar* pData = (uchar*)pGrayImg->imageData;

	int iNoisyBytes = pNoisyImg->widthStep;
	uchar* pNoisyData = (uchar*)pNoisyImg->imageData;

	int steps = 0;
	//result
	//blur
	double blur_mean = 0;
	double blur_ratio = 0;
	//noisy
	double nosiy_mean = 0;
	double nosiy_ratio = 0;

	//means DhMean and DvMean in paper
	//for edge
	// it is global mean in paper i will try local later
	double ghMean = 0; 
	double gvMean = 0;
	//for noisy
	double gNoisyhMean = 0;
	double gNoisyvMean = 0;
	//Nccand-mean
	double gNoisyMean = 0;

	//tmp color value for h v
	double ch = 0;	
	double cv = 0;
	//The Thresh blur value best detected
	const double blur_th = 0.1;
	//blur value sum
	double blurvalue = 0;
	//blur count
	int blur_cnt = 0;
	//edge count
	int h_edge_cnt = 0;
	int v_edge_cnt = 0;
	//noisy count
	int noisy_cnt = 0;
	// noisy value
	double noisy_value = 0;
	
	//mean Dh(x,y) in the paper 
	// in code it means Dh(x,y) and Ax(x,y)
	double* phEdgeMatric = new double[total];
	double* pvEdgeMatric = new double[total];
	// for noisy
	//Dh Dv in the paper
	double* phNoisyMatric = new double[total];
	double* pvNoisyMatric = new double[total];
	//Ncond in the paper
	double * NoisyM = new double[total];

	//means Ch(x,y) Cv(x,y) in the paper
	double* tmpH = new double[total];
	double* tmpV = new double[total];
	

	//for blur and noisy
	//loop 1
	for(int i = 0; i < nHei; i ++)
	{
		uchar* pOffset = pData;
		uchar* pNoisyOff = pNoisyData;
		steps = i*nWid;	

		for(int j = 0; j < nWid; j ++)
		{	
			int nSteps = steps + j;
			if(i == 0 || i == nHei -1)
			{
				//for edge
				phEdgeMatric[nSteps] = 0;
				pvEdgeMatric[nSteps] = 0;
				//for noisy
				phNoisyMatric[nSteps] = 0;
				pvNoisyMatric[nSteps] = 0;
			}
			else if(j == 0 || j == nWid -1)
			{
				//for edge
				phEdgeMatric[nSteps] = 0;
				pvEdgeMatric[nSteps] = 0;
				//for noisy
				phNoisyMatric[nSteps] = 0;
				pvNoisyMatric[nSteps] = 0;
			}
			else
			{
				//for edge
				ch = abs(*(pOffset-1) - *(pOffset+1)) * 1.0 / 255.0;
				phEdgeMatric[nSteps] = ch;
				ghMean += ch;

				cv = abs(*(pOffset-nWid) - *(pOffset+nWid)) * 1.0 / 255.0;
				pvEdgeMatric[nSteps] = cv;
				gvMean += cv;

				//for noisy
				ch = abs(*(pNoisyOff-1) - *(pNoisyOff+1)) * 1.0 / 255.0;
				phNoisyMatric[nSteps] = ch;
				gNoisyhMean += ch;
				cv = abs(*(pNoisyOff-nWid) - *(pNoisyOff+nWid)) * 1.0 / 255.0;
				pvNoisyMatric[nSteps] = cv;
				gNoisyvMean += cv;
			}
			
			double tmp_blur_value = 0;
			double tmp_ch = 0;
			double tmp_cv = 0;
			ch = (phEdgeMatric[nSteps] / 2);
			if(ch != 0)
				tmp_ch = abs((*pOffset) * 1.0 / 255 - ch) * 1.0 / ch;	
			cv = (pvEdgeMatric[nSteps] / 2);
			if(cv != 0)
				tmp_cv = abs((*pOffset) * 1.0 / 255 - cv) * 1.0 / cv;

			tmp_blur_value = max(tmp_ch,tmp_cv);
		//	blurvalue += tmp_blur_value;
			if(tmp_blur_value > blur_th) 
			{
				blur_cnt ++;
				blurvalue += tmp_blur_value;
			}

			pOffset ++;
			pNoisyOff ++;
		}
		pData += iLineBytes;
		pNoisyData += iNoisyBytes;
	}

	//for edge and noisy
	//for edge
	ghMean /= (total);
	gvMean /= (total);	
	//noisy
	gNoisyhMean /= total;
	gNoisyvMean /= total;

	//loop 2
	for(int i = 0; i < nHei; i ++)
	{
		steps = i*nWid;
		for(int j = 0; j < nWid; j ++)
		{
			int nSteps = steps + j;
			ch = phEdgeMatric[nSteps];
			tmpH[nSteps] = ch > ghMean ?  ch : 0;
			cv = pvEdgeMatric[nSteps];
			tmpV[nSteps] = cv > gvMean ?  cv : 0;

			ch = phNoisyMatric[nSteps];
			cv = pvNoisyMatric[nSteps];
			if(ch <= gNoisyhMean && cv <= gNoisyvMean)
			{
				NoisyM[nSteps] = max(ch,cv);
			}
			else
				NoisyM[nSteps] = 0;

			gNoisyMean += NoisyM[nSteps];
		}
	}
	gNoisyMean /= total;

	//loop 3
	for(int i = 0; i < nHei; i ++)
	{
		steps = i*(nWid);
		for(int j = 0; j < nWid; j ++)
		{
			int nSteps = steps + j;
			//for edge
			if(i == 0 || i == nHei -1)
			{
			//	phEdge[steps+j] = 0;
			//	pvEdge[steps+j] = 0;
			}
			else if(j == 0 || j == nWid -1)
			{
			//	phEdge[steps+j] = 0;
			//	pvEdge[steps+j] = 0;
			}
			else
			{
				//for edge
				if(tmpH[nSteps] > tmpH[nSteps-1] && tmpH[nSteps] > tmpH[nSteps+1])
				{
				//	phEdge[steps+j] = 1;
					h_edge_cnt ++;
				}
				//else phEdge[steps+j] = 0;

				if(tmpV[nSteps] > tmpV[steps-nWid] && tmpV[nSteps] > tmpV[steps+nWid])
				{
				//	pvEdge[steps+j] = 1;
					v_edge_cnt ++;
				}
			//	else pvEdge[steps+j] = 0;
				
				if(NoisyM[nSteps] > gNoisyMean)
				{
					noisy_cnt++;
					noisy_value += NoisyM[nSteps];
				}

			}
		
		}
	}

	if(pGrayImg){cvReleaseImage(&pGrayImg);pGrayImg = NULL;}
	if(pNoisyImg){cvReleaseImage(&pNoisyImg);pNoisyImg = NULL;}
	if(phEdgeMatric){delete []phEdgeMatric; phEdgeMatric = NULL;}
	if(pvEdgeMatric){delete []pvEdgeMatric; pvEdgeMatric = NULL;}
	if(phNoisyMatric){delete []phNoisyMatric; phNoisyMatric = NULL;}
	if(pvNoisyMatric){delete []pvNoisyMatric; pvNoisyMatric = NULL;}
	if(NoisyM){delete []NoisyM; NoisyM = NULL;}
	if(tmpH){delete []tmpH; tmpH = NULL;}
	if(tmpV){delete []tmpV; tmpV = NULL;}

	if ( ( blur_cnt == 0) || ( (h_edge_cnt+v_edge_cnt) == 0) || ( noisy_cnt == 0) || ( total == 0) )
	{
		blur_mean = 0;
		blur_ratio = 0;
		nosiy_mean = 0;
		nosiy_ratio = 0;
	}
	else
	{
		blur_mean = blurvalue * 1.0 / blur_cnt;
		blur_ratio = blur_cnt * 1.0 / (h_edge_cnt+v_edge_cnt);
		nosiy_mean = noisy_value * 1.0 / noisy_cnt;
		nosiy_ratio = noisy_cnt * 1.0 / total;
	}

	//the para is provided by paper
	//another para 1.55 0.86 0.24 0.66
	double gReulst = 1 -( blur_mean + 0.95 * blur_ratio + nosiy_mean * 0.3 + 0.75 * nosiy_ratio );

	fBlur.push_back( blur_mean );
	fBlur.push_back( blur_ratio );
	fBlur.push_back( nosiy_mean );
	fBlur.push_back( nosiy_ratio );
	fBlur.push_back( gReulst );

	return TOK;
}

int API_IMAGEQUALITY::ExtractFeat_GLCM(
	IplImage* 							pSrcImg,
	vector< float > 					&fGLCM)		//9*3*4=108D
{
	if( !pSrcImg || pSrcImg->nChannels != 3 )  {
		printf("ExtractFeat_GLCM err!!\n");
		return TEC_INVALID_PARAM;
	}
	
	int i,j,nRet=0;
	int Imgwidth = pSrcImg->width;
	int Imgheight = pSrcImg->height;
	int Bl_width = Imgwidth/3, Bl_height = Imgheight/3;
	CvRect BlockImg[BLOCKNUM] = {
		cvRect(0,0,Bl_width,Bl_height),
		cvRect(Imgwidth*1/3,0,Bl_width,Bl_height),
		cvRect(Imgwidth*2/3,0,Bl_width,Bl_height),
		cvRect(0,Imgheight/3,Bl_width,Bl_height),
		cvRect(Imgwidth/3,Imgheight/3,Bl_width,Bl_height),
		cvRect(Imgwidth*2/3,Imgheight/3,Bl_width,Bl_height),
		cvRect(0,Imgheight*2/3,Bl_width,Bl_height),
		cvRect(Imgwidth*1/3,Imgheight*2/3,Bl_width,Bl_height),
		cvRect(Imgwidth*2/3,Imgheight*2/3,Bl_width,Bl_height)
	};

	IplImage* pGrayImg = cvCreateImage(cvGetSize(pSrcImg),pSrcImg->depth,1);
	cvCvtColor(pSrcImg,pGrayImg,CV_BGR2GRAY);

	IplImage* blockimage = cvCreateImage(cvSize(Bl_width,Bl_height),IPL_DEPTH_8U,1);

	vector< float > blockGLCM;
	fGLCM.clear();
	for ( i=0;i<BLOCKNUM;i++)
	{
		cvResetImageROI(pGrayImg);
		cvSetImageROI(pGrayImg,BlockImg[i]);
		cvCopy(pGrayImg,blockimage);

		/**********************GLCM_ANGLE_HORIZATION*************************************/
		blockGLCM.clear();
		nRet = ExtractFeat_GLCM_Block( pGrayImg, GLCM_ANGLE_HORIZATION, blockGLCM);
		if (nRet != 0)
		{
		   	cout<<"ExtractFeat_Blur_Block err!!" << endl;
			cvReleaseImage(&blockimage);blockimage = 0;
			cvReleaseImage(&pGrayImg);pGrayImg = 0;
			
			return TEC_INVALID_PARAM;
		}
		
		for( j=0;j<blockGLCM.size();j++ )
			fGLCM.push_back( blockGLCM[j] );

		/**********************GLCM_ANGLE_DIGONAL*************************************/
		blockGLCM.clear();
		nRet = ExtractFeat_GLCM_Block( pGrayImg, GLCM_ANGLE_VERTICAL, blockGLCM);
		if (nRet != 0)
		{
		   	cout<<"ExtractFeat_Blur_Block err!!" << endl;
			cvReleaseImage(&blockimage);blockimage = 0;
			cvReleaseImage(&pGrayImg);pGrayImg = 0;
			
			return TEC_INVALID_PARAM;
		}
		
		for( j=0;j<blockGLCM.size();j++ )
			fGLCM.push_back( blockGLCM[j] );

		/**********************GLCM_ANGLE_DIGONAL*************************************/
		blockGLCM.clear();
		nRet = ExtractFeat_GLCM_Block( pGrayImg, GLCM_ANGLE_DIGONAL, blockGLCM);
		if (nRet != 0)
		{
		   	cout<<"ExtractFeat_Blur_Block err!!" << endl;
			cvReleaseImage(&blockimage);blockimage = 0;
			cvReleaseImage(&pGrayImg);pGrayImg = 0;
			
			return TEC_INVALID_PARAM;
		}
		
		for( j=0;j<blockGLCM.size();j++ )
			fGLCM.push_back( blockGLCM[j] );
	}
	
	/**********************cvReleaseImage*************************************/
	cvResetImageROI(pGrayImg);
	cvReleaseImage(&blockimage);blockimage = 0;
	cvReleaseImage(&pGrayImg);pGrayImg = 0;
}

int API_IMAGEQUALITY::ExtractFeat_GLCM_Block(IplImage* pSrcImg, int angleDirection, vector< float > &feat)	//4d
{
	if( !pSrcImg || pSrcImg->nChannels != 1 ) 
	{	
		cout<<"calGLCM: image err!!" << endl;
		return TEC_INVALID_PARAM;
	}

	int i,j;
	int width = pSrcImg->width;
	int height = pSrcImg->height;

	int * glcm = new int[GLCM_CLASS * GLCM_CLASS];
	int * histImage = new int[width * height];

	if(NULL == glcm || NULL == histImage)
		return TEC_INVALID_PARAM;

	uchar *data =(uchar*) pSrcImg->imageData;
	for(i = 0;i < height;i++){
		for(j = 0;j < width;j++){
			histImage[i * width + j] = (int)(data[pSrcImg->widthStep * i + j] * GLCM_CLASS / 256);
		}
	}

	for (i = 0;i < GLCM_CLASS;i++)
		for (j = 0;j < GLCM_CLASS;j++)
			glcm[i * GLCM_CLASS + j] = 0;

	int w,k,l;
	if(angleDirection == GLCM_ANGLE_HORIZATION)
	{
		for (i = 0;i < height;i++)
		{
			for (j = 0;j < width;j++)
			{
				l = histImage[i * width + j];
				if(j + GLCM_DIS >= 0 && j + GLCM_DIS < width)
				{
					k = histImage[i * width + j + GLCM_DIS];
					glcm[l * GLCM_CLASS + k]++;
				}
				if(j - GLCM_DIS >= 0 && j - GLCM_DIS < width)
				{
					k = histImage[i * width + j - GLCM_DIS];
					glcm[l * GLCM_CLASS + k]++;
				}
			}
		}
	}
	else if(angleDirection == GLCM_ANGLE_VERTICAL)
	{
		for (i = 0;i < height;i++)
		{
			for (j = 0;j < width;j++)
			{
				l = histImage[i * width + j];
				if(i + GLCM_DIS >= 0 && i + GLCM_DIS < height) 
				{
					k = histImage[(i + GLCM_DIS) * width + j];
					glcm[l * GLCM_CLASS + k]++;
				}
				if(i - GLCM_DIS >= 0 && i - GLCM_DIS < height) 
				{
					k = histImage[(i - GLCM_DIS) * width + j];
					glcm[l * GLCM_CLASS + k]++;
				}
			}
		}
	}
	else if(angleDirection == GLCM_ANGLE_DIGONAL)
	{
		for (i = 0;i < height;i++)
		{
			for (j = 0;j < width;j++)
			{
				l = histImage[i * width + j];

				if(j + GLCM_DIS >= 0 && j + GLCM_DIS < width && i + GLCM_DIS >= 0 && i + GLCM_DIS < height)
				{
					k = histImage[(i + GLCM_DIS) * width + j + GLCM_DIS];
					glcm[l * GLCM_CLASS + k]++;
				}
				if(j - GLCM_DIS >= 0 && j - GLCM_DIS < width && i - GLCM_DIS >= 0 && i - GLCM_DIS < height)
				{
					k = histImage[(i - GLCM_DIS) * width + j - GLCM_DIS];
					glcm[l * GLCM_CLASS + k]++;
				}
			}
		}
	}

	double entropy = 0,energy = 0,contrast = 0,homogenity = 0;
	for (i = 0;i < GLCM_CLASS;i++)
	{
		for (j = 0;j < GLCM_CLASS;j++)
		{
			if(glcm[i * GLCM_CLASS + j] > 0)
				entropy -= glcm[i * GLCM_CLASS + j] * log10(double(glcm[i * GLCM_CLASS + j]));
			energy += glcm[i * GLCM_CLASS + j] * glcm[i * GLCM_CLASS + j];
			contrast += (i - j) * (i - j) * glcm[i * GLCM_CLASS + j];
			homogenity += 1.0 / (1 + (i - j) * (i - j)) * glcm[i * GLCM_CLASS + j];
		}
	}

	//push data
	feat.clear();
	feat.push_back( float(entropy) );
	feat.push_back( float(energy) );
	feat.push_back( float(contrast) );
	feat.push_back( float(homogenity) );

	delete[] glcm;
	delete[] histImage;
	return 0;
}

/***********************************Extract brisque Feat*************************************/
//refernence paper:Blind/Referenceless Image Spatial Quality Evaluator
int API_IMAGEQUALITY::ExtractFeat_Brisque( IplImage* pSrcImg, vector<float> &feat )	//18D=2+4*4
{
	if( !pSrcImg || pSrcImg->nChannels != 3 )  {
		printf("ExtractFeat_Blur err!!\n");
		return TEC_INVALID_PARAM;
	}
	
	int i,j,nRet=0;
	int Imgwidth = pSrcImg->width;
	int Imgheight = pSrcImg->height;
	int Bl_width = Imgwidth/3, Bl_height = Imgheight/3;
	CvRect BlockImg[BLOCKNUM] = {
		cvRect(0,0,Bl_width,Bl_height),
		cvRect(Imgwidth*1/3,0,Bl_width,Bl_height),
		cvRect(Imgwidth*2/3,0,Bl_width,Bl_height),
		cvRect(0,Imgheight/3,Bl_width,Bl_height),
		cvRect(Imgwidth/3,Imgheight/3,Bl_width,Bl_height),
		cvRect(Imgwidth*2/3,Imgheight/3,Bl_width,Bl_height),
		cvRect(0,Imgheight*2/3,Bl_width,Bl_height),
		cvRect(Imgwidth*1/3,Imgheight*2/3,Bl_width,Bl_height),
		cvRect(Imgwidth*2/3,Imgheight*2/3,Bl_width,Bl_height)
	};

	IplImage* pGrayImg = cvCreateImage(cvGetSize(pSrcImg),pSrcImg->depth,1);
	cvCvtColor(pSrcImg,pGrayImg,CV_BGR2GRAY);

	IplImage* blockimage = cvCreateImage(cvSize(Bl_width,Bl_height),IPL_DEPTH_8U,1);

	vector< float > blockBrisque;
	feat.clear();
	for ( i=0;i<BLOCKNUM;i++)
	{
		cvResetImageROI(pGrayImg);
		cvSetImageROI(pGrayImg,BlockImg[i]);
		cvCopy(pGrayImg,blockimage);

		blockBrisque.clear();
		Mat gray( blockimage );
		gray.convertTo(gray, CV_64FC1);//char2double
		ExtractFeat_Brisque_Block(gray,blockBrisque);
		if (blockBrisque.size() == 0)
		{
		   	cout<<"ExtractFeat_Blur_Block err!!" << endl;
			cvReleaseImage(&blockimage);blockimage = 0;
			cvReleaseImage(&pGrayImg);pGrayImg = 0;
			
			return TEC_INVALID_PARAM;
		}

		/**********************Check Data*************************************/
		//char szImgPath[256];
		//sprintf(szImgPath, "res_block/%d.jpg", i );
		//cvSaveImage( szImgPath,blockimage );

		/**********************push data*************************************/
		for( j=0;j<blockBrisque.size();j++ )
			feat.push_back( blockBrisque[j] );
	}
	
	/**********************cvReleaseImage*************************************/
	cvResetImageROI(pGrayImg);
	cvReleaseImage(&blockimage);blockimage = 0;
	cvReleaseImage(&pGrayImg);pGrayImg = 0;

	return 0;
		
}

void API_IMAGEQUALITY::ExtractFeat_Brisque_Block(Mat imdist,vector< float > &feat)	//18D=2+4*4
{
	if (imdist.empty())
	{
		cout<<"img read fail"<<endl;
	}
	Mat mu,mu_sq;
	Mat sigma=Mat::zeros(imdist.rows,imdist.cols,imdist.type());
	Mat imgdouble;

	Mat imdist_mu;
	Mat avoidzero;
	double alpha,overallstd;
	Mat structdis;
	
	//Ptr<FilterEngine> f= createGaussianFilter( CV_64FC1, Size(3,3), 1, 1);
	//Mat tmp1=getGaussianKernel(3,1);
	//Ptr<FilterEngine> f= createSeparableLinearFilter(CV_64FC1,CV_64FC1, tmp1, tmp1,Point(-1,-1), 0);
	//Mat tmp2=Mat::ones(3,3,CV_64FC1);
	//f->apply(tmp2,tmp2);
	//cout<<tmp2<<endl;
	GaussianBlur(imdist,mu,Size(7,7),7.f/6,7.f/6,0);
	//BORDER_CONSTANT=0
	//cout<<imdist.at<double>(0,0)<<endl;
	//cout<<mu.at<double>(0,0)<<endl;
	//cout<<mu.at<double>(0,1)<<endl;
	multiply(mu,mu,mu_sq);
	//cout<<setprecision(10)<<mu_sq.at<double>(0,0)<<endl;
	//Mat imdist1=imdist.clone();
	//cout<<imdist1.at<double>(0,0)<<endl;
	//Mat mu1=mu.clone();
	//Mat imdist_mu1;
 //   subtract(imdist1,mu1,imdist_mu1);
	//imdist_mu1.mul(imdist_mu1);
	//GaussianBlur(imdist_mu1,imdist_mu1,Size(7,7),7.f/6,7.f/6,0);
	//cout<<imdist_mu1.at<double>(0,0)<<endl;

	multiply(imdist,imdist,imgdouble);

	GaussianBlur(imgdouble,imgdouble,Size(7,7),7.f/6,7.f/6,0);
	//cout<<setprecision(10)<<imgdouble.at<double>(0,0)<<endl;
	
	for (unsigned int i=0;i<imgdouble.rows;i++)
	{
		//double *data1=imgdouble.ptr<double>(i);
		//double *data2=mu_sq.ptr<double>(i);
		//double *data3=sigma.ptr<double>(i);
		for (unsigned int j=0;j<imgdouble.cols;j++)
		{
			//data3[j]=sqrt(abs(data1[j]-data2[j]));
			sigma.at<double>(i,j)=sqrt(abs(imgdouble.at<double>(i,j)-mu_sq.at<double>(i,j)));
		}
	}

	subtract(imdist,mu,imdist_mu);
	avoidzero=Mat::ones(sigma.rows,sigma.cols,sigma.type());
	add(sigma,avoidzero,sigma);

	//avoidzero=Mat::ones(imdist_mu1.rows,imdist_mu1.cols,imdist_mu1.type());
	//add(imdist_mu1,avoidzero,imdist_mu1);
	//divide(imdist_mu,imdist_mu1,structdis);
	
	divide(imdist_mu,sigma,structdis);//.......................................equation 1
	//imshow("str",structdis);
	//waitKey(1000);

	//cout<<setprecision(10)<<structdis.at<double>(0,0)<<endl;

	estimateggdparam(structdis,alpha,overallstd);

	feat.push_back(float(alpha));
	feat.push_back(float(overallstd*overallstd));

	double constvalue,meanparam,leftstd,rightstd;
	int shifts[][2]={0,1,1,0,1,1,-1,1};

	for(unsigned int  itr_shift =0;itr_shift <4;itr_shift++ )
	{
		Mat shifted_structdis = circshift(structdis, shifts[itr_shift][0], shifts[itr_shift][1] );
		//cout<<setprecision(10)<<shifted_structdis.at<double>(0,0)<<endl;
		Mat pair=structdis.mul(shifted_structdis);//.............................................................................equation 7,8,9,10
		//cout<<pair.at<double>(0,0)<<endl;

		estimateaggdparam(pair,
			                             alpha,
			                             leftstd,//left sigma
										 rightstd//right sigma
										 );
		
		constvalue=(sqrt(Gamma(1/alpha))/sqrt(Gamma(3/alpha)));
		meanparam=(rightstd-leftstd)*(Gamma(2/alpha)/Gamma(1/alpha))*constvalue;//.................equation 15

		feat.push_back(float(alpha));
		feat.push_back(float(meanparam));
		feat.push_back(float(leftstd*leftstd));
		feat.push_back(float(rightstd*rightstd));
	}
}

void API_IMAGEQUALITY::estimateggdparam(Mat vec,double &gamparam,double &sigma)
{
	
   Mat vec2=vec.clone();
   //cout<<vec2.at<double>(0,0)<<endl;
   Scalar sigma_sq=mean(vec2.mul(vec2));
   sigma=sqrt(sigma_sq[0]);
   Scalar E=mean(abs(vec));
   double rho=sigma_sq[0]/(E[0]*E[0]);

   vector<double> gam;
   vector<double> r_gam;
   vector<double> rho_r_gam;
   unsigned int number=int((10-0.2f)/0.001f)+1;
   gam.clear();
   r_gam.clear();
   rho_r_gam.clear();
   gam.resize(number);
   r_gam.resize(number);
   rho_r_gam.resize(number);
   
   for(unsigned i=0;i<number;i++)
   {
	   if (0==i)
	   {
		   gam[i]=0.2;
	   }else
	   {
		    gam[i]=gam[i-1]+0.001f;
	   }
	   r_gam[i]= (Gamma(1.f/gam[i])*Gamma(3.f/gam[i]))/(Gamma(2./gam[i])*Gamma(2./gam[i]));
	   rho_r_gam[i]=abs(rho-r_gam[i]);
   }
   //find min and pos
   //min_element(dv.begin(),dv.end()) return vector<double>::iterator, as location of point

   int pos = (int) ( min_element(rho_r_gam.begin(), rho_r_gam.end()) -  rho_r_gam.begin() );
   //gamma
   gamparam=gam[pos];
}

void API_IMAGEQUALITY::estimateaggdparam(Mat vec,double &alpha,double &leftstd,double &rightstd)
{

	vector<double> left;
	vector<double> right;
	left.clear();
	right.clear();
	for (unsigned int i=0;i<vec.rows;i++)
	{
		double *data1=vec.ptr<double>(i);
		for (unsigned int j=0;j<vec.cols;j++)
		{
			if (/*vec.at<double>(i,j)<0*/data1[j]<0)
			{
				left.push_back(/*vec.at<double>(i,j)*/data1[j]);
			}
			else if (/*vec.at<double>(i,j)>0*/data1[j]>0)
			{
				right.push_back(/*vec.at<double>(i,j)*/data1[j]);
			}
		}
	}
	for (unsigned int i=0;i<left.size();i++)
	{
		left[i]=left[i]*left[i];
	}
	for (unsigned int i=0;i<right.size();i++)
	{
		right[i]=right[i]*right[i];
	}
	double leftsum=0.f;
	for (unsigned int i=0;i<left.size();i++)
	{
		leftsum+=left[i];
	}
	double rightsum=0.f;
	for (unsigned int i=0;i<right.size();i++)
	{
		rightsum+=right[i];
	}
	leftstd=sqrt(leftsum/left.size());//mean
	rightstd =sqrt(rightsum/right.size());//mean
	double gammahat           = leftstd/rightstd;
	Mat vec2;
	multiply(vec,vec,vec2);
	Scalar tmp1=mean(abs(vec));
	Scalar tmp2=mean(vec2);
	double rhat=tmp1[0]*tmp1[0]/tmp2[0];
	
	double rhatnorm=(rhat*(gammahat*gammahat*gammahat +1)*(gammahat+1))/((gammahat*gammahat +1)*(gammahat*gammahat +1));

	vector<double> gam;
	vector<double> r_gam;
	vector<double> r_gam_rha;
	unsigned int number=int((10-0.2f)/0.001f)+1;
	gam.resize(number);
	r_gam.resize(number);
	r_gam_rha.resize(number);
	
	for(unsigned i=0;i<number;i++)
	{
		if (0==i)
		{
			gam[0]=0.2;
		} 
		else
		{
			gam[i]=gam[i-1]+0.001f;
		}
		
		r_gam[i]=(Gamma(2.f/gam[i])*Gamma(2.f/gam[i]))/(Gamma(1./gam[i])*Gamma(3./gam[i]));
		r_gam_rha[i]=(r_gam[i]-rhatnorm)*(r_gam[i]-rhatnorm);
	}


	//find min and pos
	int pos = (int) ( min_element(r_gam_rha.begin(),r_gam_rha.end()) - r_gam_rha.begin() );
	alpha = gam[pos];
}

Mat API_IMAGEQUALITY::circshift(Mat structdis, int a,int b)
{
	/*
	A = [ 1 2 3;
	         4 5 6; 
			 7 8 9];
	B=circshift(A,[0,1])

		B =

		3     1     2
		6     4     5
		9     7     8

		K>> B=circshift(A,[1,0])

		B =

		7     8     9
		1     2     3
		4     5     6

		K>> B=circshift(A,[-1,0])

		B =

		4     5     6
		7     8     9
		1     2     3

	*/

	int i,j;

	Mat shiftx=Mat::zeros(structdis.rows,structdis.cols,structdis.type());
	if (0==a)
	{//unchanged 
		shiftx=structdis.clone();
	}
	else if(1==a)
	{//		
		for (i=0;i<structdis.rows-1;i++)
		{
			for (j=0;j<structdis.cols;j++)
			{
				shiftx.at<double>(i+1,j)=structdis.at<double>(i,j);
			}
		}
			for (j=0;j<structdis.cols;j++)
		          shiftx.at<double>(0,j)=structdis.at<double>(structdis.rows-1,j);
	}
	else if (-1==a)
	{
		for (i=0;i<structdis.rows-1;i++)
		{
			for (j=0;j<structdis.cols;j++)
			{
				shiftx.at<double>(i,j)=structdis.at<double>(i+1,j);
			}
		}
		for (j=0;j<structdis.cols;j++)
			shiftx.at<double>(structdis.rows-1,j)=structdis.at<double>(0,j);
	}
	/*
	K>>  A = [ 1 2 3;4 5 6; 7 8 9];
	K>>  B=circshift(A,[0,1])

		B =

		3     1     2
		6     4     5
		9     7     8
		*/

	Mat shifty=Mat::zeros(shiftx.rows,shiftx.cols,shiftx.type());
	if (0==b)
	{
		shifty=shiftx.clone();
	}
	else if (1==b)
	{
		for (unsigned int i=0;i<shiftx.rows;i++)
		{
			for (unsigned int j=0;j<shiftx.cols-1;j++)
			{
				shifty.at<double>(i,j+1)=shiftx.at<double>(i,j);
			}
		}
		for (unsigned int i=0;i<shiftx.rows;i++)
			shifty.at<double>(i,0)=shiftx.at<double>(i,shiftx.cols-1);
	}

	return shifty;
}

double API_IMAGEQUALITY::Gamma( double x )
{//x>0
	if( x > 2 && x<= 3 )
	{
		const double c0 =  0.0000677106;
		const double c1 = -0.0003442342;
		const double c2 =  0.0015397681;
		const double c3 = -0.0024467480;
		const double c4 =  0.0109736958;
		const double c5 = -0.0002109075;
		const double c6 =  0.0742379071;
		const double c7 =  0.0815782188;
		const double c8 =  0.4118402518;
		const double c9 =  0.4227843370;
		const double c10 = 1.0000000000;
		double temp = 0;
		temp = temp + c0*pow( x-2.0, 10.0) + c1*pow( x-2.0, 9.0);
		temp = temp + c2*pow( x-2.0, 8.0) + c3*pow( x-2.0 , 7.0);
		temp = temp + c4*pow( x-2.0, 6.0) + c5*pow( x-2.0, 5.0 );
		temp = temp + c6*pow( x-2.0, 4.0 ) + c7*pow( x-2.0, 3.0 );
		temp = temp + c8*pow( x-2.0, 2.0 ) + c9*( x-2.0) + c10;
		return temp;
	}
	else if( x>0 && x<=1 )
	{
		return Gamma( x+2 )/(x*(x+1) );
	}
	else if( x > 1 && x<=2 )
	{
		return Gamma( x+1 )/x;
	}
	else if( x > 3 )
	{
		int i = 1;
		double temp = 1;
		while( ((x-i)>2 && (x-i) <= 3 ) == false )
		{
			temp = (x-i) * temp;
			i++;
		}
		temp = temp*(x-i);
		return temp*Gamma( x-i);
	}
	else
	{
		return 0;
	}
}

/***********************************Init*************************************/
int API_IMAGEQUALITY::Init( const char* 	KeyFilePath )					//[In]:KeyFilePath
{
	int i,nRet = 0;
	char tPath[1024] = {0};

	/***********************************Init imagequality Model*************************************/
	sprintf(tPath, "%s/imagequality/libsvm_imagequality2class_low_ratiowh_scene2block_050727_all",KeyFilePath);	//other feat
	printf("Load imagequality SVM Model:%s\n",tPath);
	nRet = api_imgquality.Init(tPath); 
	if (nRet != 0)
	{
	   cout<<"Fail to initialization "<<endl;
	   return TEC_INVALID_PARAM;
	}
	
	/***********************************Load dic_imgquality File**********************************/
	dic_imgquality.clear();
	sprintf(tPath, "%s/imagequality/dict_online",KeyFilePath);
	api_commen.loadWordDict(tPath,dic_imgquality);
	printf( "dict:size-%d,tag:", int(dic_imgquality.size()) );
	for ( i=0;i<dic_imgquality.size();i++ )
	{
		printf( "%d-%s ",i,dic_imgquality[i].c_str() );
	}
	printf( "\n" );

	return nRet;
}

/***********************************ExtractFeat**********************************/
int API_IMAGEQUALITY::ExtractFeat( 
	IplImage* 						pSrcImg, 
	UInt64							ImageID,			//[In]:ImageID
	vector< vector< float > > 		&feat)
{
	if(!pSrcImg || (pSrcImg->width<16) || (pSrcImg->height<16) || pSrcImg->nChannels != 3 || pSrcImg->depth != IPL_DEPTH_8U) 
	{	
		cout<<"image err!!" << endl;
		return TEC_INVALID_PARAM;
	}

	/*****************************Init*****************************/
	int i,j,m,n,width,height,rWidth,rHeight,nRet = 0;
	char szImgPath[1024];
	vector< float > tmpFeat;
	vector< float > tmpNormFeat;

	/*****************************Remove White Part*****************************/
	IplImage* ImgRemoveWhite = api_commen.RemoveWhitePart( pSrcImg, ImageID );
	if(!ImgRemoveWhite || (ImgRemoveWhite->width<16) || (ImgRemoveWhite->height<16) || ImgRemoveWhite->nChannels != 3 || ImgRemoveWhite->depth != IPL_DEPTH_8U) 
	{	
		cout<<"RemoveWhitePart err!!" << endl;
		cvReleaseImage(&ImgRemoveWhite);ImgRemoveWhite = 0;
		return TEC_INVALID_PARAM;
	}	

	/*****************************Init Crop Img*****************************/
	int cropSize = 256;
	width = ImgRemoveWhite->width;
	height = ImgRemoveWhite->height;
	float minWH = (width>height)?height:width;
	if ( minWH<cropSize )
		return TEC_INVALID_PARAM;

	int xROI = int( ImgRemoveWhite->width - 255 );
	int yROI = int( ImgRemoveWhite->height - 255 );
	int widthROI = int( 255 );
	int heightROI = int( 255 );
	
	/*****************************Get Muti ROI***************************/
	feat.clear();
	tmpFeat.clear();
	for( i=0;i<2;i++)
	{
		/*****************************Crop Img*****************************/
		IplImage *MutiROIResize;
		if ( i == 0 )
		{
			/***********************************Resize Image width && height*****************************/
			nRet = api_commen.GetReWH( ImgRemoveWhite->width, ImgRemoveWhite->height, 256, rWidth, rHeight );
			if (nRet != 0)
			{
			   	cout<<"GetReWH err!!" << endl;
				cvReleaseImage(&ImgRemoveWhite);ImgRemoveWhite = 0;
				return TEC_INVALID_PARAM;
			}

			/*****************************Resize Img*****************************/
			//IplImage *MutiROIResize = cvCreateImage(cvSize(rWidth, rHeight), ImgRemoveWhite->depth, ImgRemoveWhite->nChannels);
			MutiROIResize = cvCreateImage(cvSize(rWidth, rHeight), ImgRemoveWhite->depth, ImgRemoveWhite->nChannels);
			cvResize( ImgRemoveWhite, MutiROIResize );
		}
		else
		{
			if ( i == 1)
				cvSetImageROI( ImgRemoveWhite,cvRect(int((width-widthROI)*0.5+0.5),int((height-heightROI)*0.5+0.5), widthROI,heightROI) );	
			else if ( i == 2)
				cvSetImageROI( ImgRemoveWhite,cvRect(0,0, widthROI,heightROI) );	
			else if ( i == 3)
				cvSetImageROI( ImgRemoveWhite,cvRect(xROI,0, widthROI,heightROI) );	
			else if ( i == 4)
				cvSetImageROI( ImgRemoveWhite,cvRect(0,yROI, widthROI,heightROI) );	
			else if ( i == 5)
				cvSetImageROI( ImgRemoveWhite,cvRect(xROI,yROI, widthROI,heightROI) );	
		
			IplImage* MutiROI = cvCreateImage(cvGetSize(ImgRemoveWhite),ImgRemoveWhite->depth,ImgRemoveWhite->nChannels);
			cvCopy( ImgRemoveWhite, MutiROI, NULL );
			cvResetImageROI(ImgRemoveWhite);	

			/*****************************Resize Img*****************************/
			//IplImage *MutiROIResize = cvCreateImage(cvSize(256, 256), ImgRemoveWhite->depth, ImgRemoveWhite->nChannels);
			MutiROIResize = cvCreateImage(cvSize(256, 256), ImgRemoveWhite->depth, ImgRemoveWhite->nChannels);
			cvResize( MutiROI, MutiROIResize );	
			cvReleaseImage(&MutiROI);MutiROI = 0;
		}

		/***********************************Extract BasicInfo Feat**********************************/
		vector< float > featMean;		//36D
		vector< float > featMeanNorm;
		vector< float > featDev;		//36D
		vector< float > featDevNorm;
		vector< float > featEntropy;	//36D
		vector< float > featEntropyNorm;
		vector< float > featConstract;	//4D
		vector< float > featConstractNorm;
		nRet = ExtractFeat_BasicInfo( MutiROIResize, featMean, featDev, featEntropy, featConstract );	//36*3+4d
		if (nRet != 0)
		{
		   cout<<"Fail to ExtractFeat_Constract!! "<<endl;
		   return nRet;
		}

		nRet = api_commen.Normal_L2( featMean,featMeanNorm );
		nRet += api_commen.Normal_MinMax( featMeanNorm,featMean );
		nRet += api_commen.Normal_L2( featDev,featDevNorm );
		nRet += api_commen.Normal_MinMax( featDevNorm,featDev );
		nRet += api_commen.Normal_L2( featEntropy,featEntropyNorm );
		nRet += api_commen.Normal_MinMax( featEntropyNorm,featEntropy );
		nRet += api_commen.Normal_L2( featConstract,featConstractNorm );
		nRet += api_commen.Normal_MinMax( featConstractNorm,featConstract );
		if (nRet != 0)
		{
		   cout<<"Fail to Normal_L2!! "<<endl;
		   return nRet;
		}
		
#ifdef USE_MEAN_FEAT
		for( j=0;j<featMean.size();j++ )
			tmpFeat.push_back( featMean[j] );
#endif
		
#ifdef USE_DEV_FEAT
		for( j=0;j<featDev.size();j++ )
			tmpFeat.push_back( featDev[j] );
#endif

#ifdef USE_ENTROPY_FEAT
		for( j=0;j<featEntropy.size();j++ )
			tmpFeat.push_back( featEntropy[j] );
#endif

#ifdef USE_CONSTRACT_FEAT
		for( j=0;j<featConstract.size();j++ )
			tmpFeat.push_back( featConstract[j] );
#endif

		/*****************************Extract Feat:Blur*****************************/
#ifdef USE_BLUR_FEAT
		vector< float > featBlur;		//5*9d
		vector< float > featBlurNorm;
		nRet = ExtractFeat_Blur( MutiROIResize, featBlur );		
		if (nRet != 0)
		{
		   cout<<"Fail to ExtractFeat_Blur!! "<<endl;
		   return nRet;
		}
		
		nRet = api_commen.Normal_L2( featBlur,featBlurNorm );
		nRet += api_commen.Normal_MinMax( featBlurNorm,featBlur );
		if (nRet != 0)
		{
		   cout<<"Fail to Normal_L2!! "<<endl;
		   return nRet;
		}

		for( j=0;j<featBlur.size();j++ )
			tmpFeat.push_back( featBlur[j] );
#endif

		/*****************************Extract Feat:GLCM*****************************/
#ifdef USE_GLCM_FEAT
		vector< float > featGLCM;		//4*3*4=48D
		vector< float > featGLCMNorm;
		nRet = ExtractFeat_GLCM( MutiROIResize, featGLCM );
		if (nRet != 0)
		{
		   cout<<"Fail to ExtractFeat_GLCM!! "<<endl;
		   return nRet;
		}
		
		nRet = api_commen.Normal_L2( featGLCM,featGLCMNorm );
		nRet += api_commen.Normal_MinMax( featGLCMNorm,featGLCM );
		if (nRet != 0)
		{
		   cout<<"Fail to Normal_L2!! "<<endl;
		   return nRet;
		}

		for( j=0;j<featGLCM.size();j++ )
			tmpFeat.push_back( featGLCM[j] );
#endif

		/*****************************Extract Feat:Brisque*****************************/
#ifdef USE_BRISQUE_FEAT
		vector< float > featBrisque;	//18d
		vector< float > featBrisqueNorm;
		nRet = ExtractFeat_Brisque( MutiROIResize, featBrisque );
		if (nRet != 0)
		{
		   cout<<"Fail to brisque_feature!! "<<endl;
		   return nRet;
		}

		nRet = api_commen.Normal_L2( featBrisque, featBrisqueNorm );
		nRet += api_commen.Normal_MinMax( featBrisqueNorm,featBrisque );
		if (nRet != 0)
		{
		   cout<<"Fail to Normal_L2!! "<<endl;
		   return nRet;
		}

		for( j=0;j<featBrisque.size();j++ )
			tmpFeat.push_back( featBrisque[j] );
#endif

		/*****************************Extract Feat:CLD*****************************/
#ifdef USE_CLD_FEAT
		int CLD_Feat[CLD_DIM] = {0};
		vector< float > featCLD;
		vector< float > featCLDNorm;
		GF_INTERNAL::MultiBlock_LayoutExtractor( MutiROIResize, CLD_Feat );	//72d
		for( j=0;j<CLD_DIM;j++ )
			featCLD.push_back( CLD_Feat[j] );
		
		nRet = api_commen.Normal_L2( featCLD,featCLDNorm );
		nRet += api_commen.Normal_MinMax( featCLDNorm,featCLD );
		if (nRet != 0)
		{
		   cout<<"Fail to Normal_L2!! "<<endl;
		   return nRet;
		}

		for( j=0;j<featCLD.size();j++ )
			tmpFeat.push_back( featCLD[j] );
#endif

		/*****************************Extract Feat:EHD*****************************/
#ifdef USE_EHD_FEAT
		int EHD_Feat[EHD_DIM] = {0};
		vector< float > featEHD;
		vector< float > featEHDNorm;
		GF_INTERNAL::EdgeHistExtractor( MutiROIResize, EHD_Feat );				//80d
		for( j=0;j<EHD_DIM;j++ )
			featEHD.push_back( EHD_Feat[j] );

		nRet = api_commen.Normal_L2( featEHD,featEHDNorm );
		nRet += api_commen.Normal_MinMax( featEHDNorm,featEHD );
		if (nRet != 0)
		{
		   cout<<"Fail to Normal_L2!! "<<endl;
		   return nRet;
		}

		for( j=0;j<featEHD.size();j++ )
			tmpFeat.push_back( featEHD[j] );
#endif

		/*****************************output info*****************************/
		//sprintf(szImgPath, "res/ExtractFeat_%d_%ld.jpg", i, ImageID );
		//cvSaveImage( szImgPath,MutiROIResize );

		/*****************************cvReleaseImage*****************************/
		cvReleaseImage(&MutiROIResize);MutiROIResize = 0;
	}

	/*****************************push feat*****************************/
	feat.push_back( tmpFeat );
	
	/*****************************cvReleaseImage*****************************/
	cvReleaseImage(&ImgRemoveWhite);ImgRemoveWhite = 0;
	
	return 0;
}

/***********************************Predict**********************************/
int API_IMAGEQUALITY::Predict(
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
	int i,j,nRet = 0;
	vector< vector<float> > 		imgFeat;
	vector < pair < int,float > > 	ImageQualityRes;

	/*****************************ExtractFeat*****************************/
	nRet = ExtractFeat( image, ImageID, imgFeat );
	if ( (nRet != 0) || (imgFeat.size()<1) )
	{
		cout<<"Fail to GetFeat!! "<<endl;
		return TEC_BAD_STATE;
	}

	/************************SVM imgquality Predict*****************************/	
	ImageQualityRes.clear();
	nRet = api_imgquality.Predict( imgFeat, ImageQualityRes );
	if (nRet != 0)
	{	
	   	cout<<"Fail to Predict SVM api_imgquality !! "<<endl;
	   	return nRet;
	}		

	/************************MergeLabel*****************************/	
	Res.clear();
	nRet = MergeLabel_ImageQuality( ImageQualityRes, Res );
	if (nRet != 0)
	{	
	   	cout<<"Fail to MergeLabel_ImageQuality!! "<<endl;
	   	return nRet;
	}

	return nRet;
}

/***********************************MergeLabel**********************************/
int API_IMAGEQUALITY::MergeLabel_ImageQuality(
	vector< pair< int, float > >		inImgQualityLabel, 		//[In]:inImgQualityLabel
	vector< pair< string, float > > 	&LabelInfo) 			//[Out]:outImgLabel
{

	if ( inImgQualityLabel.size() < 1 ) 
	{
		printf( "MergeLabel[err]:inImgLabel.size()<1!!\n", inImgQualityLabel.size() );
		return TEC_INVALID_PARAM;
	}
	
	int i,label,imgQualityLabel,nRet = 0;
	float score = 0.0;
	
	LabelInfo.clear();
	for ( i=0;i<inImgQualityLabel.size();i++ )
	{
		label = inImgQualityLabel[i].first;
		score = inImgQualityLabel[i].second;		

		if ( label < dic_imgquality.size() )
			LabelInfo.push_back( make_pair( dic_imgquality[label], score ) );
		else
		{
			printf( "dic_imgquality[err]:label out of size:%d!!\n", dic_imgquality.size() );
			return TEC_INVALID_PARAM;
		}
	}
	
	return nRet;
}

/***********************************Release**********************************/
void API_IMAGEQUALITY::Release()
{
	/***********************************dict Model**********************************/
	dic_imgquality.clear();

	/***********************************SVM Model******libsvm3.2.0********************/
	api_imgquality.Release();
}



