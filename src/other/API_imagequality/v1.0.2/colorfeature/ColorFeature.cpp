/****************************************************************************
*					
*			
* Copyright (c) 2011 by Amadeu zou, all rights reserved.
*
* Version:          0.0                                                     *
* Author:           Amadeu zou                                              *
* Contact:          amadeuzou@gmail.com                                     *
* URL:                                               			            *
* Create Date:      2012-02-28                                              *
*****************************************************************************/
#include "ColorFeature.h"
#include <iostream>

///
ColorFeature::ColorFeature()
{
	_dim = 0;
	_feat = NULL;
}

///
ColorFeature::~ColorFeature()
{
	if(_feat)
		delete [] _feat;
}

///
int ColorFeature::GetSize()
{
	return _dim;
}

///
///Caculate HSL histogram(256 bins)
float* ColorFeature::HistogramHSL(int* pixelArr, int width, int height)
{
    if(!pixelArr || width < 1 || height < 1)
      return NULL;

    int  nWidth = width;
    int  nHeight = height;

    int i,j,pixCount=0;

    int pNum[256] = {0};
    float* pHSVH = new float[256];

    for( i=0;i<nHeight;i++)
          {
        for( j=0;j<nWidth;j++)
                     { 
            rgb_color rgb(pixelArr[i*nWidth + j]);

            hsl_color hsl;
            hsl.from_rgb (rgb);
            
            pNum[hsl.quantization_16_4_4()] ++;

          pixCount ++;
                   }

         }//

    for( i=0;i<256;i++){
           pHSVH[i]=pNum[i]*1.0f / pixCount;
           }

   _dim = 256;
   return pHSVH;
} 

//=====================================/
#ifdef CS_USING_OPENCV
///Caculate HSL histogram(256 bins)
float* ColorFeature::HistogramHSL(IplImage* pImgSrc)
{
	if(!pImgSrc || pImgSrc->nChannels < 3)
		return NULL;

	int  nWidth = pImgSrc->width;
	int	 nHeight = pImgSrc->height;
	int  nStep = pImgSrc->widthStep;
	uchar* data;
	data  = (uchar *)pImgSrc->imageData;
	
	int i,j,pixCount=0;

	int pNum[256] = {0};
    float* pHSVH = new float[256];

	for( i=0;i<nHeight;i++)
	{
		for( j=0;j<nWidth;j++)
		{
			rgb_color rgb(data[i*nStep+j*3+2],data[i*nStep+j*3+1],data[i*nStep+j*3+0]);

			hsl_color hsl;
			hsl.from_rgb (rgb);
            
			pNum[hsl.quantization_16_4_4()] ++;

			pixCount ++;
		}

	}//

	for( i=0;i<256;i++){
		pHSVH[i]=pNum[i]*1.0f / pixCount;
	}

	_dim = 256;
	return pHSVH;

}//

///Caculate HSL histogram(256 bins) in  Ellipse
float* ColorFeature::EllipseHistogramHSL(IplImage* pImgSrc)
{
	if(!pImgSrc || pImgSrc->nChannels < 3)
		return NULL;

	int  nWidth = pImgSrc->width;
	int	 nHeight = pImgSrc->height;
	int  nStep = pImgSrc->widthStep;

	IplImage* ellipse = cvCreateImage(cvGetSize(pImgSrc),IPL_DEPTH_8U,1);
	cvZero(ellipse);
    int ew = nWidth / 6;
	int eh = nHeight / 3;
    CvPoint CircleCenter=cvPoint(nWidth/2,nHeight/2);
    CvSize EllipseAxes=cvSize(ew,eh);
    double RotateAngle=0;
    double StartDrawingAngle=0;
    double StopDrawingAngle=360;
    cvEllipse(ellipse,CircleCenter,EllipseAxes,RotateAngle,
                 StartDrawingAngle,StopDrawingAngle,cvScalarAll(255),-1,CV_AA,0);

	uchar* data;
	data  = (uchar *)pImgSrc->imageData;
	
	int i,j,pixCount=0;

	int pNum[256] = {0};
    float* pHSVH = new float[256];

	for( i=0;i<nHeight;i++)
	{
		for( j=0;j<nWidth;j++)
		{
			if(ellipse->imageData[i*ellipse->widthStep+j] > 0)
			{
				rgb_color rgb(data[i*nStep+j*3+2],data[i*nStep+j*3+1],data[i*nStep+j*3+0]);

				hsl_color hsl;
				hsl.from_rgb (rgb);

				pNum[hsl.quantization_16_4_4()] ++;

				pixCount ++;
			}
		}

	}//

	cvReleaseImage(&ellipse);
	for( i=0;i<256;i++){
		pHSVH[i]=pNum[i]*1.0f / pixCount;
	}

	_dim = 256;
	return pHSVH;

}//
///
IplImage* ColorFeature::HueAdjustment(IplImage* src, int hue)
{
	if(!src || src->nChannels < 3 )
		return NULL;

	IplImage* dst = cvCreateImage(cvGetSize(src), src->depth , src->nChannels );

	int  nWidth = src->width;
	int	 nHeight = src->height;
	int  nStep = src->widthStep;
	uchar* data;
	data  = (uchar *)src->imageData;
	

	for(int i=0;i<nHeight;i++)
	{
		for(int j=0;j<nWidth;j++)
		{
			rgb_color rgb(data[i*nStep+j*3+2],data[i*nStep+j*3+1],data[i*nStep+j*3+0]);

			hsl_color hsl;
			hsl.from_rgb (rgb);

			hsl.h += hue;
			if(hsl.h < 0)
				hsl.h = -hsl.h ;
			if(hsl.h >= 360)
				hsl.h -= 360;

			rgb.from_hsl(hsl.h ,hsl.s ,hsl.l );
			dst->imageData[i*dst->widthStep+j*3+0] = rgb.b ;
			dst->imageData[i*dst->widthStep+j*3+1] = rgb.g ;
			dst->imageData[i*dst->widthStep+j*3+2] = rgb.r ;
            
		}

	}//

	return dst;
}
#endif
//========================================//

///Histogram distance
float ColorFeature::HistogramDistance(float* x,float* y,int len)
{
	
	float dist = 0.0,mins=0,sx=0,sy=0;
	
	for (int i = 0; i < len; i++)
	{
		mins += MIN(x [i],y [i ]);
		sx += x[i];
		sy += y[i];
	}
	dist = mins / (MIN(sx ,sy ) + 0.000001f);
	dist = 1 - dist;
	return dist;
}

///
int ColorFeature::ColorTableIndex(float* feat)
{
	if(!feat)
		return -1;

	float d_min = 1;
    int index = 0;
	for(int i = 0; i < TABLE_COLS; i++)
	{
		float dist = HistogramDistance(feat, (float*)TABLE_FEAT[i], FEAT_DIM);
		if(dist < d_min)
		{
			index = i;
			d_min = dist;
		}

	}

	return index;
}
