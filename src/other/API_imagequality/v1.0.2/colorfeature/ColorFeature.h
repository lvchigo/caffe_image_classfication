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

#ifndef COLORFEATURE
#define COLORFEATURE

//#define CS_USING_OPENCV 1

#ifdef CS_USING_OPENCV
#include <cv.h>
#include <highgui.h>
#endif

#include "ColorSpace.h"

class ColorFeature
{

private:
	int _dim;
	float* _feat;

public:
	ColorFeature();
	~ColorFeature();
	int GetSize();
	float* Feature();

#ifdef CS_USING_OPENCV
	float* HistogramHSL(IplImage* pImgSrc);
	float* EllipseHistogramHSL(IplImage* pImgSrc);
   IplImage* HueAdjustment(IplImage* src, int hue);
#endif

   float* HistogramHSL(int* pixel, int width, int height);
	float HistogramDistance(float* x,float* y,int len);
	int ColorTableIndex(float* feat);
	

};

#endif
