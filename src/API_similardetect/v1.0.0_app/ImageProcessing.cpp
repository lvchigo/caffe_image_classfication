#include <math.h>
#include <iostream>
#include <string.h>
#include "ImageProcessing.h"

using namespace std;

/***********************************Init*************************************/
/// construct function 
API_IMAGEPROCESS::API_IMAGEPROCESS()
{
}

/// destruct function 
API_IMAGEPROCESS::~API_IMAGEPROCESS(void)
{
}

unsigned char * API_IMAGEPROCESS::ImageCopy(const unsigned char *src, int width, int height, int nChannel)
{
	if( (!src) || (width<MIN_IMAGE_WIDTH) || (height<MIN_IMAGE_HEIGHT) || (nChannel!=3) )
		return NULL;

	int i,j,k;
	int nSize = width * height * nChannel;
	unsigned char *dst = new unsigned char[nSize];

	for (i=0; i<height; ++i)
    {
        for (j=0; j<width; ++j)
        {
        	for (k=0; k<nChannel; ++k)
        	{
				dst[i*width*nChannel+j*nChannel+k] = src[i*width*nChannel+j*nChannel+k];
        	}
		}
	}

	return dst;
}

unsigned char * API_IMAGEPROCESS::ImageROI(const unsigned char *src, int width, int height, int nChannel, int roi_x, int roi_y, int roi_w, int roi_h)
{
    if( (!src) || (width<MIN_IMAGE_WIDTH) || (height<MIN_IMAGE_HEIGHT) || (nChannel<0) ||
		(roi_x<0) || (roi_y<0) || (roi_w<0) || (roi_h<0) || 
		(roi_x>width-1) || (roi_y>height-1) || (roi_w>width) || (roi_h>height) ||
		(roi_x+roi_w>width) || (roi_y+roi_h>height) )
		return NULL;

	int i,j,k;
	int nSize = roi_w * roi_h * nChannel;
	unsigned char *dst = new unsigned char[nSize];

	for (i=roi_y; i<roi_h; ++i)
    {
        for (j=roi_x; j<roi_w; ++j)
        {
        	for (k=0; k<nChannel; ++k)
        	{
				dst[i*roi_w*nChannel+j*nChannel+k] = src[i*roi_w*nChannel+j*nChannel+k];
        	}
		}
	}

	return dst;
}

unsigned char * API_IMAGEPROCESS::ImageResize(const unsigned char *src, int width, int height, int nChannel, int scale_w, int scale_h)
{
	if( (!src) || (width<MIN_IMAGE_WIDTH) || (height<MIN_IMAGE_HEIGHT) || (nChannel!=3) || (scale_w<1) || (scale_h<1) )
		return NULL;

	int i,j,k,scale_x,scale_y,index=0;
	int nSize = scale_w * scale_h * nChannel;
	unsigned char *dst = new unsigned char[nSize];

	for (i=0; i<scale_h; ++i)
    {
    	scale_y = int(i*height*1.0/scale_h+0.5);
        for (j=0; j<scale_w; ++j)
        {
        	scale_x = int(j*width*1.0/scale_w+0.5);
        	for (k=0; k<nChannel; ++k)
        	{
        		index = scale_y*width*nChannel+scale_x*nChannel+k;
				dst[i*scale_w*nChannel+j*nChannel+k] = src[index];
        	}
		}
	}

	return dst;
}

unsigned char * API_IMAGEPROCESS::ImageRGB2Gray(const unsigned char *src, int width, int height, int nChannel)
{
	if( (!src) || (width<MIN_IMAGE_WIDTH) || (height<MIN_IMAGE_HEIGHT) || (nChannel!=3) )
		return NULL;

	int i,j,gray,R,G,B;
	int nSize = width * height;
	unsigned char *dst = new unsigned char[nSize];

	for (i=0; i<height; ++i)
    {
        for (j=0; j<width; ++j)
        {
        	B = src[i*width*nChannel+j*nChannel+0];
			G = src[i*width*nChannel+j*nChannel+1];
			R = src[i*width*nChannel+j*nChannel+2];
			gray = (R*30 + G*59 + B*11 + 50) / 100;
			dst[i*width+j] = gray;
		}
	}

	return dst;
}

//only for gray image
unsigned char API_IMAGEPROCESS::GetMedianNum(unsigned char *bArray, int iFilterLen)
{
	int		i,j;
	unsigned char bTemp;
	
	for (j = 0; j < iFilterLen - 1; j ++)
	{
		for (i = 0; i < iFilterLen - j - 1; i ++)
		{
			if (bArray[i] > bArray[i + 1])
			{				
				bTemp = bArray[i];
				bArray[i] = bArray[i + 1];
				bArray[i + 1] = bTemp;
			}
		}
	}
	
	if ((iFilterLen & 1) > 0)
		bTemp = bArray[(iFilterLen + 1) / 2];
	else
		bTemp = (bArray[iFilterLen / 2] + bArray[iFilterLen / 2 + 1]) / 2;
	
	return bTemp;
}

//only for gray image
unsigned char * API_IMAGEPROCESS::ImageMedianFilter(const unsigned char *src, int width, int height, int nChannel, int Filter_Kernal )
{		
	if( (!src) || (width<MIN_IMAGE_WIDTH) || (height<MIN_IMAGE_HEIGHT) || (nChannel!=1) )
		return NULL;

	int i,j,k,iFilterLen;
	int nSize = width * height * nChannel;
	unsigned char *dst = new unsigned char[nSize];
	int half_kernal = int(Filter_Kernal*0.5);
	unsigned char *img_roi = NULL;
	int bin_bound = 0;
	
	for (i=0; i<height; ++i)
    {
        for (j=0; j<width; ++j)
        {
        	if ( (i<half_kernal) || (j<half_kernal) || (i>width-half_kernal-1) || (j>height-half_kernal-1) )
				bin_bound = 1;
			else
				bin_bound = 0;

			if (bin_bound != 1)
        		img_roi = ImageROI(src, width, height, nChannel, (i-half_kernal), (j-half_kernal), Filter_Kernal, Filter_Kernal);
			
        	for (k=0; k<nChannel; ++k)
        	{
        		if (bin_bound == 1)
					dst[i*width*nChannel+j*nChannel+k] = src[i*width*nChannel+j*nChannel+k];
				else
				{
					iFilterLen = Filter_Kernal*Filter_Kernal;
					dst[i*width*nChannel+j*nChannel+k] = GetMedianNum(img_roi, iFilterLen);
				}
        	}	

			//delete
			if (img_roi) {delete [] img_roi;img_roi = NULL;}
		}
	}

	//delete
	if (img_roi) {delete [] img_roi;img_roi = NULL;}

	return dst;
}

