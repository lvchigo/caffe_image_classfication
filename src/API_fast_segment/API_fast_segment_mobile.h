/*
 * =====================================================================================
 *
 *       filename:  API_xml.h
 *
 *    description:  xml interface
 *
 *        version:  1.0
 *        created:  2016-03-08
 *       revision:  none
 *       compiler:  g++
 *
 *         author:  xiaogao
 *        company:  in66.com
 *
 *      copyright:  2016 itugo Inc. All Rights Reserved.
 *      
 * =====================================================================================
 */


#ifndef _API_FAST_SEGMENT_H_
#define _API_FAST_SEGMENT_H_

#pragma once
#include <stdio.h>
#include "API_commen/API_commen.h"
#include "API_commen/TErrorCode.h"


using namespace std;

class API_FAST_SEGMENT
{

/***********************************Common***********************************/
typedef unsigned char uchar;
typedef unsigned long long UInt64;

/***********************************public***********************************/
public:

	/// construct function 
    API_FAST_SEGMENT();
    
	/// distruct function
	~API_FAST_SEGMENT(void);

	/***********************************Predict**********************************/
	int do_fast_segment(
		uchar* 	bgr, 				//[In]:image->bgr
		string  iid,
		int 	width, 				//[In]:image->width
		int 	height,				//[In]:image->height
		int 	channel,			//[In]:image->channel
		int*	rect);				//[Out]:Res:<x1,y1,x2,y2>

/***********************************private***********************************/
private:
	API_COMMEN api_commen;

	uchar* getMaxChannel(
		uchar* 	bgr, 				//[In]:image->bgr
		int 	width, 				//[In]:image->width
		int 	height,				//[In]:image->height
		int 	channel);			//[In]:image->channel
	
	uchar* rgb2gray(
		uchar* 	bgr, 				//[In]:image->bgr
		int 	width, 				//[In]:image->width
		int 	height,				//[In]:image->height
		int 	channel,			//[In]:image->channel
		int 	&minPixel,
		int 	&maxPixel);			
	
	uchar* ImageStretchByHistogram(
		uchar* 	src, 				//[In]:image->bgr
		int 	width, 				//[In]:image->width
		int 	height,				//[In]:image->height
		int 	channel);			//[In]:image->channel 

	uchar* gaussianFilter(uchar* data, int width, int height, int channel);

	uchar* Salient_LC(
		uchar* 	gray, 				//[In]:image->bgr
		int 	width, 				//[In]:image->width
		int 	height,				//[In]:image->height
		int 	channel);			//[In]:image->channel 
	
	int FindMaxContour( 
		uchar* 	gray, 				//[In]:image->bgr
		int 	width, 				//[In]:image->width
		int 	height,				//[In]:image->height
		int 	channel,			//[In]:image->channel
		int* 	rect );

};

#endif

