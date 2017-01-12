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
#include <vector>
#include <cfloat>
#include <string>

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
	
	int do_fast_segment(
		uchar* 	bgr, 				//[In]:image->bgr
		string  iid,
		int 	width, 				//[In]:image->width
		int 	height,				//[In]:image->height
		int 	channel,			//[In]:image->channel
		int*	rect);				//[Out]:Res:<x1,y1,x2,y2>
		
	
private:

	void RGB2LAB(
		const vector<unsigned int>& 	ubuff,
		vector<double>& 				lvec,
		vector<double>& 				avec,
		vector<double>& 				bvec);

	void GaussianSmooth(
		const vector<double>&			inputImg,
		const int&						width,
		const int&						height,
		const vector<double>&			kernel,
		vector<double>& 				smoothImg);

	void GetSaliencyMap(
		const vector<unsigned int>&		inputimg,
		const int&						width,
		const int&						height,
		vector<int>&					output) ; 

	int FindMaxContour( 
		uchar* 	gray, 				//[In]:image->bgr
		int 	width, 				//[In]:image->width
		int 	height,				//[In]:image->height
		int 	channel,			//[In]:image->channel
		int* 	rect );

	//==============================================================================
	/// Normalize
	//==============================================================================
	void Normalize(
		const vector<double>&			input,
		const int&						width,
		const int&						height,
		vector<int>& 					output,
		const int&						normrange = 255)
	{
		int i,x,y;
		double maxval = 0;
		double minval = 10000000;
		
		i = 0;
		for( y = 0; y < height; y++ )
		{
			for( x = 0; x < width; x++ )
			{
				if( maxval < input[i] ) maxval = input[i];
				if( minval > input[i] ) minval = input[i];
				i++;
			}
		}
		
		double range = maxval-minval;
		if( 0 == range ) 
			range = 1;
		
		i = 0;
		output.clear();
		output.resize(width*height);
		for( y = 0; y < height; y++ )
		{
			for( x = 0; x < width; x++ )
			{
				output[i] = int((normrange*(input[i]-minval))*1.0/range +0.5);
				i++;
			}
		}
	}

};

#endif

