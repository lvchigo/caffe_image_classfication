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

#include <string>
#include <vector>
#include <opencv/cv.h>

#include "CmSaliencyRC.h"

using namespace cv;
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

	/***********************************Init*************************************/
	void Init();

	/***********************************do_fast_segment*************************************/
	int do_fast_segment( IplImage *img, string iid, Vec4i &rect );

	/***********************************Release*************************************/
	void Release();

/***********************************private***********************************/
private:
	float *pixDist;
	API_COMMEN api_commen;

	int ImageHistogram(IplImage *src1,IplImage *dst1);
	int ImageStretchByHistogram(IplImage *src1,IplImage *dst1);
	int FindMaxContour( IplImage* src, string iid, int type, int T1_Binbary, int T2_Binbary, 
		IplImage *imgThreshold, IplImage *pContourImg, Vec4i &rect );

	/***********************************Salient_LC*************************************/
	//int Salient_LC(IplImage *src, IplImage *dst, int numColorBlock, Vec4i &rect);
	int Salient_LC(IplImage *src, IplImage *dst, IplImage *dst2, IplImage *dst3, int numColorBlock, Vec4i &rect);
	/***********************************Salient_RC*************************************/
	//int Salient_RC(IplImage *src, string iid, IplImage *dst);
	int Salient_RC(IplImage *src, string iid, Mat &matOutput);
	/***********************************Salient_HC*************************************/
	int Salient_HC(IplImage *src, IplImage *dst, IplImage *dst2, IplImage *dst3, int numColorBlock, Vec4i &rect);
	/***********************************ColorHistogram*************************************/
 	int ColorHistogram(IplImage *image, int numColorBlock, vector< float > &Res);

};

#endif

