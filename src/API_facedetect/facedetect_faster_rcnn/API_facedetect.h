/*
 * =====================================================================================
 *
 *       filename:  API_facedetect.h
 *
 *    description:  face detect interface
 *
 *        version:  1.0
 *        created:  2016-04-20
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


#ifndef _API_FACEDETECT_H_
#define _API_FACEDETECT_H_

#include <vector>
#include <cv.h>

#include "API_commen.h"
#include "API_caffe_faster_rcnn.h"

using namespace cv;
using namespace std;

struct FaceDetectInfo
{
	string 	label;
    float 	score;
    Vec4i 	rect;
};

class API_FACE_DETECT
{

/***********************************Common***********************************/
#define IMAGE_SIZE 320	//320:30ms,512:50ms,720:80ms

/***********************************public***********************************/
public:

	/// construct function 
    API_FACE_DETECT();
    
	/// distruct function
	~API_FACE_DETECT(void);

	/***********************************Init*************************************/
	int Init( 
		const char* 	KeyFilePath,						//[In]:KeyFilePath
		const int 		binGPU, 							//[In]:USE GPU(1) or not(0)
		const int 		deviceID );							//[In]:GPU ID

	/***********************************Predict**********************************/
	int Predict(
		IplImage					*image, 				//[In]:image
		vector< FaceDetectInfo >	&Res);					//[In]:Layer Name by Extract

	/***********************************Release**********************************/
	void Release();

/***********************************private***********************************/
private:

	/***********************************Init**********************************/
	int i,j,nRet = 0;
	RunTimer<double> 	run;
	vector< string > 	dic;

	/***********************************Init**********************************/
	API_COMMEN 				api_commen;
	API_CAFFE_FasterRCNN	api_caffe_FasterRCNN;

	/***********************************MergeVOC20classLabel**********************************/
	int Merge(
		IplImage						*image,			//[In]:image
		vector< FasterRCNNInfo >		inImgLabel, 	//[In]:ImgDetail from inImgLabel
		vector< FaceDetectInfo >		&LabelInfo);	//[Out]:LabelInfo

	int Filter_Face(
		FaceDetectInfo					tmpFaceInfo,
		vector< FaceDetectInfo > 		vecPartFace,
		int 							&bin_sv);

	int Fix_AddRect_Face(
		IplImage						*image,
		int								BinMode,		//1-Half Face,2-only Eye Face,3-only mouse Face
		int								inX1,			//only for BinMode=2-Part Face
		int								inWidth,		//only for BinMode=2-Part Face
		vector< FaceDetectInfo > 		inputFaceInfo,
		vector< FaceDetectInfo > 		&outputFaceInfo);

	int Fix_AddRect_PartFace(
		IplImage						*image,
		int								BinMode,		//1-muti part rect,2-only eye/mouse
		vector< FaceDetectInfo > 		inFaceInfo,
		vector< FaceDetectInfo > 		&outFaceInfo);

	int Fix_HalfFace(
		IplImage						*image,
		vector< FaceDetectInfo > 		inputFaceInfo,
		vector< FaceDetectInfo > 		&outputFaceInfo);

	int Fix_PartFace(
		IplImage						*image,
		int								BinMode,		//1-muti part rect,2-only eye/mouse
		vector< FaceDetectInfo > 		inputFaceInfo,
		vector< FaceDetectInfo > 		&outputFaceInfo);

	int ColorHistogram(
		IplImage 						*image, 			//[In]:image
		int 							numColorBlock,		//[In]:numColorBlock
		vector< float >					&Res);				//[Out]:Res

};

#endif

	

