/*
 * =====================================================================================
 *
 *       filename:  API_facedetect.h
 *
 *    description:  face detect interface
 *
 *        version:  1.0
 *        created:  2016-04-29
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


#ifndef _API_FACE_ANNOATION_H_
#define _API_FACE_ANNOATION_H_

#include <opencv/cv.h>
#include <vector>

#include "caffe/caffe.hpp"
#include "API_facedetect.h"

using namespace cv;
using namespace std;
using namespace caffe;

struct FaceAnnoationInfo
{
	string 			label;			//(face,other)
    float 			score;			//[0,1]
    Vec4i 			rect;			//(x1,y1,x2,y2)
	int 			annoation[10];	//(eye-left,eye-right,nose,mouse-left,mouse-right)
	float			angle;			//(-90,90)
};

class API_FACE_ANNOATION
{

/***********************************Common***********************************/
#define FACE_BLOB_WIDTH 40	
#define FACE_BLOB_HEIGHT 40
#define CHANNEL 3

/***********************************public***********************************/
public:

    /// construct function 
    API_FACE_ANNOATION();
    
	/// distruct function
	~API_FACE_ANNOATION(void);

	/***********************************Init*************************************/
	int Init( 
		const char* 	KeyFilePath,						//[In]:KeyFilePath
		const int 		binGPU, 							//[In]:USE GPU(1) or not(0)
		const int 		deviceID );							//[In]:GPU ID

	/***********************************Predict**********************************/
	int Predict(
		IplImage							*image, 	//[In]:image
		vector< FaceAnnoationInfo >			&Res);		//[Out]:res of face annoation 

	/***********************************Get_Rotate_Image**********************************/
	IplImage* Get_Rotate_ROI(IplImage *image, FaceAnnoationInfo faceAnnoation);

	/***********************************Draw Rotate Box**********************************/
	void Draw_Rotate_Box( IplImage* img, FaceAnnoationInfo faceAnnoation, Scalar color );
	
    /***********************************Release**********************************/
	void Release(void);

/***********************************private***********************************/
private:

	//net
	caffe::shared_ptr<caffe::Net<float> > net_dl;
	IplImage *imgMean;
	IplImage *imgSTD;
	API_FACE_DETECT api_face_detect;

	int ReadImageToBlob( IplImage *image, Blob<float>& image_blob);

	/***********************************Add_Face_Rect**********************************/
	int Add_Face_Rect(
		IplImage					*image,
		vector< FaceDetectInfo > 	inputFaceInfo,
		vector< FaceDetectInfo > 	&AddFaceInfo);

	void BoxPoints( FaceAnnoationInfo faceAnnoation, CvPoint2D32f pt[4] );
	void cvBoxPoints( CvPoint center, int width, int height, float angle, CvPoint2D32f pt[4] );
	void rotateImage(IplImage* img, IplImage *img_rotate, CvPoint2D32f center, float degree);

};

#endif


