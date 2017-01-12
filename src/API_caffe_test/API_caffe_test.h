/*
 * =====================================================================================
 *
 *       filename:  API_mutilabel.h
 *
 *    description:  mutilabel detect interface
 *
 *        version:  1.0
 *        created:  2016-06-20
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


#ifndef _API_MUTILABEL_H_
#define _API_MUTILABEL_H_

#include <vector>
#include <cv.h>

#include "API_commen/API_commen.h"
#include "API_caffe_test/API_caffe.h"

using namespace cv;
using namespace std;
using namespace caffe;

enum ENUM_LOG
{
	enum_module_in_logo=1
};

struct MutiLabelInfo
{
	string 			label;
    float 			score;
    Vec4i 			rect;
	vector<float>	feat;
};

struct FaceAnnoationInfo
{
	string 			label;			//(face,other)
    float 			score;			//[0,1]
    Vec4i 			rect;			//(x1,y1,x2,y2)
	int 			annoation[10];	//(eye-left,eye-right,nose,mouse-left,mouse-right)
	float			angle;			//(-90,90)
};

class API_MUTI_LABEL
{

/***********************************Common***********************************/
#define IMAGE_SIZE 320	//320:30ms,512:50ms,720:80ms
/***********************************Face:Common***********************************/
#define FACE_BLOB_WIDTH 40	
#define FACE_BLOB_HEIGHT 40
#define CHANNEL 3
/***********************************public***********************************/
public:

	/// construct function 
    API_MUTI_LABEL();
    
	/// distruct function
	~API_MUTI_LABEL(void);

	/***********************************Init*************************************/
	int Init( 
		const char* 	KeyFilePath,						//[In]:KeyFilePath
		const int 		binGPU, 							//[In]:USE GPU(1) or not(0)
		const int 		deviceID );							//[In]:GPU ID

	/***********************************Predict**********************************/
	int Predict(
		IplImage					*image, 				//[In]:image
		string						imageURL,				//[In]:image URL for CheckData
		unsigned long long			imageID,				//[In]:image ID for CheckData
		unsigned long long			childID,				//[In]:image child ID for CheckData
		float						MutiLabel_T,			//[In]:Prdict_0.8,ReSample_0.9
		vector< MutiLabelInfo >		&Res);					//[Out]:Res

	/***********************************Face:Predict**********************************/
	int Face_Predict(
		IplImage							*image, 				//[In]:image
		vector< MutiLabelInfo > 			Res_MultiLabel,			//[In]:res of MultiLabel
		vector< FaceAnnoationInfo >			&Res_FaceAnnoation);	//[Out]:res of face annoation 

	/***********************************Face:Get_Rotate_Image**********************************/
	IplImage* Face_Get_Rotate_ROI(IplImage *image, FaceAnnoationInfo faceAnnoation);

	/***********************************Face:Draw Rotate Box**********************************/
	void Face_Draw_Rotate_Box( IplImage* image, FaceAnnoationInfo faceAnnoation, Scalar color );

	/***********************************Release**********************************/
	void Release();

/***********************************private***********************************/
private:

	/***********************************Init**********************************/
	int i,j,nRet;
	RunTimer<double> 	run;
	vector< string > 	dic;
	
	/***********************************Init**********************************/
	API_COMMEN 				api_commen;
	//API_FACE_ANNOATION 		api_face_annoation;
	API_CAFFE_FasterRCNN_MULTILABEL	api_caffe_FasterRCNN_multilabel;

	/***********************************Face:Init**********************************/
	caffe::shared_ptr<caffe::Net<float> > net_dl;
	IplImage *imgMean;
	IplImage *imgSTD;

	/***********************************Init*************************************/
	int Face_Init( 
		const char* 	KeyFilePath,						//[In]:KeyFilePath
		const int 		binGPU, 							//[In]:USE GPU(1) or not(0)
		const int 		deviceID );							//[In]:GPU ID

	int Face_ReadImageToBlob( IplImage *image, Blob<float>& image_blob);

	/***********************************MergeVOC20classLabel**********************************/
	int MutiLabel_Merge(
		IplImage								*image,			//[In]:image
		float									MutiLabel_T,	//[In]:Prdict_0.8,ReSample_0.9
		vector< FasterRCNNInfo_MULTILABEL >		inImgLabel, 	//[In]:ImgDetail from inImgLabel
		vector< MutiLabelInfo >					&LabelInfo);	//[Out]:LabelInfo

	/***********************************Add_Face_Rect**********************************/
	//int Add_Face_Rect(
	//	IplImage					*image,
	//	vector< FaceDetectInfo > 	inputFaceInfo,
	//	vector< FaceDetectInfo > 	&AddFaceInfo);

	void Face_BoxPoints( FaceAnnoationInfo faceAnnoation, CvPoint2D32f pt[4] );
	void Face_cvBoxPoints( CvPoint center, int width, int height, float angle, CvPoint2D32f pt[4] );
	void Face_rotateImage(IplImage* img, IplImage *img_rotate, CvPoint2D32f center, float degree);

};

#endif

	

