#pragma once
#include <vector>
#include <cv.h>

#include "API_commen.h"
#include "API_caffe.h"

//BING
//#include "kyheader.h"
//#include "Objectness_predict.h"
//#include "ValStructVec.h"
//#include "CmShow.h"
//BINGpp
#include "stdafx.h"
#include "Objectness_predict.h"
#include "ValStructVec.h"

//fast rcnn
#include "CaffeCls.hpp"
#include "imagehelper.hpp"

using namespace sdktest;

using namespace cv;
using namespace std;

class API_FACE_DETECT
{

/***********************************Config***********************************/
#define WIDTH 256
#define HEIGHT 256
typedef unsigned long long UInt64;

//bingpp
#define USING_BINGPP 1

//#define USE_MODEL_VGG16 1

/***********************************public***********************************/
public:

	/// construct function 
    API_FACE_DETECT();
    
	/// distruct function
	~API_FACE_DETECT(void);

	/***********************************Init*************************************/
	int Init( 
		const char* 	KeyFilePath,						//[In]:KeyFilePath
		const char* 	layerName, 							//[In]:layerName:"fc7"
		const int 		binGPU, 							//[In]:USE GPU(1) or not(0)
		const int 		deviceID );							//[In]:GPU ID

	/***********************************Get_Xml_Hypothese******************/
	int Get_Xml_Hypothese( 
		string ImageID, int width, int height, 
		vector< pair< string, Vec4i > > vecLabelRect, 
		vector< pair< string, Vec4i > > &vecOutRect);
	
	/***********************************Get_Bing_Hypothese*************************************/
	int Get_Bing_Hypothese( IplImage *img, vector< pair<float, Vec4i> > &outBox, int BinTraining = 0 );

	/***********************************Get_iou_cover*************************************/
	int Get_iou_cover( 
		vector< pair< float, Vec4i > >				bingBox, 
		vector< pair< string, Vec4i > > 			gtBox, 
		vector< pair<Vec4i, pair< double, int > > > &vectorIouCover );

	/***********************************Predict**********************************/
	int Predict(
		IplImage										*image, 			//[In]:image
		string											ImageID,			//[In]:ImageID
		vector< pair< pair< string, Vec4i >, float > >	&Res);				//[In]:Layer Name by Extract

	/***********************************Release**********************************/
	void Release();

/***********************************private***********************************/
private:

	/***********************************Init**********************************/
	int i,j,nRet=0;
	int numObjClass 	= 10;	//voc:20;50;56;90;92;1000
	int TImgSize 		= 32;	//T of img size
	int TRectSize 		= 24;	//T of rect size
	RunTimer<double> 	run;
	char 				szImgPath[256];
	vector< string > 	dic_voc20Class;
	vector< int > 		tgt_cls;

	/***********************************Init**********************************/
	Objectness 		*objNess;
	FastRCNNCls		*fastRCNN;
	//API_CAFFE 		api_caffe;
	API_COMMEN 		api_commen;

	/***********************************GetTwoRect_Intersection_Union*************************************/
	int GetTwoRect_Intersection_Union( vector<Vec4i> inBox, double &Intersection, double &Union );

	/***********************************MergeVOC20classLabel**********************************/
	int MergeVOC20classLabel(
		vector< pair< pair< int, Vec4i >, float > >			inImgLabel, 		//[In]:ImgDetail from GetLabel
		vector< pair< pair< string, Vec4i >, float > > 		&LabelInfo);			//[Out]:LabelInfo


};


	

