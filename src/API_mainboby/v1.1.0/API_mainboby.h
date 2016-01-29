#pragma once
#include <vector>
#include <opencv/cv.h>

#include "API_commen.h"
#include "API_caffe.h"

#include "kyheader.h"
#include "Objectness_predict.h"
#include "ValStructVec.h"
#include "CmShow.h"

#include "CaffeCls.hpp"
#include "imagehelper.hpp"

using namespace sdktest;

using namespace cv;
using namespace std;

class API_MAINBOBY
{

/***********************************Config***********************************/
#define WIDTH 256
#define HEIGHT 256
typedef unsigned long long UInt64;

//#define USE_MODEL_VGG16 1

/***********************************public***********************************/
public:

	/// construct function 
    API_MAINBOBY();
    
	/// distruct function
	~API_MAINBOBY(void);

	/***********************************Init*************************************/
	int Init( 
		const char* 	KeyFilePath,						//[In]:KeyFilePath
		const char* 	layerName, 							//[In]:layerName:"fc7"
		const int 		binGPU, 							//[In]:USE GPU(1) or not(0)
		const int 		deviceID );							//[In]:GPU ID

	/***********************************ResizeImg*************************************/
	IplImage* ResizeImg( IplImage *img, int MaxLen = 512 );

	/***********************************Get_iou_cover*************************************/
	int Get_iou_cover( 
		vector< pair< float, Vec4i > > 				bingBox, 
		vector< pair< string, Vec4i > > 			gtBox, 
		vector< pair<Vec4i, pair< double, int > > > &vectorIouCover );

	/***********************************xml*************************************/
	int load_xml( string loadXml, string &ImageID, vector< pair< string, Vec4i > > &vecLabelRect );
	int write_xml( string ImageID, string xmlSavePath, vector< pair< string, Vec4i > > vecLabelRect );
	int Get_Xml_Hypothese( 
		string ImageID, int width, int height, 
		vector< pair< string, Vec4i > > vecLabelRect, 
		vector< pair< string, Vec4i > > &vecOutRect,
		int binHypothese);
	
	/***********************************Get_Bing_Hypothese*************************************/
	int Get_Bing_Hypothese( IplImage *img, vector< pair<float, Vec4i> > &outBox, int BinTraining = 0 );

	/***********************************Predict**********************************/
	int Predict(
		IplImage										*image, 			//[In]:image
		UInt64											ImageID,			//[In]:ImageID
		const char* 									layerName,
		vector< pair< pair< string, Vec4i >, float > >	&Res);				//[In]:Layer Name by Extract
	
	/***********************************Release**********************************/
	void Release();

/***********************************private***********************************/
private:

	/***********************************Init**********************************/
	int i,j,nRet=0;
	int numObjClass 	= 90;	//voc:20;50;56;90;92;1000
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


	

