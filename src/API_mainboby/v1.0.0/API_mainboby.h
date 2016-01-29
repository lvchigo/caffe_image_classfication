#pragma once
#include <vector>
#include <opencv/cv.h>

#include "API_commen.h"
#include "API_caffe.h"
#include "API_linearsvm.h"

#include "kyheader.h"
#include "Objectness_predict.h"
#include "ValStructVec.h"
#include "CmShow.h"

using namespace cv;
using namespace std;

class API_MAINBOBY
{

/***********************************Config***********************************/
#define WIDTH 256
#define HEIGHT 256
typedef unsigned long long UInt64;

#define GET_ROI_TRAIN 1

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

	/***********************************Get_Hypothese*************************************/
	int Get_Bing_Hypothese( IplImage *img, vector< pair<float, Vec4i> > &outBox, int BinTraining );
	int Get_iou_cover( 
		vector< pair< float, Vec4i > > 				bingBox, 
		vector< pair< string, Vec4i > > 			gtBox, 
		vector< pair<Vec4i, pair< double, double > > > &vectorIouCover );

	/***********************************Get_Hypothese*************************************/
	int Get_Hypothese( IplImage *img, ValStructVec<float, Vec4i> &outBox );
	int Get_Hypothese_Entropy( IplImage *img, UInt64 ImageID, ValStructVec<float, Vec4i> &outBox );

	/***********************************Predict_Hypothese**********************************/
	int Predict_Hypothese(
		IplImage						*image, 			//[In]:image
		UInt64							ImageID,			//[In]:ImageID
		const char* 					layerName);			//[In]:Layer Name by Extract

	/***********************************Predict**********************************/
	int Predict(
		IplImage										*image, 			//[In]:image
		UInt64											ImageID,			//[In]:ImageID
		const char* 									layerName,			//[In]:Layer Name by Extract
		vector< pair< pair< string, Vec4i >, float > >	&Res);				//[Out]:Res
	
	/***********************************Release**********************************/
	void Release();

/***********************************private***********************************/
private:

	/***********************************Init**********************************/
	RunTimer<double> 	run;
	char 				szImgPath[256];

	/***********************************Init**********************************/
	Objectness 		*objNess;
	API_CAFFE 		api_caffe;
	API_COMMEN 		api_commen;
	
	API_LINEARSVM 	api_libsvm_90class;
	API_LINEARSVM 	api_libsvm_in6Class;

	vector< string > dic_90Class;

	/***********************************GetCldEhdFeat_Hypothese*************************************/
	int GetCldEhdFeat_Hypothese( IplImage *MainBody, vector< float > &feat );

	/***********************************GetTwoRect_Union_Intersection*************************************/
	int GetTwoRect_Intersection_Union( ValStructVec<float, Vec4i> inBox, double &Intersection, double &Union  );
	int GetTwoRect_Intersection_Union_vector( vector<Vec4i> inBox, double &Intersection, double &Union );

	/***********************************GetIouFeat_Hypothese*************************************/
	int GetIouFeat_Hypothese( int index, ValStructVec<float, Vec4i> inBox, vector< float > &feat );

	/***********************************GetIoU_Hypothese_withEntropy*************************************/
	int GetIoU_Hypothese_withEntropy( 
		ValStructVec<float, Vec4i> 		inBox, 
		vector< pair<int, double> > 	vecEntropy,  
		vector< pair<int, int> > 		vecClusterIndex, 
		const int 						MAX_CLUSTERS, 
		ValStructVec<float, Vec4i> 		&outIoU );

	/***********************************ExtractFeat_Entropy*************************************/
	int ExtractFeat_Entropy( vector< float > inFeat, double &Entropy );
	double ExtractFeat_Entropy_cell(Mat img);

	/***********************************GetIoU_Hypothese*************************************/
	int GetIoU_Hypothese( ValStructVec<float, Vec4i> inBox, vector< pair<int, int> > vecClusterIndex, const int MAX_CLUSTERS, ValStructVec<float, Vec4i> &outIoU );

	/***********************************Extract Feat**********************************/
	int ExtractFeat( 
		IplImage*						image,								//[In]:image
		UInt64							ImageID,							//[In]:ImageID
		const char* 					layerName,							//[In]:Layer Name by Extract
		vector< vector< float > >		&vecIn73ClassFeat,					//for in90class
		vector< vector< float > >		&vecIn6ClassFeat);					//for in6class

	/***********************************Predict_in90class**********************************/
	int Predict_in90class(
		IplImage						*image, 			//[In]:image
		UInt64							ImageID,			//[In]:ImageID
		const char* 					layerName,			//[In]:Layer Name by Extract
		vector< pair< string, float > >	&Res);				//[Out]:Res:In37Class/ads6Class/imgquality3class

	/***********************************Predict_in37class**********************************/
	int Predict_in37class(
		IplImage						*image, 			//[In]:image
		UInt64							ImageID,			//[In]:ImageID
		const char* 					layerName,			//[In]:Layer Name by Extract
		vector< pair< string, float > >	&Res);				//[Out]:Res:In37Class/ads6Class/imgquality3class

	/***********************************Merge Label**********************************/
	int Merge90classLabel(
		vector< pair< int, float > >		inImgLabel, 		//[In]:ImgDetail from GetLabel
		vector< pair< string, float > > 	&LabelInfo,			//[Out]:LabelInfo
		vector< pair< int,float > > 		&intLabelInfo );	//[Out]:intLabelInfo

	int Merge90_37classLabel(
		vector< pair< int, float > >		inImgLabel, 		//[In]:ImgDetail from GetLabel
		vector< pair< string, float > > 	&LabelInfo,			//[Out]:LabelInfo
		vector< pair< int,float > > 		&intLabelInfo );	//[Out]:intLabelInfo

	int MergeIn90_6ClassLabel(
		vector< pair< int, float > > 		inImgLabel, 		//[In]:inImgLabel
		vector< pair< string, float > > 	&LabelInfo,			//[Out]:outImgLabel
		vector< pair< int,float > > 		&intLabelInfo );	//[Out]:intLabelInfo

	int Merge_Predict(
		vector< pair< pair< string, Vec4i >, float > > 	inImgLabel,
		vector< pair< pair< string, Vec4i >, float > > 	&outImgLabel);

	int Merge_Predict_Check(
		vector< pair< pair< string, Vec4i >, float > > 	inImgLabel,
		vector< pair< pair< string, Vec4i >, float > > 	&outImgLabel);
		
};


