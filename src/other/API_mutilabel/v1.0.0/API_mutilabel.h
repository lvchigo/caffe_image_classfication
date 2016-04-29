#pragma once
#include <vector>
#include <opencv/cv.h>

#include "API_commen.h"
#include "API_caffe.h"
#include "API_linearsvm.h"


using namespace cv;
using namespace std;

class API_MUTILABEL
{

/***********************************Config***********************************/
#define WIDTH 256
#define HEIGHT 256
typedef unsigned long long UInt64;

#define USE_CLASSLABEL 1	//USE_CLASSLABEL-1,NO-0

/***********************************public***********************************/
public:

	/// construct function 
    API_MUTILABEL();
    
	/// distruct function
	~API_MUTILABEL(void);

	/***********************************Init*************************************/
	int Init( 
		const char* 	KeyFilePath,						//[In]:KeyFilePath
		const char* 	layerName, 							//[In]:layerName:"fc7"
		const int 		binGPU, 							//[In]:USE GPU(1) or not(0)
		const int 		deviceID );							//[In]:GPU ID

	/***********************************Predict**********************************/
	int Predict(
		IplImage						*image, 			//[In]:image
		UInt64							ImageID,			//[In]:ImageID
		const char* 					layerName,			//[In]:Layer Name by Extract
		vector< pair< string, float > >	&Res);				//[Out]:Res:In37Class/ads6Class/imgquality3class

	/***********************************Release**********************************/
	void Release();

/***********************************private***********************************/
private:

	/***********************************Init**********************************/
	API_CAFFE 		api_caffe;
	API_COMMEN 		api_commen;
	
	API_LINEARSVM 	api_libsvm_73class;
	API_LINEARSVM 	api_libsvm_in6Class;
	API_LINEARSVM 	api_libsvm_95class;

	vector< string > dic_73Class;
	vector< string > dic_in37class;
	vector< string > dic_95Class;

	/***********************************Extract Feat**********************************/
	int ExtractFeat( 
		IplImage*						image,								//[In]:image
		UInt64							ImageID,							//[In]:ImageID
		const char* 					layerName,							//[In]:Layer Name by Extract
		vector< vector< float > >		&vecIn73ClassFeat,					//for in73class
		vector< vector< float > >		&vecIn6ClassFeat);					//for in6class

	/***********************************Merge Label**********************************/
	int Merge37classLabel(
		vector< pair< int, float > >		inImgLabel, 		//[In]:ImgDetail from GetLabel
		vector< pair< string, float > > 	&LabelInfo,			//[Out]:LabelInfo
		vector< pair< int,float > > 		&intLabelInfo );	//[Out]:intLabelInfo

	int Merge73classLabel(
		vector< pair< int, float > >		inImgLabel, 		//[In]:ImgDetail from GetLabel
		vector< pair< string, float > > 	&LabelInfo,			//[Out]:LabelInfo
		vector< pair< int,float > > 		&intLabelInfo );	//[Out]:intLabelInfo

	int MergeIn6ClassLabel(
		vector< pair< int, float > > 		inImgLabel, 		//[In]:inImgLabel
		vector< pair< string, float > > 	&LabelInfo,			//[Out]:outImgLabel
		vector< pair< int,float > > 		&intLabelInfo );	//[Out]:intLabelInfo

	int Merge95classLabel(
		vector< pair< int, float > >		inImgLabel, 		//[In]:ImgDetail from GetLabel
		vector< pair< string, float > > 	&LabelInfo,			//[Out]:LabelInfo
		vector< pair< int,float > > 		&intLabelInfo );	//[Out]:intLabelInfo

};


	

