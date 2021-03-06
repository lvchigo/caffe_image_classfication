#pragma once
#include <vector>
#include <cv.h>

#include "API_commen.h"
#include "API_caffe.h"
#include "API_pca.h"
#include "API_libsvm.h"
#include "API_linearsvm.h"

using namespace cv;
using namespace std;

class API_ONLINE_CLASSIFICATION
{

/***********************************Config***********************************/
#define WIDTH 256
#define HEIGHT 256
typedef unsigned long long UInt64;

#define USE_LINEARSVM 1		//LINEARSVM-1,libsvm-0
#define USE_CLASSLABEL 1	//USE_CLASSLABEL-1,NO-0

/***********************************public***********************************/
public:

	/// construct function 
    API_ONLINE_CLASSIFICATION();
    
	/// distruct function
	~API_ONLINE_CLASSIFICATION(void);

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
	
#ifdef USE_LINEARSVM
	API_LINEARSVM 	api_libsvm_73class;
	API_LINEARSVM 	api_libsvm_in6Class;
	API_LINEARSVM 	api_libsvm_ads6Class;
	API_LINEARSVM 	api_libsvm_imgquality3class;
#else
	API_LIBSVM 		api_libsvm_73class;
	API_LIBSVM 		api_libsvm_in6Class;
	API_LIBSVM 		api_libsvm_ads6Class;
	API_LIBSVM 		api_libsvm_imgquality3class;
	API_PCA 		api_pca;
#endif

	vector< string > dic_73Class;
	vector< string > dic_in37class;
	vector< string > dic_ads6Class;
	vector< string > dic_imgquality3class;

	/***********************************Extract Feat**********************************/
	int ExtractFeat( 
		IplImage* 						image, 								//[In]:image
		UInt64							ImageID,							//[In]:ImageID
		const char* 					layerName,							//[In]:Layer Name by Extract
		vector< vector< float > > 		&vecIn73ClassFeat,					//for in73class
		vector< vector< float > > 		&vecIn6ClassFeat,					//for in6class
		vector< vector< float > > 		&vecAds6ClassFeat,					//for ads6class
		vector< vector< float > > 		&vecImageQuality3ClassFeat );		//for imagequality

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

	int MergeAds6ClassLabel(
		vector< pair< int, float > > 		inImgLabel, 		//[In]:inImgLabel
		vector< pair< string, float > > 	&LabelInfo);		//[Out]:outImgLabel

	int MergeImageQuality3ClassLabel(
		vector< pair< int, float > >		inImgQualityLabel, 		//[In]:inImgQualityLabel
		vector< pair< int, float > >		inImgClassLabel, 		//[In]:inImgClassLabel
		vector< pair< string, float > > 	&LabelInfo); 			//[Out]:outImgLabel

};


	

