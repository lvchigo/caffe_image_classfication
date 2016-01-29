#pragma once
#include <vector>
#include <cv.h>

#include "API_commen.h"
#include "API_pca.h"
#include "API_libsvm.h"

using namespace cv;
using namespace std;

class API_IN5CLASS_CLASSIFICATION
{

/***********************************Config***********************************/
#define WIDTH 256
#define HEIGHT 256
typedef unsigned long long UInt64;

/***********************************public***********************************/
public:

	/// construct function 
    API_IN5CLASS_CLASSIFICATION();
    
	/// distruct function
	~API_IN5CLASS_CLASSIFICATION(void);

	/***********************************Init*************************************/
	int Init( const char* KeyFilePath, char *svVocabulary );

	int ExtractFeat( 
		IplImage						*img, 				//[In]:image
		UInt64							ImageID,			//[In]:ImageID
		vector< vector< float > > 		&BOWFeat ); 		//[Out]:BOWFeat

	/***********************************Predict**********************************/
	int Predict(
		IplImage						*image, 			//[In]:image
		UInt64							ImageID,			//[In]:ImageID
		vector< pair< string, float > >	&Res);				//[Out]:Res

	/***********************************Release**********************************/
	void Release();

/***********************************private***********************************/
private:

	Mat 			vocabulary;
	API_COMMEN 		api_commen;
	API_PCA 		api_pca;
	API_LIBSVM 		api_libsvm_in5Class;

	vector< string > dic_5Class;

	int MergeIn5ClassLabel(
		vector< pair< int, float > > 		inImgLabel, 		//[In]:inImgLabel
		vector< pair< string, float > > 	&LabelInfo);		//[Out]:outImgLabel

};


	

