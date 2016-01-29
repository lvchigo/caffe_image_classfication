#pragma once
#include <vector>
#include <cv.h>
#include  <opencv2/opencv.hpp>

#include "API_commen.h"
#include "API_libsvm.h"
#include "API_linearsvm.h"

using namespace cv;
using namespace std;

class API_IMAGEQUALITY
{

/***********************************Config***********************************/
typedef unsigned char uchar;
typedef unsigned long long UInt64;

/***********************************Feat Config***********************************/
#define USE_MEAN_FEAT 1			//36D
#define USE_DEV_FEAT 1			//36D
#define USE_ENTROPY_FEAT 1		//36D
#define USE_CONSTRACT_FEAT 1	//4D
#define USE_BLUR_FEAT 1			//9*5=45D
#define USE_GLCM_FEAT 1			//9*3*4=108D
//#define USE_BRISQUE_FEAT 1	//9*18D
#define USE_CLD_FEAT 1			//72D
#define USE_EHD_FEAT 1			//80D

/***********************************Model Config***********************************/
//#define USE_LINEARSVM 1		//LINEARSVM-1,libsvm-0

/***********************************public***********************************/
public:

	/// construct function 
    API_IMAGEQUALITY();
    
	/// distruct function
	~API_IMAGEQUALITY(void);

	/***********************************Init*************************************/
 	int Init( const char* 	KeyFilePath );					//[In]:KeyFilePath

	/***********************************Extract Feat**********************************/
	int ExtractFeat( 
		IplImage* 						pSrcImg, 
		UInt64							ImageID,			//[In]:ImageID
		vector< vector< float > > 		&feat);

	/***********************************Predict**********************************/
	int Predict(
		IplImage						*image, 			//[In]:image
		UInt64							ImageID,			//[In]:ImageID
		vector< pair< string, float > >	&Res);				//[Out]:Res

	/***********************************Release**********************************/
	void Release();

/***********************************private***********************************/
private:

	API_COMMEN 		api_commen;
#ifdef USE_LINEARSVM
	API_LINEARSVM 	api_imgquality;
#else
	API_LIBSVM		api_imgquality;
#endif

	vector< string > dic_imgquality;

	/***********************************Extract BasicInfo Feat**********************************/
	int ExtractFeat_BasicInfo( 
		IplImage* 					pSrcImg, 
		vector< float > 			&meanBasicInfo, 				//36D
		vector< float > 			&devBasicInfo, 					//36D
		vector< float > 			&entropyBasicInfo, 				//36D
		vector< float > 			&constractBasicInfo );			//4D

	void ExtractFeat_BasicInfo_Block(
		IplImage* 					ScaleImg,
		vector< float > 			&meanBasicInfo, 			//9D
		vector< float > 			&devBasicInfo, 				//9D
		vector< float > 			&entropyBasicInfo, 			//9D
		vector< float > 			&constractBasicInfo);		//1D

	double ExtractFeat_Entropy_cell( Mat img );
	
	/***********************************Extract Blur Feat**********************************/
	//refernence paper:No-reference Image Quality Assessment using blur and noisy
	int ExtractFeat_Blur( IplImage* pSrcImg, vector< float > &fBlur );	//9*5D
	int ExtractFeat_Blur_Block( IplImage* pSrcImg, vector< float > &fBlur );	//5D

	/***********************************Extract BRISQUE Feat**********************************/
	//refernence paper:Blind/Referenceless Image Spatial Quality Evaluator
	int ExtractFeat_Brisque( IplImage* pSrcImg, vector<float> &feat);				//9*18D
	void ExtractFeat_Brisque_Block(Mat imdist,vector<float> &feat);				//18D=2+4*4
	void estimateggdparam(Mat vec,double &gamparam,double &sigma);
	void estimateaggdparam(Mat vec,double &alpha,double &leftstd,double &rightstd);
	Mat circshift(Mat structdis,int a,int b);
	double Gamma( double x );

	/***********************************Extract GLCM Feat**********************************/
	int ExtractFeat_GLCM( IplImage* pSrcImg, vector< float > &fGLCM);		//9*3*4=108D
	int ExtractFeat_GLCM_Block(IplImage* pSrcImg, int angleDirection, vector< float > &feat);
	
	/***********************************MergeLabel**********************************/
	int MergeLabel_ImageQuality(
		vector< pair< int, float > >		inImgQualityLabel, 		//[In]:inImgQualityLabel
		vector< pair< string, float > > 	&LabelInfo); 			//[Out]:outImgLabel
	
};


	

