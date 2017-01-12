/*
 * =====================================================================================
 *
 *       filename:  API_online_classification.h
 *
 *    description:  online classification interface 
 *
 *        version:  1.0
 *        created:  2016-01-23
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

#ifndef _API_ONLINE_CLASSIFICATION_H_
#define _API_ONLINE_CLASSIFICATION_H_


#include <vector>
#include <opencv/cv.h>

#include "API_commen.h"
#include "API_multilabel_caffe.h"

using namespace cv;
using namespace std;

class API_MUTILABEL_IMGTEXT
{

/***********************************Config***********************************/
#define WIDTH 256
#define HEIGHT 256
typedef unsigned long long UInt64;

#define USE_CLASSLABEL 1	//USE_CLASSLABEL-1,NO-0

/***********************************public***********************************/
public:

	/// construct function 
    API_MUTILABEL_IMGTEXT();
    
	/// distruct function
	~API_MUTILABEL_IMGTEXT(void);

	/***********************************Init*************************************/
	int Init( 
		const char* 	KeyFilePath,						//[In]:KeyFilePath
		const char* 	layerName, 							//[In]:layerName:"fc7"
		const int 		binGPU, 							//[In]:USE GPU(1) or not(0)
		const int 		deviceID );							//[In]:GPU ID

	/***********************************Predict**********************************/
	int Predict(
		IplImage						*image, 			//[In]:image
		const char* 					layerName,			//[In]:Layer Name by Extract
		vector< pair< string, float > >	&ResProb,
		vector< pair< string, float > >	&ResScore,
		int								&label_add);			//[Out]:Res:In37Class/ads6Class/imgquality3class

	/***********************************Release**********************************/
	void Release();

/***********************************private***********************************/
private:

	/***********************************Init**********************************/
	API_MUTILABEL_CAFFE 	api_caffe;
	API_COMMEN 				api_commen;

	vector< string > 		dic_20Class;

	/***********************************Extract Feat**********************************/
	int ExtractFeat( 
		IplImage* 						image, 								//[In]:image
		UInt64							ImageID,							//[In]:ImageID
		const char* 					layerName,							//[In]:Layer Name by Extract
		vector< vector< float > > 		&vecFeat);	

	/***********************************Merge Label**********************************/

	int Merge20classLabel(
		vector< pair< int, float > >		inImgLabel, 		//[In]:ImgDetail from GetLabel
		vector< pair< string, float > > 	&LabelInfo,			//[Out]:LabelInfo
		vector< pair< int,float > > 		&intLabelInfo );	//[Out]:intLabelInfo

};

#endif
	

