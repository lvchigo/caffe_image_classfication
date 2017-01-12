/*
 * =====================================================================================
 *
 *       filename:  API_caffe.h
 *
 *    description:  caffe for face detect interface
 *
 *        version:  1.0
 *        created:  2016-04-20
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


#ifndef _API_CAFFE_H_
#define _API_CAFFE_H_

#include <opencv/cv.h>
#include <vector>

#include "API_commen/API_commen.h"
#include "API_commen/TErrorCode.h"
#include "API_caffe/v2.1.0/include/caffe/caffe.hpp"

using namespace cv;
using namespace std;
using namespace caffe;

struct Info_Label
{
	int 			label;
    float 			score;
};

class API_CAFFE
{

/***********************************public***********************************/
public:

    /// construct function 
    API_CAFFE();
    
	/// distruct function
	~API_CAFFE(void);
    
	/***********************************Init*************************************/
	int Init( 
		const char* 					DL_DeployFile, 		//[In]:DL_DeployFile
		const char* 					DL_ModelFile, 		//[In]:DL_ModelFile
		const char* 					layerName, 			//[In]:layerName:"fc7"
		const int 						binGPU, 			//[In]:USE GPU(1) or not(0)
		const int 						deviceID );			//[In]:GPU ID

	/***********************************Predict**********************************/
	int Predict(
		IplImage						*image, 			//[In]:image
		const float						Min_T,				//[In]:Min_T
		vector<Info_Label> 				&vecResInfo);		//[Out]:Res
	
    /***********************************Release**********************************/
	void Release(void);

/***********************************private***********************************/
private:

	/***********************************Init**********************************/
	int nRet;	
	API_COMMEN api_commen;
	caffe::shared_ptr<caffe::Net<float> > net_dl;

};

#endif

