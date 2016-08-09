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


#ifndef _API_CAFFE_FASTER_RCNN_H_
#define _API_CAFFE_FASTER_RCNN_H_

#include <opencv/cv.h>
#include <vector>

#include "API_commen.h"
#include "caffe/caffe.hpp"
#include <TErrorCode.h>

using namespace cv;
using namespace std;
using namespace caffe;

struct FasterRCNNInfo
{
	int 			label;
    float 			score;
    Vec4i 			rect;
	vector<float>	feat;
};

struct RectInfo
{
	int 			label;
    float 			score;
	float 			bbox[4];	//output bbox-bias
	float 			rect[4];	//output bbox
	vector<float>	feat;
};

class API_CAFFE_FasterRCNN
{

/***********************************public***********************************/
public:

    /// construct function 
    API_CAFFE_FasterRCNN();
    
	/// distruct function
	~API_CAFFE_FasterRCNN(void);
    
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
		vector<FasterRCNNInfo> 			&vecResInfo);		//[Out]:Res
	
    /***********************************Release**********************************/
	void Release(void);

/***********************************private***********************************/
private:

	/***********************************Init**********************************/
	int nRet;	
	API_COMMEN api_commen;
	caffe::shared_ptr<caffe::Net<float> > net_dl;

	/***********************************private**********************************/
	int bbox_regression( vector<RectInfo> vecRectInfo, int width, int height, vector<FasterRCNNInfo> &vecResInfo );
	int bbox_NMS(vector<FasterRCNNInfo> src, vector<FasterRCNNInfo> &dst, float overlap);

};

#endif

