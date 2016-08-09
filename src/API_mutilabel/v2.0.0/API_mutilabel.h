/*
 * =====================================================================================
 *
 *       filename:  API_mutilabel.h
 *
 *    description:  mutilabel detect interface
 *
 *        version:  1.0
 *        created:  2016-06-20
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


#ifndef _API_MUTILABEL_H_
#define _API_MUTILABEL_H_

#include <vector>
#include <cv.h>

#include "API_commen.h"
#include "API_caffe_mutilabel.h"

using namespace cv;
using namespace std;

enum ENUM_LOG
{
	enum_module_in_logo=1
};

struct MutiLabelInfo
{
	string 			label;
    float 			score;
    Vec4i 			rect;
	vector<float>	feat;
};

class API_MUTI_LABEL
{

/***********************************Common***********************************/
#define IMAGE_SIZE 512	//320:30ms,512:50ms,720:80ms

/***********************************public***********************************/
public:

	/// construct function 
    API_MUTI_LABEL();
    
	/// distruct function
	~API_MUTI_LABEL(void);

	/***********************************Init*************************************/
	int Init( 
		const char* 	KeyFilePath,						//[In]:KeyFilePath
		const int 		binGPU, 							//[In]:USE GPU(1) or not(0)
		const int 		deviceID );							//[In]:GPU ID

	/***********************************Predict**********************************/
	int Predict(
		IplImage					*image, 				//[In]:image
		string						imageURL,				//[In]:image URL for CheckData
		unsigned long long			imageID,				//[In]:image ID for CheckData
		unsigned long long			childID,				//[In]:image child ID for CheckData
		float						MutiLabel_T,			//[In]:Prdict_0.8,ReSample_0.9
		vector< MutiLabelInfo >		&Res);					//[Out]:Res

	/***********************************Release**********************************/
	void Release();

/***********************************private***********************************/
private:

	/***********************************Init**********************************/
	int i,j,nRet;
	RunTimer<double> 	run;
	vector< string > 	dic;
	
	/***********************************Init**********************************/
	API_COMMEN 				api_commen;
	API_CAFFE_FasterRCNN	api_caffe_FasterRCNN;

	/***********************************MergeVOC20classLabel**********************************/
	int Merge(
		IplImage						*image,			//[In]:image
		float							MutiLabel_T,	//[In]:Prdict_0.8,ReSample_0.9
		vector< FasterRCNNInfo >		inImgLabel, 	//[In]:ImgDetail from inImgLabel
		vector< MutiLabelInfo >			&LabelInfo);	//[Out]:LabelInfo

};

#endif

	

