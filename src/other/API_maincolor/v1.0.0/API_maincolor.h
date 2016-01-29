#pragma once
#include <vector>
#include <cv.h>

#include "API_commen.h"

using namespace cv;
using namespace std;

class API_MAINCOLOR
{

/***********************************Config***********************************/
typedef unsigned char uchar;
typedef unsigned long long UInt64;

#define WIDTH 256
#define HEIGHT 256

/***********************************public***********************************/
public:

	/// construct function 
    API_MAINCOLOR();
    
	/// distruct function
	~API_MAINCOLOR(void);

	/***********************************Predict**********************************/
	int Predict(
		uchar* 										bgr, 				//[In]:image->bgr
		int 										width, 				//[In]:image->width
		int 										height,				//[In]:image->height
		int 										channel,			//[In]:image->channel
		UInt64										ImageID,			//[In]:ImageID
		int 										numColorBlock,		//[In]:numColorBlock
		vector< pair< vector< int >,float > >		&Res);				//[Out]:Res

	/***********************************Predict**********************************/
	int Predict_ipl(
		IplImage									*image, 			//[In]:image
		UInt64										ImageID,			//[In]:ImageID
		int 										numColorBlock,		//[In]:numColorBlock
		vector< pair< vector< int >,float > >		&Res);				//[Out]:Res

/***********************************private***********************************/
private:
	API_COMMEN 		api_commen;

	int ColorHistogram(
		uchar*										bgr,				//[In]:image->bgr
		int 										width,				//[In]:image->width
		int 										height, 			//[In]:image->height
		int 										channel,			//[In]:image->channel
		UInt64										ImageID,			//[In]:ImageID
		int 										numColorBlock,		//[In]:numColorBlock)
		vector< pair< vector< int >,float > >		&Res);				//[Out]:Res

	int ColorHistogram_ipl(
		IplImage 									*image, 			//[In]:image
		UInt64										ImageID,			//[In]:ImageID
		int 										numColorBlock,		//[In]:numColorBlock)
		vector< pair< vector< int >,float > >		&Res);				//[Out]:Res
	
	void gaussianFilter(
		uchar* 										data, 				//[In]:image->data
		int 										width, 				//[In]:image->width
		int 										height,  			//[In]:image->height
		int 										channel);			//[In]:image->channel

};


	

