#pragma once
#include <vector>

using namespace std;

class API_MAINCOLOR_1_0_0
{

/***********************************Config***********************************/
typedef unsigned char uchar;

/***********************************public***********************************/
public:

	/// construct function 
    API_MAINCOLOR_1_0_0();
    
	/// distruct function
	~API_MAINCOLOR_1_0_0(void);

	/***********************************Predict**********************************/
	int Predict(
		uchar* 										bgr, 				//[In]:image->bgr
		int 										width, 				//[In]:image->width
		int 										height,				//[In]:image->height
		int 										channel,			//[In]:image->channel
		int 										numColorBlock,		//[In]:numColorBlock
		vector< pair< vector< int >,float > >		&Res);				//[Out]:Res:<[b,g,r],score>

/***********************************private***********************************/
private:

	/***********************************ColorHistogram***********************************/
	int ColorHistogram(
		uchar*										bgr,				//[In]:image->bgr
		int 										width,				//[In]:image->width
		int 										height, 			//[In]:image->height
		int 										channel,			//[In]:image->channel
		int 										numColorBlock,		//[In]:numColorBlock)
		vector< pair< vector< int >,float > >		&Res);				//[Out]:Res

	/***********************************gaussianFilter***********************************/
	void gaussianFilter(
		uchar* 										data, 				//[In]:image->data
		int 										width, 				//[In]:image->width
		int 										height,  			//[In]:image->height
		int 										channel);			//[In]:image->channel

};


	

