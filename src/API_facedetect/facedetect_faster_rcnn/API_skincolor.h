#pragma once
#include <vector>
#include <cv.h>

using namespace cv;
using namespace std;

class API_SKINCOLOR
{

/***********************************Config***********************************/


/***********************************public***********************************/
public:

	/// construct function 
    API_SKINCOLOR();
    
	/// distruct function
	~API_SKINCOLOR(void);

	/***********************************Init*************************************/
	void SkinRGB(IplImage* rgb,IplImage* _dst);
	void cvSkinRG(IplImage* rgb,IplImage* gray);
	void cvThresholdOtsu(IplImage* src, IplImage* dst);
	void cvSkinOtsu(IplImage* src, IplImage* dst);
	void cvSkinYUV(IplImage* src,IplImage* dst);
	void cvSkinHSV(IplImage* src,IplImage* dst);

	//Process
	int Process_SkinColor(
		IplImage						*image,
		string 							ImageID,
		vector< pair<float,Vec4i> > 	inputFaceInfo,
		vector< pair<float,Vec4i> > 	&outputFaceInfo);

	int Process_UnNormalFace(
		IplImage						*image,
		string 							ImageID,
		vector< pair<float,Vec4i> > 	inputFaceInfo,
		vector< pair<float,Vec4i> > 	&outputFaceInfo);

/***********************************private***********************************/
private:

	


};


	

