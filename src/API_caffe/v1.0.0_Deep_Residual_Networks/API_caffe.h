#pragma once
#include <opencv/cv.h>
#include <vector>

#include "caffe/caffe.hpp"
#include <TErrorCode.h>

using namespace cv;
using namespace std;
using namespace caffe;

class API_CAFFE
{

/***********************************Common***********************************/
#define WIDTH 256
#define HEIGHT 256
#define BLOB_WIDTH 224
#define BLOB_HEIGHT 224
#define CHANNEL 3

#define NMAX 1000000
#define NMIN (-1000000)
#define NAMELEN 100

/***********************************Config***********************************/
const int LABEL_NUM[1] = {1000};	//number of deeplearning image label{1000class};
typedef unsigned long long UInt64;

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
		const char* 					DL_Meanfile, 		//[In]:DL_Meanfile
		const char* 					layerName, 			//[In]:layerName:"fc7"
		const int 						binGPU, 			//[In]:USE GPU(1) or not(0)
		const int 						deviceID );			//[In]:GPU ID

	/***********************************GetLabelFeat**********************************/
	int GetLabelFeat(
		const vector<Mat_<Vec3f> > 		img_dl,				//[In]:source image(s)
		UInt64							imgID,				//[In]:source imageID
		const char* 					layerName,			//[In]:Layer Name by Extract
		const int						bExtractFeat,		//[In]:Get Label(1),Extract Feat(2),both(3)
		vector< pair< int, float > >	&label, 			//[Out]:label--top 2 label
		vector< vector<float> >			&imgFeat);			//[Out]:imgFeat--need Normalization

	/***********************************GetLabelFeat_GoogleNet**********************************/
	int GetLabelFeat_GoogleNet(
		const vector<Mat_<Vec3f> > 		img_dl,				//[In]:source image(s)
		UInt64							imgID,				//[In]:source imageID
		const int 						labelNum,			//[In]:label Num
		const char* 					layerName,			//[In]:Layer Name by Extract
		const int						bExtractFeat,		//[In]:Get Label(1),Extract Feat(2),both(3)
		vector< pair< int, float > >	&label, 			//[Out]:ImgDetail
		vector< vector<float> >			&imgFeat);			//[Out]:imgFeat 
	
    /***********************************Release**********************************/
	void Release(void);

/***********************************private***********************************/
private:

	//net
	//shared_ptr<Net<float> > net_dl;
	caffe::shared_ptr<caffe::Net<float> > net_dl;
	Blob<float> mean_dl;

	int ReadImageToBlob( const vector<Mat_<Vec3f> > img_dl, int ModelMode, Blob<float>& image_blob);
	template <typename Dtype> bool DataMaxMin( const Dtype* d_from, int d_count, float& d_max, float& d_min );
	void blob2image(Blob<float>& image_blob, UInt64 imgID);
	float CalcL1Norm(const float* d_from, int d_count); 
	void info(Blob<float>& image_blob);
	int DL_FileList2LabelFeat( int ModelMode, const vector<Mat_<Vec3f> > img_dl, UInt64 ImageID,
		const long labelNum, const char* layerName, vector< pair< int, float > > &vecLabel, vector< vector<float> > &imgFeat);

};

