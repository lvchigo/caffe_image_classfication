#pragma once
#include <vector>
#include <cv.h>

using namespace cv;
using namespace std;

class API_SKINCOLOR
{

/***********************************Config***********************************/
#define PCA_DIMS 			1000
#define PCA_RATIO 			0.9

#define PCA_MEAN			"mean"
#define PCA_EIGEN_VECTOR	"eigen_vector"

/***********************************public***********************************/
public:

	/// construct function 
    API_SKINCOLOR();
    
	/// distruct function
	~API_SKINCOLOR(void);

	/***********************************Init*************************************/
	void PCA_Feat_Init( const char *PCAModel );

	/***********************************Learn*************************************/
	int PCA_Feat_Learn( vector< pair< int, vector<float> > > inFeat, const char *outPCAModel );

	/***********************************Encode*************************************/
	int PCA_Feat_Encode( vector<float> inFeat, vector<float> &EncodeFeat );

	/***********************************Decode*************************************/
	int PCA_Feat_Decode( vector<float> EncodeFeat, const int DecodeDim, vector<float> &DecodeFeat );

	/***********************************Release*************************************/
	void PCA_Feat_Release();

/***********************************private***********************************/
private:

	PCA *pca_encoding;


};


	

