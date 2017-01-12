////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Applicable File Name: SD_global.h (SD:SimilarDetect)
// Editor: xiaogao
//
// Copyright (c) IN Inc(2016-2017)
// 
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef _IN_IMAGE_SIMILAR_DETECT_H_
#define _IN_IMAGE_SIMILAR_DETECT_H_

#include <string>

#include "ColorLayout.h"
#include "EdgeHist.h"
#include "API_imagequality.h"
#include "ImageProcessing.h"

using namespace std;

class IN_IMAGE_SIMILAR_DETECT_1_0_0
{

////////////////////////////////////////////////////////////////////////////
//configuration Macro definition
////////////////////////////////////////////////////////////////////////////
#define IN_CLD_COLOR_DIM		36 
#define IN_EHD_TEXTURE_DIM    	80
#define BLUR_TEXTURE_DIM    	5

#ifndef MABS
//fast abs of integer number
#define MABS(x)                (((x)+((x)>>31))^((x)>>31))     
#endif //MABS

typedef struct tagSD_Filter_Feature
{
	unsigned char quality;
	unsigned char feat_cld[IN_CLD_COLOR_DIM];
}SD_Filter_Feature;

typedef struct tagSD_Global_Feature
{
	unsigned char quality;
	unsigned char feat_cld[IN_CLD_COLOR_DIM];
	unsigned char feat_ehd[IN_EHD_TEXTURE_DIM];
}SD_Global_Feature;

/***********************************public***********************************/
public:

	/// construct function 
    IN_IMAGE_SIMILAR_DETECT_1_0_0();
    
	/// distruct function
	~IN_IMAGE_SIMILAR_DETECT_1_0_0(void);

	/***********************************A key Filter***********************************/

	/** Get_Feat_Score 
	 * @brief : get the input image feat and quality score .
	 * @return : 0(TOK) if succeeds, otherwise Error code is returned. (lookup in TErrorCode.h)
	 */
	int Filter_Get_Feat_Score(			 
		const unsigned char 	*pImage, 		//[In]input image data:h-w-c;			 
		const int 				width, 			//[In]input image width;			 
		const int 				height, 		//[In]input image height;			 
		const int 				nChannel,		//[In]input image channel;
		string		 			&Feat);			//[Out]quality score && image Feat;

	/** SimilarDetect
	 * @brief : detect the input feat for similar copy .
	 * @return : 0(TOK) if succeeds, otherwise Error code is returned. (lookup in TErrorCode.h)
	 */
	int Filter_SimilarDetect(
		const string 			Feat1, 				//[In]input image1 feat;
		const string 			Feat2, 				//[In]input image2 feat;
		int						&mode_same,			//[Out]1-same image;0-other;
		int						&mode_quality);		//[Out]2-Feat2 quality better;1-Feat2 quality better;0-error;

	/***********************************Photo Album***********************************/
	int Album_Get_Feat_Score(			 
		const unsigned char 	*pImage, 		//[In]input image data:h-w-c;			 
		const int 				width, 			//[In]input image width;			 
		const int 				height, 		//[In]input image height;			 
		const int 				nChannel,		//[In]input image channel;
		string		 			&Feat);			//[Out]quality score && image Feat;

	int Album_SimilarDetect(
		const string 			Feat1, 				//[In]input image1 feat;
		const string 			Feat2, 				//[In]input image2 feat;
		int						&mode_same,			//[Out]2-same image;1-similar image;0-other;
		int						&mode_quality);		//[Out]2-Feat2 quality better;1-Feat2 quality better;0-error;

	/***********************************Test***********************************/
	int Filter_SimilarDetect_Test(
		const string 			Feat1, 				//[In]input image1 feat;
		const string 			Feat2, 				//[In]input image2 feat;
		int						&mode_same,			//[Out]1-same image;0-other;
		int						&mode_quality,		//[Out]2-Feat2 quality better;1-Feat2 quality better;0-error;
		int 					&maxSingle, 
		int 					&maxSum);

	int Album_Get_Feat_Score_Test(			 
		const unsigned char 	*pImage, 		//[In]input image data:h-w-c;			 
		const int 				width, 			//[In]input image width;			 
		const int 				height, 		//[In]input image height;			 
		const int 				nChannel,		//[In]input image channel;
		string		 			&Feat,			//[Out]quality score && image Feat;
		vector<double>			&Time);			//[Out]Time
		
	//for Test
	int Album_SimilarDetect_Test(
		const string 			Feat1, 				//[In]input image1 feat;
		const string 			Feat2, 				//[In]input image2 feat;
		int						&mode_same,			//[Out]2-same image;1-similar image;0-other;
		int						&mode_quality,		//[Out]2-Feat2 quality better;1-Feat2 quality better;0-error;
		double					&dist_cld,			//[Out]
		double					&dist_ehd,			//[Out]
		double					&Dist);				//[Out]

/***********************************private***********************************/
private:

	int i,j,ret;
	GF_CLD_INTERNAL	 api_cld;
	GF_EHD_INTERNAL	 api_ehd;
	API_IMAGEQUALITY api_imagequality;
	API_IMAGEPROCESS api_imageprocess;

	RunTimer<double> run;

	/***********************************IsSimilar***********************************/
	bool Filter_IsSimilar_Test(unsigned char *query_feat, unsigned char *db_feat, int &maxSingle, int &maxSum);
	bool IsSimilar(unsigned char *query_feat, unsigned char *db_feat);
	/***********************************ch***********************************/
	int ColorHistogram( unsigned char *bgr, int width, int height, int channel, int numColorBlock, unsigned char &Entropy);
	int ExtractFeat_Entropy( float *inFeat, int len, float &Entropy );
	float * Normal_L2( float *inFeat, int len );
};

#endif 

