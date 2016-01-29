#ifndef _API_IMAGEQUALITY_H_
#define _API_IMAGEQUALITY_H_
#include <vector>
#include <iostream>
#include "ImageProcessing.h"

using namespace std;

class API_IMAGEQUALITY
{

/***********************************public***********************************/
public:

	/// construct function 
    API_IMAGEQUALITY();
    
	/// distruct function
	~API_IMAGEQUALITY(void);

	/***********************************ExtractFeat_Blur**********************************/
	int ExtractFeat_Blur( unsigned char *pSrcImg, int width, int height, int nChannel, vector< float > &fBlur );	//9*5D

	int ExtractFeat_Blur_test( unsigned char *pSrcImg, int width, int height, int nChannel, float &fBlur );

/***********************************private***********************************/
private:

	int i,j,ret;
	API_IMAGEPROCESS api_imageprocess;
	
	/***********************************Extract Blur Feat**********************************/
	//refernence paper:No-reference Image Quality Assessment using blur and noisy

	//50ms
	int ExtractFeat_Blur_Block( unsigned char *pSrcImg, int width, int height, int nChannel, vector< float > &fBlur );	//5D

	//40ms
	int ExtractFeat_Blur_Block_App( unsigned char *pSrcImg, int width, int height, int nChannel, vector< float > &fBlur );//5D
};

#endif
	

