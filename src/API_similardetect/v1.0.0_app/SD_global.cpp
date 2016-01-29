#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <time.h>

#include <sys/types.h>
#include <dirent.h>
#include <sys/stat.h>
#include <pthread.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <list>
#include <map>
#include <algorithm>	
#include "TErrorCode.h"
#include "SD_global.h"

using namespace std;

//-------------------------------------------------------------------------

/***********************************Init*************************************/
/// construct function 
IN_IMAGE_SIMILAR_DETECT_1_0_0::IN_IMAGE_SIMILAR_DETECT_1_0_0()
{
}

/// destruct function 
IN_IMAGE_SIMILAR_DETECT_1_0_0::~IN_IMAGE_SIMILAR_DETECT_1_0_0(void)
{
}

//////////////////////////////////////////////////////////////////////////////////////
bool IN_IMAGE_SIMILAR_DETECT_1_0_0::IsSimilar(unsigned char *query_feat, unsigned char *db_feat)
{
	int nTmp, nSum = 0;
	for (i = 0; i < IN_CLD_COLOR_DIM; i++) {
		nTmp = query_feat[i];
		nTmp -= db_feat[i];
		nTmp = MABS(nTmp);
		if (nTmp > 10)	//nTmp:2
			return false;
		nSum += nTmp;
		if (nSum > 32)	//nSum:8
			return false;
	}
	return true;
}

bool IN_IMAGE_SIMILAR_DETECT_1_0_0::Filter_IsSimilar_Test(
	unsigned char *query_feat, unsigned char *db_feat, int &maxSingle, int &maxSum )
{
	int nTmp, nSum = 0;
	int T_nTmp = 10;		//T_nTmp:2
	int T_nSum = 32;		//T_nSum:12

	maxSingle = 0;
	maxSum = 0;
	
	for (i = 0; i < IN_CLD_COLOR_DIM; i++) {
		//Single
		nTmp = query_feat[i];
		nTmp -= db_feat[i];
		nTmp = MABS(nTmp);
		if (nTmp > T_nTmp)
		{
			maxSingle = T_nTmp+1;
			maxSum = T_nSum+1;
			return false;
		}
		if ( nTmp>maxSingle )
			maxSingle = nTmp;

		//sum
		nSum += nTmp;
		if (nSum > T_nSum)
		{
			maxSum = T_nSum+1;
			return false;
		}
	}
	
	maxSum = nSum;
	
	return true;
}



/***********************************A key Filter***********************************/

/** Get_Feat_Score 
 * @brief : get the input image feat and quality score .
 * @return : 0(TOK) if succeeds, otherwise Error code is returned. (lookup in TErrorCode.h)
 */
int IN_IMAGE_SIMILAR_DETECT_1_0_0::Filter_Get_Feat_Score(			 
	const unsigned char 	*pImage, 		//[In]input image data:h-w-c;			 
	const int 				width, 			//[In]input image width;			 
	const int 				height, 		//[In]input image height;			 
	const int 				nChannel,		//[In]input image channel;
	string		 			&Feat)			//[Out]quality score && image Feat;
{
	if ( (!pImage) || (width<32) || (height<32) || (nChannel!=3) )
		return TEC_INVALID_PARAM;

	//init
	ret=0;
	int bin_resize = 0;
	SD_Filter_Feature *pFeat = NULL;
	pFeat = new SD_Filter_Feature;
	if (!pFeat)
		return TEC_NO_MEMORY;

	memset(pFeat, 0, sizeof(pFeat));

	//resize
	unsigned char *ImageMedia = NULL;
	if (width != 256 || height != 256 )
	{
		ImageMedia = api_imageprocess.ImageResize(pImage, width, height, nChannel, 256, 256);
		bin_resize = 1;
	}
	else
		ImageMedia = (unsigned char *)pImage;

	//count feat
	api_cld.MultiBlock_LayoutExtractor(ImageMedia, 256, 256, 3, pFeat->feat_cld );

	//count ch
	ret = ColorHistogram(ImageMedia, 256, 256, 3, 4, pFeat->quality);
	if ( ret!=0 )
	{
		if (pFeat) delete pFeat;
		if ((bin_resize==1)&&(ImageMedia)) {delete [] ImageMedia;ImageMedia = NULL;}
		return TEC_INVALID_PARAM;
	}

	//merge result
	unsigned char *char_feat = new unsigned char[1+IN_CLD_COLOR_DIM];
	memset(char_feat, 0, (1+IN_CLD_COLOR_DIM) * sizeof(unsigned char));
	char_feat[0] = pFeat->quality;
	for(i=0;i<IN_CLD_COLOR_DIM; i++)
	{
		char_feat[i+1] = pFeat->feat_cld[i];
	}

	//jstring should [1,127]
	for(i=0;i<1+IN_CLD_COLOR_DIM; i++)
	{
		char_feat[i] = (char_feat[i]<1)?1:char_feat[i];
		char_feat[i] = (char_feat[i]>127)?127:char_feat[i];
	}
	
	Feat.assign((char*)(char_feat),(1+IN_CLD_COLOR_DIM));	//unsigned char * to string
	
	//delete
	if (pFeat) delete pFeat;
	if (char_feat) {delete [] char_feat;char_feat = NULL;}
	if ((bin_resize==1)&&(ImageMedia)) {delete [] ImageMedia;ImageMedia = NULL;}

	return TOK;
}


/** SimilarDetect
 * @brief : detect the input feat for similar copy .
 * @return : 0(TOK) if succeeds, otherwise Error code is returned. (lookup in TErrorCode.h)
 */
int IN_IMAGE_SIMILAR_DETECT_1_0_0::Filter_SimilarDetect(
	const string 			Feat1, 				//[In]input image1 feat;
	const string 			Feat2, 				//[In]input image2 feat;
	int						&mode_same,			//[Out]1-same image;0-other;
	int						&mode_quality)		//[Out]2-Feat2 quality better;1-Feat2 quality better;0-error;
{
	if ( (Feat1.size()!=37) || (Feat2.size()!=37) )
		return TEC_INVALID_PARAM;

	//init
	bool res_IsSimilar = false;
	double dist_cld,dist_ehd,Dist;
	double w1,w2;

	SD_Filter_Feature *pFeat = NULL;
	pFeat = new SD_Filter_Feature[2];
	if (!pFeat)
		return TEC_NO_MEMORY;

	memset(pFeat, 0, sizeof(pFeat));

	//change data
	char *feat1 = (char*)Feat1.c_str();
	unsigned char *pFeat1 = (unsigned char*)feat1;

	char *feat2 = (char*)Feat2.c_str();
	unsigned char *pFeat2 = (unsigned char*)feat2;

	//load data
	{
		pFeat[0].quality = pFeat1[0];
		pFeat[1].quality = pFeat2[0];
		for(j=0;j<IN_CLD_COLOR_DIM;j++)
		{
			pFeat[0].feat_cld[j] = pFeat1[1+j];
			pFeat[1].feat_cld[j] = pFeat2[1+j];
		}
	}

	//count similar image
	res_IsSimilar = IsSimilar(pFeat[0].feat_cld, pFeat[1].feat_cld);

	//merge result
	if ( ( res_IsSimilar == true ) )
		mode_same = 1;
	else
		mode_same = 0;

	if ( ( pFeat[1].quality > pFeat[0].quality ) && ( pFeat[0].quality > 0 ) && ( pFeat[1].quality > 0 ) )
		mode_quality = 2;
	else if ( ( pFeat[0].quality >= pFeat[1].quality ) && ( pFeat[0].quality > 0 ) && ( pFeat[1].quality > 0 ) )
		mode_quality = 1;
	else
		mode_quality = 0;
	 
	//delete
	if (pFeat) {delete [] pFeat;pFeat = NULL;}
	
	return TOK;
}

int IN_IMAGE_SIMILAR_DETECT_1_0_0::Filter_SimilarDetect_Test(
	const string 			Feat1, 				//[In]input image1 feat;
	const string 			Feat2, 				//[In]input image2 feat;
	int						&mode_same,			//[Out]1-same image;0-other;
	int						&mode_quality,		//[Out]2-Feat2 quality better;1-Feat2 quality better;0-error;
	int 					&maxSingle, 
	int 					&maxSum)
{
	if ( (Feat1.size()!=37) || (Feat2.size()!=37) )
		return TEC_INVALID_PARAM;

	//init
	bool res_IsSimilar = false;
	double dist_cld,dist_ehd,Dist;
	double w1,w2;

	SD_Filter_Feature *pFeat = NULL;
	pFeat = new SD_Filter_Feature[2];
	if (!pFeat)
		return TEC_NO_MEMORY;

	memset(pFeat, 0, sizeof(pFeat));

	//change data
	char *feat1 = (char*)Feat1.c_str();
	unsigned char *pFeat1 = (unsigned char*)feat1;

	char *feat2 = (char*)Feat2.c_str();
	unsigned char *pFeat2 = (unsigned char*)feat2;

	//load data
	{
		pFeat[0].quality = pFeat1[0];
		pFeat[1].quality = pFeat2[0];
		for(j=0;j<IN_CLD_COLOR_DIM;j++)
		{
			pFeat[0].feat_cld[j] = pFeat1[1+j];
			pFeat[1].feat_cld[j] = pFeat2[1+j];
		}
	}

	//count similar image
	res_IsSimilar = Filter_IsSimilar_Test(pFeat[0].feat_cld, pFeat[1].feat_cld,maxSingle,maxSum);

	//merge result
	if ( ( res_IsSimilar == true ) )
		mode_same = 1;
	else
		mode_same = 0;

	if ( ( pFeat[1].quality > pFeat[0].quality ) && ( pFeat[0].quality > 0 ) && ( pFeat[1].quality > 0 ) )
		mode_quality = 2;
	else if ( ( pFeat[0].quality >= pFeat[1].quality ) && ( pFeat[0].quality > 0 ) && ( pFeat[1].quality > 0 ) )
		mode_quality = 1;
	else
		mode_quality = 0;
	 
	//delete
	if (pFeat) {delete [] pFeat;pFeat = NULL;}
	
	return TOK;
}


int IN_IMAGE_SIMILAR_DETECT_1_0_0::Album_Get_Feat_Score(			 
	const unsigned char 	*pImage, 		//[In]input image data:h-w-c;			 
	const int 				width, 			//[In]input image width;			 
	const int 				height, 		//[In]input image height;			 
	const int 				nChannel,		//[In]input image channel;
	string		 			&Feat)			//[Out]quality score && image Feat;
{
	if ( (!pImage) || (width<32) || (height<32) || (nChannel!=3) )
		return TEC_INVALID_PARAM;

	//init
	ret=0;
	int bin_resize = 0;
	float res_quality = 0;
	SD_Global_Feature *pFeat = NULL;
	pFeat = new SD_Global_Feature;
	if (!pFeat)
		return TEC_NO_MEMORY;

	memset(pFeat, 0, sizeof(pFeat));

	//resize
	unsigned char *ImageMedia = NULL;
	if (width != 256 || height != 256 )
	{
		ImageMedia = api_imageprocess.ImageResize(pImage, width, height, nChannel, 256, 256);
		bin_resize = 1;
	}
	else
		ImageMedia = (unsigned char *)pImage;

	//count feat
	api_cld.MultiBlock_LayoutExtractor(ImageMedia, 256, 256, 3, pFeat->feat_cld );
	api_ehd.EdgeHistExtractor(ImageMedia, 256, 256, 3, pFeat->feat_ehd);

	//count blur
	//vector < float > fBlur;
	float tmpBlur = 0;
	//ret = api_imagequality.ExtractFeat_Blur(ImageMedia, 256, 256, 3, fBlur);
	ret = api_imagequality.ExtractFeat_Blur_test(ImageMedia, 256, 256, 3, tmpBlur);
	if ( ret!=0 )
	{
		if (pFeat) delete pFeat;
		if ((bin_resize==1)&&(ImageMedia)) {delete [] ImageMedia;ImageMedia = NULL;}
		return TEC_INVALID_PARAM;
	}

	//count ch
	ret = ColorHistogram(ImageMedia, 256, 256, 3, 4, pFeat->quality);
	if ( ret!=0 )
	{
		if (pFeat) delete pFeat;
		if ((bin_resize==1)&&(ImageMedia)) {delete [] ImageMedia;ImageMedia = NULL;}
		return TEC_INVALID_PARAM;
	}
	
	//count quality score
	//res_quality = pFeat->quality*7.0/(sqrt(fBlur[0]));
	res_quality = 0.5*tmpBlur + 0.5*pFeat->quality;
	res_quality = (res_quality>127)?127:res_quality;
	pFeat->quality = int( res_quality );	//tmp_Entropy*255/10

	//merge result
	unsigned char *char_feat = new unsigned char[1+IN_CLD_COLOR_DIM+IN_EHD_TEXTURE_DIM];
	memset(char_feat, 0, (1+IN_CLD_COLOR_DIM+IN_EHD_TEXTURE_DIM) * sizeof(unsigned char));
	char_feat[0] = pFeat->quality;
	for(i=0;i<IN_CLD_COLOR_DIM; i++)
		char_feat[i+1] = pFeat->feat_cld[i];
	for(i=0;i<IN_EHD_TEXTURE_DIM; i++)
		char_feat[i+1+IN_CLD_COLOR_DIM] = pFeat->feat_ehd[i];

	//jstring should [1,127]
	for(i=0;i<1+IN_CLD_COLOR_DIM+IN_EHD_TEXTURE_DIM; i++)
	{
		char_feat[i] = (char_feat[i]<1)?1:char_feat[i];
		char_feat[i] = (char_feat[i]>127)?127:char_feat[i];
	}
	
	Feat.assign((char*)(char_feat),(1+IN_CLD_COLOR_DIM+IN_EHD_TEXTURE_DIM));	//unsigned char * to string
	
	//delete
	if (pFeat) delete pFeat;
	if (char_feat) {delete [] char_feat;char_feat = NULL;}
	if ((bin_resize==1)&&(ImageMedia)) {delete [] ImageMedia;ImageMedia = NULL;}

	return TOK;
}

int IN_IMAGE_SIMILAR_DETECT_1_0_0::Album_SimilarDetect(
	const string 			Feat1, 				//[In]input image1 feat;
	const string 			Feat2, 				//[In]input image2 feat;
	int						&mode_same,			//[Out]2-same image;1-similar image;0-other;
	int						&mode_quality)		//[Out]2-Feat2 quality better;1-Feat2 quality better;0-error;
{
	if ( (Feat1.size()!=117) || (Feat2.size()!=117) )
		return TEC_INVALID_PARAM;

	//init
	bool res_IsSimilar = false;
	double dist_cld,dist_ehd,Dist;
	double w1,w2;

	SD_Global_Feature *pFeat = NULL;
	pFeat = new SD_Global_Feature[2];
	if (!pFeat)
		return TEC_NO_MEMORY;

	memset(pFeat, 0, sizeof(pFeat));

	//change data
	char *feat1 = (char*)Feat1.c_str();
	unsigned char *pFeat1 = (unsigned char*)feat1;

	char *feat2 = (char*)Feat2.c_str();
	unsigned char *pFeat2 = (unsigned char*)feat2;

	//load data
	{
		pFeat[0].quality = pFeat1[0];
		pFeat[1].quality = pFeat2[0];
		for(j=0;j<IN_CLD_COLOR_DIM;j++)
		{
			pFeat[0].feat_cld[j] = pFeat1[1+j];
			pFeat[1].feat_cld[j] = pFeat2[1+j];
		}
		for(j=0;j<IN_EHD_TEXTURE_DIM;j++)
		{
			pFeat[0].feat_ehd[j] = pFeat1[1+IN_CLD_COLOR_DIM+j];
			pFeat[1].feat_ehd[j] = pFeat2[1+IN_CLD_COLOR_DIM+j];
		}
	}

	//count distance
	dist_cld = 10000;
	dist_ehd = 10000;
	dist_cld = api_cld.CLDDist( pFeat[0].feat_cld, pFeat[1].feat_cld );
	dist_ehd = api_ehd.EHDDist( pFeat[0].feat_ehd, pFeat[1].feat_ehd );
	//count similar image
	res_IsSimilar = IsSimilar(pFeat[0].feat_cld, pFeat[1].feat_cld);

	//merge result
	if ( ( res_IsSimilar == true ) )
		mode_same = 2;
	else if ( ( dist_cld <= 30 ) && ( dist_ehd <= 170 ) )
		mode_same = 1;
	else
		mode_same = 0;

	if ( ( pFeat[1].quality > pFeat[0].quality ) && ( pFeat[0].quality > 0 ) && ( pFeat[1].quality > 0 ) )
		mode_quality = 2;
	else if ( ( pFeat[0].quality >= pFeat[1].quality ) && ( pFeat[0].quality > 0 ) && ( pFeat[1].quality > 0 ) )
		mode_quality = 1;
	else
		mode_quality = 0;
	 
	//delete
	if (pFeat) {delete [] pFeat;pFeat = NULL;}
	
	return TOK;
}

int IN_IMAGE_SIMILAR_DETECT_1_0_0::Album_Get_Feat_Score_Test(			 
	const unsigned char 	*pImage, 		//[In]input image data:h-w-c;			 
	const int 				width, 			//[In]input image width;			 
	const int 				height, 		//[In]input image height;			 
	const int 				nChannel,		//[In]input image channel;
	string		 			&Feat,			//[Out]quality score && image Feat;
	vector<double>			&Time)			//[Out]Time
{
	if ( (!pImage) || (width<32) || (height<32) || (nChannel!=3) )
		return TEC_INVALID_PARAM;

	//init
	ret=0;
	int bin_resize = 0;
	float res_quality = 0;
	SD_Global_Feature *pFeat = NULL;
	pFeat = new SD_Global_Feature;
	if (!pFeat)
		return TEC_NO_MEMORY;

	memset(pFeat, 0, sizeof(pFeat));
	Time.clear();

	//resize
	unsigned char *ImageMedia = NULL;
	if (width != 256 || height != 256 )
	{
		ImageMedia = api_imageprocess.ImageResize(pImage, width, height, nChannel, 256, 256);
		bin_resize = 1;
	}
	else
		ImageMedia = (unsigned char *)pImage;

	//count feat
	run.start();
	api_cld.MultiBlock_LayoutExtractor(ImageMedia, 256, 256, 3, pFeat->feat_cld );
	run.end();
	Time.push_back(double(run.time()*1000.0));

	run.start();
	api_ehd.EdgeHistExtractor(ImageMedia, 256, 256, 3, pFeat->feat_ehd);
	run.end();
	Time.push_back(double(run.time()*1000.0));

	//count blur
	vector < float > fBlur;
	float tmpBlur = 0;
	run.start();
	//ret = api_imagequality.ExtractFeat_Blur(ImageMedia, 256, 256, 3, fBlur);
	ret = api_imagequality.ExtractFeat_Blur_test(ImageMedia, 256, 256, 3, tmpBlur);
	run.end();
	Time.push_back(double(run.time()*1000.0));
	if ( ret!=0 )
	{
		if (pFeat) delete pFeat;
		if ((bin_resize==1)&&(ImageMedia)) {delete [] ImageMedia;ImageMedia = NULL;}
		return TEC_INVALID_PARAM;
	}

	//count ch
	run.start();
	ret = ColorHistogram(ImageMedia, 256, 256, 3, 4, pFeat->quality);
	run.end();
	Time.push_back(double(run.time()*1000.0));
	if ( ret!=0 )
	{
		if (pFeat) delete pFeat;
		if ((bin_resize==1)&&(ImageMedia)) {delete [] ImageMedia;ImageMedia = NULL;}
		return TEC_INVALID_PARAM;
	}
	
	//count quality score
	//res_quality = pFeat->quality*7.0/(sqrt(fBlur[0]));
	res_quality = 0.5*tmpBlur + 0.5*pFeat->quality;
	res_quality = (res_quality>127)?127:res_quality;
	pFeat->quality = int( res_quality );
	//printf("blur:%.2f_%d\n",tmpBlur,pFeat->quality);

	//merge result
	unsigned char *char_feat = new unsigned char[1+IN_CLD_COLOR_DIM+IN_EHD_TEXTURE_DIM];
	memset(char_feat, 0, (1+IN_CLD_COLOR_DIM+IN_EHD_TEXTURE_DIM) * sizeof(unsigned char));
	char_feat[0] = pFeat->quality;
	for(i=0;i<IN_CLD_COLOR_DIM; i++)
		char_feat[i+1] = pFeat->feat_cld[i];
	for(i=0;i<IN_EHD_TEXTURE_DIM; i++)
		char_feat[i+1+IN_CLD_COLOR_DIM] = pFeat->feat_ehd[i];

	//jstring should [1,127]
	for(i=0;i<1+IN_CLD_COLOR_DIM+IN_EHD_TEXTURE_DIM; i++)
	{
		char_feat[i] = (char_feat[i]<1)?1:char_feat[i];
		char_feat[i] = (char_feat[i]>127)?127:char_feat[i];
	}
	
	Feat.assign((char*)(char_feat),(1+IN_CLD_COLOR_DIM+IN_EHD_TEXTURE_DIM));	//unsigned char * to string

	/*	if (char_feat[0]>64 )
	{
		//check data
		printf("char_feat:");
		for(i=0;i<117;i++)
		{
			printf("%d-%d ",i,char_feat[i]);
			if((i+1)%20==0)
				printf("\n");
		}
		printf("\n");

		//check data
		char *feat1 = (char*)Feat.c_str();
		unsigned char *pFeat1 = (unsigned char*)feat1;
		printf("pFeat1:");
		for(i=0;i<117;i++)
		{
			printf("%d-%d ",i,pFeat1[i]);
			if((i+1)%20==0)
				printf("\n");
		}
		printf("\n");
	}*/
	
	//delete
	if (pFeat) delete pFeat;
	if (char_feat) {delete [] char_feat;char_feat = NULL;}
	if ((bin_resize==1)&&(ImageMedia)) {delete [] ImageMedia;ImageMedia = NULL;}

	return TOK;
}

int IN_IMAGE_SIMILAR_DETECT_1_0_0::Album_SimilarDetect_Test(
		const string 			Feat1, 				//[In]input image1 feat;
		const string 			Feat2, 				//[In]input image2 feat;
		int						&mode_same,			//[Out]2-same image;1-similar image;0-other;
		int						&mode_quality,		//[Out]2-Feat2 quality better;1-Feat2 quality better;0-error;
		double					&dist_cld,			//[Out]
		double					&dist_ehd,			//[Out]
		double					&Dist)				//[Out]
{
	if ( (Feat1.size()!=117) || (Feat2.size()!=117) )
		return TEC_INVALID_PARAM;

	//init
	bool res_IsSimilar = false;
	double w1,w2;

	SD_Global_Feature *pFeat = NULL;
	pFeat = new SD_Global_Feature[2];
	if (!pFeat)
		return TEC_NO_MEMORY;

	memset(pFeat, 0, sizeof(pFeat));
	
	//change data
	char *feat1 = (char*)Feat1.c_str();
	unsigned char *pFeat1 = (unsigned char*)feat1;

	char *feat2 = (char*)Feat2.c_str();
	unsigned char *pFeat2 = (unsigned char*)feat2;
	
	//load data
	{
		pFeat[0].quality = pFeat1[0];
		pFeat[1].quality = pFeat2[0];
		for(j=0;j<IN_CLD_COLOR_DIM;j++)
		{
			pFeat[0].feat_cld[j] = pFeat1[1+j];
			pFeat[1].feat_cld[j] = pFeat2[1+j];
		}
		for(j=0;j<IN_EHD_TEXTURE_DIM;j++)
		{
			pFeat[0].feat_ehd[j] = pFeat1[1+IN_CLD_COLOR_DIM+j];
			pFeat[1].feat_ehd[j] = pFeat2[1+IN_CLD_COLOR_DIM+j];
		}
	}
	
	//count distance
	dist_cld = 10000;
	dist_ehd = 10000;
	dist_cld = api_cld.CLDDist( pFeat[0].feat_cld, pFeat[1].feat_cld );
	dist_ehd = api_ehd.EHDDist( pFeat[0].feat_ehd, pFeat[1].feat_ehd );
	//count similar image
	res_IsSimilar = IsSimilar(pFeat[0].feat_cld, pFeat[1].feat_cld);
	
	//merge result
	if ( ( res_IsSimilar == true ) )
		mode_same = 2;
	else if ( ( dist_cld <= 30 ) && ( dist_ehd <= 170 ) )
		mode_same = 1;
	else
		mode_same = 0;

	if ( ( pFeat[1].quality > pFeat[0].quality ) && ( pFeat[0].quality > 0 ) && ( pFeat[1].quality > 0 ) )
		mode_quality = 2;
	else if ( ( pFeat[0].quality >= pFeat[1].quality ) && ( pFeat[0].quality > 0 ) && ( pFeat[1].quality > 0 ) )
		mode_quality = 1;
	else
		mode_quality = 0;
	 
	//delete
	if (pFeat) {delete [] pFeat;pFeat = NULL;}
	
	return TOK;
}


int IN_IMAGE_SIMILAR_DETECT_1_0_0::ColorHistogram(	
	unsigned char			*bgr,				//[In]:image->bgr	
	int 					width,				//[In]:image->width	
	int 					height, 			//[In]:image->height	
	int 					channel,			//[In]:image->channel	
	int 					numColorBlock,		//[In]:numColorBlock
	unsigned char			&Entropy)			//[Out]:Entropy
{
	ret = TOK;
	int k,count,tmpPixel,feat_len,index;
	float tmp_Entropy = 0;
	Entropy = 0;

	int blockPixel = int( 256.0/numColorBlock + 0.5 );
	double size = 1.0/ (height * width) ;

	feat_len = numColorBlock*numColorBlock*numColorBlock;
	float *pCH = new float[feat_len];
	memset(pCH, 0, feat_len * sizeof(float));

	for (i = 0; i < height; i++) 
	{
		for (j = 0; j < width; j++)
		{
			index = 0;
			for(k=0;k<channel;k++)
			{
				tmpPixel = bgr[i*width*channel+j*channel+k];
				index = int(tmpPixel*1.0/blockPixel);	//norm	
				index >> 2;
			}
			pCH[index] += 1;
		}
	}

	//norm
	for (i = 0; i < feat_len; i++) 
	{
		pCH[i] = pCH[i] * size ;
	}

	//ExtractFeat_Entropy
	ret = ExtractFeat_Entropy( pCH, feat_len, tmp_Entropy );
	if ( ret!=0 )
	{
		if (pCH) {delete [] pCH;pCH = NULL;}
		return TEC_INVALID_PARAM;
	}

	tmp_Entropy = (tmp_Entropy>2)?2:tmp_Entropy;
	Entropy = int( tmp_Entropy*63.5 + 0.5 );	//tmp_Entropy*127/2:max-127
	
	if (pCH) {delete [] pCH;pCH = NULL;}
	
    return ret;
}

// calculate entropy of an image
int IN_IMAGE_SIMILAR_DETECT_1_0_0::ExtractFeat_Entropy( float *inFeat, int len, float &Entropy )
{
    if( (!inFeat) || (len<1) )
    {
        return TEC_INVALID_PARAM;
    }

    ret = TOK;

	float *NormFeat = Normal_L2( inFeat, len );

    Entropy = 0;
    for(i =0;i<len;i++)
    {
        if(NormFeat[i]>0)
            Entropy = Entropy-NormFeat[i]*(log(NormFeat[i])/log(2.0));
    }

	if (NormFeat) {delete [] NormFeat;NormFeat = NULL;}

    return ret;
}

float * IN_IMAGE_SIMILAR_DETECT_1_0_0::Normal_L2( float *inFeat, int len )
{
	if( (!inFeat) || (len<1) )
		return NULL;
	
    double Sum = 0.0;
	float *NormFeat = new float[len];
	memset(NormFeat, 0, len * sizeof(float));

    /************************Normalization*****************************/
    for ( j=0;j < len;j++ )
    {
        Sum += pow(inFeat[j],2);
    }
    Sum = sqrt(Sum);

    /************************Normalization*****************************/
    for (j = 0; j < len; j++)
    {
        if ( Sum == 0 )
            NormFeat[j] = 0;
        else
            NormFeat[j] = inFeat[j]*1.0/Sum;
    }

    return NormFeat;
}



