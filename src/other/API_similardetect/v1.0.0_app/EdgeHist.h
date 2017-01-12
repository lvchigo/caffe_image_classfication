#ifndef _GF_EHD_INTERNAL_H_
#define _GF_EHD_INTERNAL_H_

#include <iostream>
#include "ImageProcessing.h"

using namespace std;

class GF_EHD_INTERNAL
{

#define		Te_Define				11
#define		Desired_Num_of_Blocks	1100
#define     EDGE_DIM             	80 
#define     TOTAL_EHD_DIM  			150 

#define 	WIDTH	256
#define 	HEIGHT	256

typedef	struct Edge_Histogram_Descriptor{
	double Local_Edge[80]; 
} EHD;

#define	NoEdge						0
#define	vertical_edge				1
#define	horizontal_edge				2
#define	non_directional_edge		3
#define	diagonal_45_degree_edge		4
#define	diagonal_135_degree_edge	5

//EHD	m_pEdge_Histogram;
//char m_pEdge_HistogramElement[80];

double QuantTable[5][8] = { 
	{0.010867,0.057915,0.099526,0.144849,0.195573,0.260504,0.358031,0.530128}, 
	{0.012266,0.069934,0.125879,0.182307,0.243396,0.314563,0.411728,0.564319},
	{0.004193,0.025852,0.046860,0.068519,0.093286,0.123490,0.161505,0.228960},
	{0.004174,0.025924,0.046232,0.067163,0.089655,0.115391,0.151904,0.217745},
	{0.006778,0.051667,0.108650,0.166257,0.224226,0.285691,0.356375,0.450972},
};

/***********************************public***********************************/
public:

	/// construct function 
	GF_EHD_INTERNAL();
	
	/// distruct function
	~GF_EHD_INTERNAL(void);

	//Extracting MPEG-7 EHD
	void EdgeHistExtractor(unsigned char *src, int width, int height, int nChannel, unsigned char EHD[]);

	//Computing Similarity of 2 EHD
	double EHDDist(unsigned char EHD1[], unsigned char EHD2[]);

/***********************************private***********************************/
private:

	API_IMAGEPROCESS api_imageprocess;
	
	/***********************************private***********************************/
	int StartExtracting(unsigned char *MediaData, int width, int height, int nChannel, unsigned char *EHD_FV);
	unsigned long GetBlockSize(unsigned long image_width, unsigned long image_height, unsigned long desired_num_of_blocks);
	void EdgeHistogramGeneration(unsigned char* pImage_Y, unsigned long image_width, unsigned long image_height, unsigned long block_size, EHD* pLocal_Edge, int Te_Value);
	int GetEdgeFeature(unsigned char *pImage_Y, int image_width, int block_size, int Te_Value);
	void SetEdgeHistogram(EHD* pEdge_Histogram, unsigned char *pEdge_HistogramElement);
	void Make_Global_SemiGlobal(int *LocalHistogramOnly, int *TotalHistogram);

};

#endif

