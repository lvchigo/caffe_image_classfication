#include <stdio.h>
//#include <math.h>
#include <cmath>
#include <memory.h>

#include "EdgeHist.h"
using namespace std;

/***********************************Init*************************************/
/// construct function 
GF_EHD_INTERNAL::GF_EHD_INTERNAL()
{
}

/// destruct function 
GF_EHD_INTERNAL::~GF_EHD_INTERNAL(void)
{
}

int GF_EHD_INTERNAL::StartExtracting(unsigned char *MediaData, int width, int height, int nChannel, unsigned char *EHD_FV)
{   
	unsigned long	desired_num_of_blocks;
	unsigned long	block_size;
	int		       Te_Value;

	EHD		       *pLocal_Edge = NULL;
	pLocal_Edge = new  EHD[1];
	memset(pLocal_Edge, 0, sizeof(pLocal_Edge));

	unsigned char *m_pEdge_HistogramElement = new unsigned char[80];
	memset(m_pEdge_HistogramElement, 0, 80 * sizeof(unsigned char));

	Te_Value              = Te_Define;  //11
	desired_num_of_blocks = Desired_Num_of_Blocks; //1100

	//////////////////////// Making Gray Image///////////////
	int i, j, xsize, ysize;
	unsigned char   *pResampleImage = NULL;
	int             max_x = 0, max_y = 0; 
	int             min_x = width-1; 
	int             min_y = height - 1;
	double          scale, EWweight, NSweight, EWtop, EWbottom;
	unsigned char   NW, NE, SW, SE;
	int            min_size, re_xsize, re_ysize;

	xsize = width;
	ysize = height;
	min_x = min_y = 0;
	int nSize = xsize * ysize;
	unsigned char *pGrayImage = new unsigned char[nSize];
	memset(pGrayImage, 0, nSize); 

	unsigned char *pImgData = MediaData;
	int nImgSize = width * height;
	int counter = 0;
	unsigned char b = 0, g = 0, r = 0;

	for(int i = 0; i < nImgSize; i++)
	{
		b = (unsigned char)(pImgData[counter]);
		g = (unsigned char)(pImgData[counter+1]);
		r = (unsigned char)(pImgData[counter+2]);
		pGrayImage[i] = (unsigned char)((r + g + b) / 3.0);
		counter += 3;
	}

	min_size = (xsize > ysize)? ysize: xsize;  //

	if(min_size < 70)  //不会进去的 
	{
		///////////////////////////////////upsampling///////////////////////////
		scale = 70.0 / min_size;
		re_xsize = (int)(xsize * scale + 0.5);
		re_ysize = (int)(ysize * scale + 0.5);
		pResampleImage = new unsigned char[re_xsize*re_ysize];
		for(j=0; j<re_ysize; j++)for(i=0; i<re_xsize; i++)
		{
			EWweight = i/scale-floor(i/scale);
			NSweight = j/scale-floor(j/scale);
			NW = pGrayImage[(int)floor(i/scale)+(int)floor(j/scale)*xsize];
			NE = pGrayImage[(int)floor(i/scale)+1+(int)floor(j/scale)*xsize];
			SW = pGrayImage[(int)floor(i/scale)+(int)(floor(j/scale)+1)*xsize];
			SE = pGrayImage[(int)floor(i/scale)+1+(int)(floor(j/scale)+1)*xsize];
			EWtop = NW + EWweight*(NE-NW);
			EWbottom = SW + EWweight*(SE-SW);
			pResampleImage[i+j*re_xsize] = (unsigned char)(EWtop + NSweight*(EWbottom-EWtop)+0.5);
		}

		block_size = GetBlockSize(re_xsize, re_ysize, desired_num_of_blocks);
		if(block_size<2)
			block_size = 2;
		EdgeHistogramGeneration(pResampleImage, re_xsize, re_ysize, block_size, pLocal_Edge, Te_Value);
		delete  [] pResampleImage;
	}
	else
	{
		block_size = GetBlockSize(xsize, ysize, desired_num_of_blocks);
		if(block_size < 2)
			block_size = 2;
		EdgeHistogramGeneration(pGrayImage, xsize, ysize, block_size, pLocal_Edge, Te_Value);
	}

	SetEdgeHistogram( pLocal_Edge, m_pEdge_HistogramElement);

	for( i = 0; i < 80; i++)
	{
		EHD_FV[i] = (unsigned char)(m_pEdge_HistogramElement[i]);
	}

	delete	[] pLocal_Edge;
	delete	[] pGrayImage;

	return 0; 
	//return m_pEdge_HistogramElement;
}

//------- 获取块的大小---------------------------------------------------------------------
unsigned long GF_EHD_INTERNAL::GetBlockSize(unsigned long image_width, unsigned long image_height, unsigned long desired_num_of_blocks)
{
	unsigned long	block_size;
	double		temp_size;

	temp_size = (double) sqrt((double)(image_width * image_height / desired_num_of_blocks));
	block_size = (unsigned long) (temp_size / 2) * 2;

	return block_size;
}

//--------------------------获取边缘特征直方图--------------------------------------------------
void GF_EHD_INTERNAL::EdgeHistogramGeneration(
	unsigned char* pImage_Y, 
	unsigned long image_width,
	unsigned long image_height,
	unsigned long block_size,
	EHD* pLocal_Edge, 
	int Te_Value)
{
	int Count_Local[16],sub_local_index;
	int Offset, EdgeTypeOfBlock;
	unsigned int i, j;
	long	LongTyp_Local_Edge[80];

	// Clear
	memset(Count_Local, 0, 16*sizeof(int));		
	memset(LongTyp_Local_Edge, 0, 80*sizeof(long));

	for(j = 0; j <= image_height - block_size; j += block_size)
	{
		for(i = 0; i <= image_width - block_size; i += block_size)
		{
			sub_local_index = (int)(i * 4 / image_width) + (int)(j * 4 / image_height) * 4;
			Count_Local[sub_local_index]++;

			Offset = image_width * j + i;

			EdgeTypeOfBlock = GetEdgeFeature(pImage_Y + Offset, image_width,
				block_size, Te_Value);
			switch(EdgeTypeOfBlock) {
			case NoEdge:
				break;
			case vertical_edge:
				LongTyp_Local_Edge[sub_local_index*5]++;
				break;
			case horizontal_edge:
				LongTyp_Local_Edge[sub_local_index*5+1]++;
				break;
			case diagonal_45_degree_edge:
				LongTyp_Local_Edge[sub_local_index*5+2]++;
				break;
			case diagonal_135_degree_edge:
				LongTyp_Local_Edge[sub_local_index*5+3]++;
				break;
			case non_directional_edge:
				LongTyp_Local_Edge[sub_local_index*5+4]++;
				break;
			} //switch(EdgeTypeOfBlock)
		} // for( i )
	}

	for( i = 0; i < 80; i++)
	{			// Range 0.0 ~ 1.0
		sub_local_index = (int)(i/5); // 0-16个块
		pLocal_Edge->Local_Edge[i] =
			(double)LongTyp_Local_Edge[i]/Count_Local[sub_local_index];
	}
}

//----------------------------------计算边缘特征-------------------------------------------------------------------------
int GF_EHD_INTERNAL::GetEdgeFeature(
	unsigned char *pImage_Y,
	int image_width,
	int block_size,
	int Te_Value)
{
	int i, j;
	double	d1, d2, d3, d4;
	int e_index;
	double dc_th = Te_Value;
	double e_h, e_v, e_45, e_135, e_m, e_max;

	d1=0.0;
	d2=0.0;
	d3=0.0;
	d4=0.0;
	// 统计块的四个子块上的像素和 
	for(j = 0; j < block_size; j++)
	{
		for(i=0; i<block_size; i++)
		{
			if(j<block_size/2)
			{
				if(i<block_size/2)
					d1+=(pImage_Y[i+image_width*j]);
				else
					d2+=(pImage_Y[i+image_width*j]);
			}
			else
			{
				if(i<block_size/2)
					d3+=(pImage_Y[i+image_width*j]);
				else
					d4+=(pImage_Y[i+image_width*j]);
			}
		}
	}

	//求均值
	d1 = d1/(block_size*block_size/4.0);
	d2 = d2/(block_size*block_size/4.0);
	d3 = d3/(block_size*block_size/4.0);
	d4 = d4/(block_size*block_size/4.0);

	//计算差值，即块的梯度值
	e_h = fabs(d1+d2-(d3+d4));
	e_v = fabs(d1+d3-(d2+d4));
	e_45 = sqrt((float)2)*fabs(d1-d4);
	e_135 = sqrt((float)2)*fabs(d2-d3);

	e_m = 2*fabs(d1-d2-d3+d4);

	e_max = e_v;
	e_index = vertical_edge;
	if(e_h>e_max)
	{
		e_max=e_h;
		e_index = horizontal_edge;
	}
	if(e_45>e_max){
		e_max=e_45;
		e_index = diagonal_45_degree_edge;
	}
	if(e_135>e_max){
		e_max=e_135;
		e_index = diagonal_135_degree_edge;
	}
	if(e_m>e_max){
		e_max=e_m;
		e_index = non_directional_edge;
	}
	if(e_max<dc_th)
		e_index = NoEdge;

	return(e_index);
}

// 量化特征向量进行
void GF_EHD_INTERNAL::SetEdgeHistogram(EHD*	pEdge_Histogram, unsigned char *m_pEdge_HistogramElement)
{
	int i, j;
	double iQuantValue;

	for( i=0; i < 80; i++ )
	{
		j=0;
		while(1){
			if( j < 7 ) // SIZI-1 
				iQuantValue = (QuantTable[i%5][j]+QuantTable[i%5][j+1])/2.0;
			else 
				iQuantValue = 1.0;
			if(pEdge_Histogram->Local_Edge[i] <= iQuantValue)
				break;
			j++;
		}

		m_pEdge_HistogramElement[i] = j;
	}

	//for( i=0; i < 80; i++ ){
	//	m_pEdge_Histogram.Local_Edge[i] = QuantTable[ i%5 ][ m_pEdge_HistogramElement[i] ];
	//}
}

void GF_EHD_INTERNAL::EdgeHistExtractor(unsigned char *src, int width, int height, int nChannel, unsigned char EHD[])
{
	if(!EHD ||!src)
	{
		printf("Please CHECK your input image!\n");

		return;
	}

	int bin_resize = 0;
	unsigned char pEHD[80];
	memset(pEHD, 0, sizeof(unsigned char) * 80); 

	//resize
	unsigned char *ImageMedia = NULL;
	if (width != WIDTH || height != HEIGHT )
	{
		ImageMedia = api_imageprocess.ImageResize(src, width, height, nChannel, 256, 256);
		bin_resize = 1;
	}
	else
		ImageMedia = (unsigned char *)src;

	int rt = StartExtracting(ImageMedia, width, height, nChannel, pEHD);
	if(rt != 0)
	{
		printf("StartExtracting error \n"); 
		if ((bin_resize==1)&&(ImageMedia)) {delete [] ImageMedia;ImageMedia = NULL;}
	}

	for(int i = 0; i < 80; i++)
	{
		EHD[i] = (unsigned char)pEHD[i];
	}

	if ((bin_resize==1)&&(ImageMedia)) {delete [] ImageMedia;ImageMedia = NULL;}

	return;
}

//----------------------------------------------------------------------------
void GF_EHD_INTERNAL::Make_Global_SemiGlobal(int *LocalHistogramOnly, int *TotalHistogram)
{
	int i, j;	
	int base_index; 
	//局部复制到0-5上 
	memcpy(TotalHistogram + 5, LocalHistogramOnly, EDGE_DIM *sizeof(int) );

	// Make Global Histogram Start
	for(i=0; i<5; i++)
		TotalHistogram[i]=0.0;
	for( j=0; j < EDGE_DIM; j+=5)
	{
		for( i=0; i < 5; i++) 
		{
			TotalHistogram[i] += TotalHistogram[5+i+j]; 
		}
	}  // for( j ) 

	for(i=0; i<5; i++)
		// Global *5.
		TotalHistogram[i] = (TotalHistogram[i] * 5)>>4;

	// Make Global Histogram end


	// Make Semi-Global Histogram start
	for(i=85; i <105; i++)
	{
		j = i - 85;
		TotalHistogram[i] =
			(TotalHistogram[5+j]
		+TotalHistogram[5+20+j]
		+TotalHistogram[5+40+j]
		+TotalHistogram[5+60+j]) >>2;
	}

	for(i=105; i < 125; i++) 
	{
		j = i-105;
		base_index  = 5+20*(j/5)+j%5;

		TotalHistogram[i] =
			(TotalHistogram[base_index]
		+TotalHistogram[base_index+5]
		+TotalHistogram[base_index+10]
		+TotalHistogram[base_index+15])>>2;
	}
	///////////////////////////////////////////////////////
	//				4 area Semi-Global
	///////////////////////////////////////////////////////
	//  Upper area 2.

	for(i=125; i < 135; i++) 
	{
		j = i-125;    // j = 0 ~ 9
		
		base_index = 5+10*(j/5) + j%5; 

		TotalHistogram[i] =
			(TotalHistogram[base_index]
		+TotalHistogram[base_index + 5]
		+TotalHistogram[base_index + 20]
		+TotalHistogram[base_index +25]) >>2;
	}
	//  Down area 2.
	for(i=135; i < 145; i++) 
	{
		j = i-135;    // j = 0 ~ 9
		base_index = 5 + 10*(j/5)+40+j%5; 

		TotalHistogram[i] =
			(TotalHistogram[base_index]
		+TotalHistogram[base_index + 5]
		+TotalHistogram[base_index + 20]
		+TotalHistogram[base_index + 25]) >>2;
	}

	// Center Area 1 
	for(i=145; i < 150; i++) 
	{
		j = i-145;    // j = 0 ~ 9
		base_index = 5+25+j%5; 

		TotalHistogram[i] =
			(TotalHistogram[base_index]
		+TotalHistogram[base_index + 5]  //5
		+TotalHistogram[base_index + 20]  //20
		+TotalHistogram[base_index + 25])>> 2; 
	}
	// Make Semi-Global Histogram end

}

double GF_EHD_INTERNAL::EHDDist(unsigned char EHD1[], unsigned char EHD2[])
{
	// ------------------------- Calculate the distance ------------------------
	int EHD1_LOCAL[EDGE_DIM], EHD2_LOCAL[EDGE_DIM];
	int EHD1_ALL[150], EHD2_ALL[150];
	
	for(int i =0; i < EDGE_DIM; i++)
	{
		EHD1_LOCAL[i] = (int)EHD1[i];
		EHD2_LOCAL[i] = (int)EHD2[i];
	}

	//求解 全部特征 
	Make_Global_SemiGlobal(EHD1_LOCAL, EHD1_ALL);
	Make_Global_SemiGlobal(EHD2_LOCAL, EHD2_ALL);

	double dist = 0.0;
	for(int i=0; i < TOTAL_EHD_DIM; i++)
	{
		// Global(5)+Semi_Global(65) 
		dist += (abs((EHD1_ALL[i] - EHD2_ALL[i])));
	}

	return 	dist;
};

