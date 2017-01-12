#ifndef _GF_CLD_INTERNAL_H_
#define _GF_CLD_INTERNAL_H_

#include <iostream>
#include "ImageProcessing.h"

using namespace std;

class GF_CLD_INTERNAL
{
#ifndef M_PI
	#define M_PI (3.14159265358979323846)
#endif

#ifndef FLT_MAX
	#define FLT_MAX 10000000000.0
#endif

#ifndef MABS
	//fast abs of integer number
	#define MABS(x)                (((x)+((x)>>31))^((x)>>31))     
#endif //MABS

	unsigned char zigzag_scan[64]={        /* Zig-Zag scan pattern  */
		0,1,8,16,9,2,3,10,17,24,32,25,18,11,4,5,
		12,19,26,33,40,48,41,34,27,20,13,6,7,14,21,28,
		35,42,49,56,57,50,43,36,29,22,15,23,30,37,44,51,
		58,59,52,45,38,31,39,46,53,60,61,54,47,55,62,63
	};

	///////////////////////////////////////////////////////////////////////////////////////
	#define WIDTH	256
	#define HEIGHT	256
	
	#define NUMBEROFYCOEFF 6
	#define NUMBEROFCCOEFF 3

	#define CLD_COLOR_DIM      72    //使用
	#define CLD_SINGLE_DIM     18 
	#define BLOB_NUM         	2 
	#define MIN_IMAGE_WIDTH  	32 
	#define MIN_IMAGE_HEIGHT 	32

	/***********************************public***********************************/
public:

	/// construct function 
	GF_CLD_INTERNAL();
	
	/// distruct function
	~GF_CLD_INTERNAL(void);

	//CLD
	//Extracting MPEG-7 CLD cld_len 12 或是18
	void MultiBlock_LayoutExtractor(unsigned char *src, int width, int height, int nChannel, unsigned char *LayoutFV);

	//Computing Similarity of 2 CLD
	double CLDDist(unsigned char CLD1[], unsigned char CLD2[]);

/***********************************private***********************************/
private:

	API_IMAGEPROCESS api_imageprocess;

	/***********************************private***********************************/
	void GetBGRChannelData(unsigned char *src, int width, int height, int nChannel, unsigned char *B, unsigned char *G, unsigned char *R);
	void ColorLayoutExtractor(unsigned char *src, int width, int height, int nChannel, unsigned char CLD[], int cld_len);
	void CreateSmallImage(unsigned char *src, int width, int height, int nChannel, int small_img[3][64]);
	void init_fdct(double trans_coffe[8][8]);
	void fdct(int *block,double trans_coffe[8][8]);
	int quant_ydc(int i);
	int quant_cdc(int i);
	int quant_ac(int i);

};

#endif

