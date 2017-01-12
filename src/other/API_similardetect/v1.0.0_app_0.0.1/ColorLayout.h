#include <iostream>
using namespace std;

namespace SIMILARDETECT_INTERNAL{

	//imgprocess
	unsigned char * ImageCopy(unsigned char *src, int width, int height, int nChannel);
	unsigned char * ImageROI(unsigned char *src, int width, int height, int nChannel, int roi_x, int roi_y, int roi_w, int roi_h);
	unsigned char * ImageResize(unsigned char *src, int width, int height, int nChannel, int scale_w, int scale_h);
	
	//CLD
	//Extracting MPEG-7 CLD cld_len 12 ªÚ «18
	void ColorLayoutExtractor(unsigned char *src, int width, int height, int nChannel, unsigned char CLD[], int cld_len);

	//Computing Similarity of 2 CLD
	double CLDDist(int CLD1[], int CLD2[]);

	void MultiBlock_LayoutExtractor(unsigned char *src, int width, int height, int nChannel, unsigned char *LayoutFV);

}
