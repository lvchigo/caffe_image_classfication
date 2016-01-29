#include "cv.h"

namespace GF_INTERNAL{

//Extracting MPEG-7 CLD cld_len 12 ªÚ «18
void ColorLayoutExtractor(IplImage *Image, int CLD[], int cld_len);

//Computing Similarity of 2 CLD
double CLDDist(int CLD1[], int CLD2[]);

void MultiBlock_LayoutExtractor(IplImage *img, int *LayoutFV);

}
