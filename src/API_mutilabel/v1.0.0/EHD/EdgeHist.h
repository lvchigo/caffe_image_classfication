#include "cv.h"

namespace GF_INTERNAL{

//Extracting MPEG-7 EHD
void EdgeHistExtractor(IplImage *Image, int EHD[]);

//Computing Similarity of 2 EHD
double EHDDist(int EHD1[], int EHD2[]);

} //end of namespace
