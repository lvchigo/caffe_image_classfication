#include<stdio.h>
#include<stdlib.h>
#include "in_image_color_std.h"

const int KERNELS = 15;
const int BINS = 16;
const int UINT8PIX = 256;
const float MINPERCENT = 0.005;

#ifndef MIN
#define MIN(A,B)  ( ((A) > (B))? (B) : (A) )
#endif

typedef unsigned char uchar;
typedef struct RgbColor {
   int r;
   int g;
   int b;
   float c;
   RgbColor(){
     r = 0;
     g = 0;
     b = 0;
     c = 0.0f;
   }
 }RgbColor;


 int comp(const void*a,const void*b)
 {
	 RgbColor* ca = (RgbColor* )a;
	 RgbColor* cb = (RgbColor* )b;
	 return cb->c - ca->c;
 }

/*
  *
  */
 int main_color_from_bgra(int* argb, int width, int height, float lightness)
  {
	 int mr = 185;
	 int mg = 185;
	 int mb = 185;
	 int main_rgb = 0;
	 if(NULL == argb || width < KERNELS+1 || height < KERNELS+1){
		 main_rgb = (mr & 0xFF) | ((mg & 0xFF) << 8) | ((mb & 0xFF) << 16) | ((255 & 0xFF) << 24);
		 return main_rgb;
	 }
	 lightness = (lightness < 0.0f || lightness > 1.0f)? 0.8f:lightness;

 	 int kernels = KERNELS;
 	 int bins = BINS;

 	 int radius = (kernels - 1)/2;
 	 radius = radius < 1? 1:radius;
 	 int blocks = (2*radius+1)*(2*radius+1);
 	 int range = UINT8PIX / bins;
 	 int npixels = (width - 2*radius)*(height - 2*radius);

 	 int ncolors = bins*bins*bins;
 	 //std::vector < std::pair<RgbColor, float> > colorlist;
 	 //colorlist.resize(ncolors);
 	 RgbColor* colorlist = new RgbColor[ncolors];
 	 if(NULL == colorlist){
 		main_rgb = (mr & 0xFF) | ((mg & 0xFF) << 8) | ((mb & 0xFF) << 16) | ((255 & 0xFF) << 24);
 		return main_rgb;
 	 }
 	 for (int i = 0; i < ncolors; i++) {
 		colorlist[i].r = 128;
 		colorlist[i].g = 128;
 		colorlist[i].b = 128;
 		colorlist[i].c = 0.0f;
 	 }


 	 int rsum = 0, gsum = 0, bsum = 0;
 	 int idx, pixel;
 	 int r, g, b;
 	 int ri, gi, bi;
 	 for(int h = radius; h < height - radius; h++){
 		 for(int w = radius; w < width - radius; w++){

 			 rsum = 0;
 			 gsum = 0;
 			 bsum = 0;
 			 //do mean filter
 			 for(int m = h - radius; m <= h + radius; m++){
 				 for(int n = w - radius; n <= w + radius; n++){
 					idx = m * width + n;
 					pixel = argb[idx];
 					r = ((pixel >> 16) & 0xFF);
 					g = ((pixel >> 8) & 0xFF);
 					b = (pixel & 0xFF);

 					rsum += r;
 					gsum += g;
 					bsum += b;
 				 }//for-m
 			 }//for-n

 			 // pixel after filter
 			r = (rsum / blocks);
 			g = (gsum / blocks);
 			b = (bsum / blocks);

 			// vector pixel
 			ri = r / range;
 			gi = g / range;
 			bi = b / range;
 			idx = bins*bins*ri + bins*gi + bi;
 			colorlist[idx].r += r;
 			colorlist[idx].g += g;
 			colorlist[idx].b += b;
 			colorlist[idx].c += 1.0f;

 		 }//for-w
 	 }//for-h

 	 //sort
 	 //sort( colorlist.begin(), colorlist.end(), pair_sort_comp );
 	qsort(colorlist, ncolors, sizeof(RgbColor), comp);

 	// *****
 	float score = colorlist[0].c;
 	mr = (int) (colorlist[0].r/score);
    mg = (int) (colorlist[0].g/score);
 	mb = (int) (colorlist[0].b/score);
 	//
 	float ml = (mr + mg + mb)/3.0f/255.0f;
 	idx = 0;
 	while(true){
 		//
		idx++;
		if (ml <= lightness && ml > 0.2)
			break;
		//if (idx >  ncolors-1) break; MIN(bins, ncolors-1)
		if (idx > ncolors-1) {
			mr = 185;
			mg = 185;
			mb = 185;
			ml = (mr + mg + mb) / 3.0f / 255.0f;
			break;
		}
 		//
		score = colorlist[idx].c;
		mr = (int) (colorlist[idx].r / score);
		mg = (int) (colorlist[idx].g / score);
		mb = (int) (colorlist[idx].b / score);
		ml = (mr + mg + mb)/3.0f/255.0f;

 	}
 	//
 	main_rgb = (mr & 0xFF) | ((mg & 0xFF) << 8) | ((mb & 0xFF) << 16) | ((255 & 0xFF) << 24);
 	//
 	//colorlist.swap(colorlist);
 	if(NULL != colorlist){
 		delete [] colorlist;
 		colorlist = NULL;
 	}

 	//
 	return main_rgb;
 }
