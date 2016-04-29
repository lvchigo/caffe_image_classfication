#pragma once
#include <string>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h> 
#include <sys/types.h> 
#include <iostream>
#include <unistd.h>

#include <cv.h>
#include <cxcore.h>
#include <highgui.h>

#include "API_skincolor.h"
#include "TErrorCode.h"

using namespace cv;
using namespace std;


/***********************************Init*************************************/
/// construct function 
API_SKINCOLOR::API_SKINCOLOR()
{
}

/// destruct function 
API_SKINCOLOR::~API_SKINCOLOR(void)
{
}
  
// skin region location using rgb limitation   
void API_SKINCOLOR::SkinRGB(IplImage* rgb,IplImage* _dst)  
{  
    assert(rgb->nChannels==3&& _dst->nChannels==3);  
  
    static const int R=2;  
    static const int G=1;  
    static const int B=0;  
  
    IplImage* dst=cvCreateImage(cvGetSize(_dst),8,3);  
    cvZero(dst);  
  
    for (int h=0;h<rgb->height;h++) {  
        unsigned char* prgb=(unsigned char*)rgb->imageData+h*rgb->widthStep;  
        unsigned char* pdst=(unsigned char*)dst->imageData+h*dst->widthStep;  
        for (int w=0;w<rgb->width;w++) {  
            if ((prgb[R]>95 && prgb[G]>40 && prgb[B]>20 &&  
                prgb[R]-prgb[B]>15 && prgb[R]-prgb[G]>15/*&& 
                !(prgb[R]>170&&prgb[G]>170&&prgb[B]>170)*/)||//uniform illumination    
                (prgb[R]>200 && prgb[G]>210 && prgb[B]>170 &&  
                abs(prgb[R]-prgb[B])<=15 && prgb[R]>prgb[B]&& prgb[G]>prgb[B])//lateral illumination   
                ) 
            {
                memcpy(pdst,prgb,3);  
            }             
            prgb+=3;  
            pdst+=3;  
        }  
    }  
    cvCopyImage(dst,_dst);  
    cvReleaseImage(&dst);  
}  

// skin detection in rg space   
void API_SKINCOLOR::cvSkinRG(IplImage* rgb,IplImage* gray)  
{  
    assert(rgb->nChannels==3&&gray->nChannels==1);  
      
    const int R=2;  
    const int G=1;  
    const int B=0;  
  
    double Aup=-1.8423;  
    double Bup=1.5294;  
    double Cup=0.0422;  
    double Adown=-0.7279;  
    double Bdown=0.6066;  
    double Cdown=0.1766;  
    for (int h=0;h<rgb->height;h++) {  
        unsigned char* pGray=(unsigned char*)gray->imageData+h*gray->widthStep;  
        unsigned char* pRGB=(unsigned char* )rgb->imageData+h*rgb->widthStep;  
        for (int w=0;w<rgb->width;w++)   
        {  
            int s=pRGB[R]+pRGB[G]+pRGB[B];  
            double r=(double)pRGB[R]/s;  
            double g=(double)pRGB[G]/s;  
            double Gup=Aup*r*r+Bup*r+Cup;  
            double Gdown=Adown*r*r+Bdown*r+Cdown;  
            double Wr=(r-0.33)*(r-0.33)+(g-0.33)*(g-0.33);  
            if (g<Gup && g>Gdown && Wr>0.004)  
            {  
                *pGray=255;  
            }  
            else  
            {   
                *pGray=0;  
            }  
            pGray++;  
            pRGB+=3;  
        }  
    }  
  
}  

// implementation of otsu algorithm   
// author: onezeros#yahoo.cn   
// reference: Rafael C. Gonzalez. Digital Image Processing Using MATLAB   
void API_SKINCOLOR::cvThresholdOtsu(IplImage* src, IplImage* dst)  
{  
    int height=src->height;  
    int width=src->width;  
  
    //histogram   
    float histogram[256]={0};  
    for(int i=0;i<height;i++) {  
        unsigned char* p=(unsigned char*)src->imageData+src->widthStep*i;  
        for(int j=0;j<width;j++) {  
            histogram[*p++]++;  
        }  
    }  
    //normalize histogram   
    int size=height*width;  
    for(int i=0;i<256;i++) {  
        histogram[i]=histogram[i]/size;  
    }  
  
    //average pixel value   
    float avgValue=0;  
    for(int i=0;i<256;i++) {  
        avgValue+=i*histogram[i];  
    }  
  
    int threshold;    
    float maxVariance=0;  
    float w=0,u=0;  
    for(int i=0;i<256;i++) {  
        w+=histogram[i];  
        u+=i*histogram[i];  
  
        float t=avgValue*w-u;  
        float variance=t*t/(w*(1-w));  
        if(variance>maxVariance) {  
            maxVariance=variance;  
            threshold=i;  
        }  
    }  
  
    cvThreshold(src,dst,threshold,255,CV_THRESH_BINARY);  
}  
  
void API_SKINCOLOR::cvSkinOtsu(IplImage* src, IplImage* dst)  
{  
    assert(dst->nChannels==1&& src->nChannels==3);  
  
    IplImage* ycrcb=cvCreateImage(cvGetSize(src),8,3);  
    IplImage* cr=cvCreateImage(cvGetSize(src),8,1);  
    cvCvtColor(src,ycrcb,CV_BGR2YCrCb);  
    cvSplit(ycrcb,0,cr,0,0);  
  
    cvThresholdOtsu(cr,cr);  
    cvCopyImage(cr,dst);  
    cvReleaseImage(&cr);  
    cvReleaseImage(&ycrcb);  
}  
  
void API_SKINCOLOR::cvSkinYUV(IplImage* src,IplImage* dst)  
{  
    IplImage* ycrcb=cvCreateImage(cvGetSize(src),8,3);  
    //IplImage* cr=cvCreateImage(cvGetSize(src),8,1);   
    //IplImage* cb=cvCreateImage(cvGetSize(src),8,1);   
    cvCvtColor(src,ycrcb,CV_BGR2YCrCb);  
    //cvSplit(ycrcb,0,cr,cb,0);   
  
    static const int Cb=2;  
    static const int Cr=1;  
    static const int Y=0;  
  
    //IplImage* dst=cvCreateImage(cvGetSize(_dst),8,3);   
    cvZero(dst);  
  
    for (int h=0;h<src->height;h++) {  
        unsigned char* pycrcb=(unsigned char*)ycrcb->imageData+h*ycrcb->widthStep;  
        unsigned char* psrc=(unsigned char*)src->imageData+h*src->widthStep;  
        unsigned char* pdst=(unsigned char*)dst->imageData+h*dst->widthStep;  
        for (int w=0;w<src->width;w++) {  
            if (pycrcb[Cr]>=133&&pycrcb[Cr]<=173&&pycrcb[Cb]>=77&&pycrcb[Cb]<=127)  
            {  
                    memcpy(pdst,psrc,3);  
            }  
            pycrcb+=3;  
            psrc+=3;  
            pdst+=3;  
        }  
    }  
    //cvCopyImage(dst,_dst);   
    //cvReleaseImage(&dst);   
}  
  
void API_SKINCOLOR::cvSkinHSV(IplImage* src,IplImage* dst)  
{  
    IplImage* hsv=cvCreateImage(cvGetSize(src),8,3);  
    //IplImage* cr=cvCreateImage(cvGetSize(src),8,1);   
    //IplImage* cb=cvCreateImage(cvGetSize(src),8,1);   
    cvCvtColor(src,hsv,CV_BGR2HSV);  
    //cvSplit(ycrcb,0,cr,cb,0);   
  
    static const int V=2;  
    static const int S=1;  
    static const int H=0;  
  
    //IplImage* dst=cvCreateImage(cvGetSize(_dst),8,3);   
    cvZero(dst);  
  
    for (int h=0;h<src->height;h++) {  
        unsigned char* phsv=(unsigned char*)hsv->imageData+h*hsv->widthStep;  
        unsigned char* psrc=(unsigned char*)src->imageData+h*src->widthStep;  
        unsigned char* pdst=(unsigned char*)dst->imageData+h*dst->widthStep;  
        for (int w=0;w<src->width;w++) {  
            if (phsv[H]>=7&&phsv[H]<=29)  
            {  
                    memcpy(pdst,psrc,3);  
            }  
            phsv+=3;  
            psrc+=3;  
            pdst+=3;  
        }  
    }  
    //cvCopyImage(dst,_dst);   
    //cvReleaseImage(&dst);   
}  
  
int main()  
{     
      
    IplImage* img= cvLoadImage("D:/skin.jpg"); //\u968f\u4fbf\u653e\u4e00\u5f20jpg\u56fe\u7247\u5728D\u76d8\u6216\u53e6\u884c\u8bbe\u7f6e\u76ee\u5f55   
    IplImage* dstRGB=cvCreateImage(cvGetSize(img),8,3);  
    IplImage* dstRG=cvCreateImage(cvGetSize(img),8,1);  
    IplImage* dst_crotsu=cvCreateImage(cvGetSize(img),8,1);  
    IplImage* dst_YUV=cvCreateImage(cvGetSize(img),8,3);  
    IplImage* dst_HSV=cvCreateImage(cvGetSize(img),8,3);  
  
  
    cvNamedWindow("inputimage", CV_WINDOW_AUTOSIZE);  
    cvShowImage("inputimage", img);  
    cvWaitKey(0);  
  
    SkinRGB(img,dstRGB);  
    cvNamedWindow("outputimage1", CV_WINDOW_AUTOSIZE);  
    cvShowImage("outputimage1", dstRGB);  
    cvWaitKey(0);  
    cvSkinRG(img,dstRG);  
    cvNamedWindow("outputimage2", CV_WINDOW_AUTOSIZE);  
    cvShowImage("outputimage2", dstRG);  
    cvWaitKey(0);  
    cvSkinOtsu(img,dst_crotsu);  
    cvNamedWindow("outputimage3", CV_WINDOW_AUTOSIZE);  
    cvShowImage("outputimage3", dst_crotsu);  
    cvWaitKey(0);  
    cvSkinYUV(img,dst_YUV);  
    cvNamedWindow("outputimage4", CV_WINDOW_AUTOSIZE);  
    cvShowImage("outputimage4", dst_YUV);  
    cvWaitKey(0);  
    cvSkinHSV(img,dst_HSV);  
    cvNamedWindow("outputimage5", CV_WINDOW_AUTOSIZE);  
    cvShowImage("outputimage5", dst_HSV);  
    cvWaitKey(0);  
    return 0;  
}  


