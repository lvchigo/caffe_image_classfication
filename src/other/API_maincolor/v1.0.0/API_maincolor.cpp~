#pragma once
//#include <cuda_runtime.h>
//#include <google/protobuf/text_format.h>
#include <queue>  // for std::priority_queue
#include <utility>  // for pair

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <dirent.h>
#include <unistd.h>

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <ml.h>
#include <cvaux.h>

#include <time.h>
#include <sys/mman.h> /* for mmap and munmap */
#include <sys/types.h> /* for open */
#include <sys/stat.h> /* for open */
#include <fcntl.h>     /* for open */
#include <pthread.h>

#include <vector>
#include <list>
#include <map>
#include <algorithm>

#include "API_maincolor.h"
#include "TErrorCode.h"

using namespace cv;
using namespace std;

/***********************************Init*************************************/
/// construct function 
API_MAINCOLOR::API_MAINCOLOR()
{
}

/// destruct function 
API_MAINCOLOR::~API_MAINCOLOR(void)
{
}

/***********************************Init*************************************/
static bool ImgSortComp(
	const pair< vector< int >, float > elem1, 
	const pair< vector< int >, float > elem2)
{
	return (elem1.second > elem2.second);
}


/***********************************Predict**********************************/
int API_MAINCOLOR::Predict(
	uchar*										bgr,				//[In]:image->bgr
	int 										width,				//[In]:image->width
	int 										height, 			//[In]:image->height
	int 										channel,			//[In]:image->channel
	UInt64										ImageID,			//[In]:ImageID
	int 										numColorBlock,		//[In]:numColorBlock
	vector< pair< vector< int >,float > >		&Res)				//[Out]:Res

{
	if( !bgr || (width<16) || (height<16) || channel != 3 ) 
	{	
		cout<<"image err!!" << endl;
		return TEC_INVALID_PARAM;
	}	

	/*****************************Init*****************************/
	int i,m,n,rWidth,rHeight,nRet = 0;
	char szImgPath[1024];

	/*****************************Change Img Format*****************************/
	IplImage* image = api_commen.uchar2ipl( bgr, width, height, channel );	
	IplImage* correspond = cvCreateImage( cvSize(image->width*4, image->height*2), image->depth, image->nChannels );
	cvSetImageROI( correspond, cvRect( 0, 0, image->width, image->height ) );
    cvCopy( image, correspond );

	/*****************************predict Img*****************************/
	Res.clear();
	nRet = ColorHistogram( bgr, width, height, channel, ImageID, numColorBlock, Res );	//img_resize
	if (nRet != 0)
	{	
		cout<<"Fail to ColorHistogramExtract!! "<<endl;
		cvReleaseImage(&image);image = 0;
		
		return nRet;
	}

	/*****************************check Img*****************************/
	IplImage* imgCH = cvCreateImage( cvGetSize(image), image->depth, image->nChannels );
	for (m = 0; m < image->height; m++) 
	{
		for (n = 0; n < image->width; n++) 
		{
			((uchar *)(imgCH->imageData + m*image->widthStep))[n*image->nChannels + 0] = Res[0].first[0];	//B
			((uchar *)(imgCH->imageData + m*image->widthStep))[n*image->nChannels + 1] = Res[0].first[1];	//G
			((uchar *)(imgCH->imageData + m*image->widthStep))[n*image->nChannels + 2] = Res[0].first[2];	//R
		}
	}
	
	cvSetImageROI( correspond, cvRect( 0, image->height, image->width, image->height ) );
	cvCopy( imgCH, correspond );

	/*****************************cvSmooth Img*****************************/
	gaussianFilter( bgr, width, height, channel );

	/*****************************Change Img Format*****************************/
	IplImage* img_smooth1 = api_commen.uchar2ipl( bgr, width, height, channel );	
	cvSetImageROI( correspond, cvRect( image->width, 0, image->width, image->height ) );
    cvCopy( img_smooth1, correspond );

	/*****************************predict Img*****************************/
	Res.clear();
	nRet = ColorHistogram( bgr, width, height, channel, ImageID, numColorBlock, Res );	//img_resize
	if (nRet != 0)
	{	
		cout<<"Fail to ColorHistogramExtract!! "<<endl;
		cvReleaseImage(&image);image = 0;
		
		return nRet;
	}

	/*****************************check Img*****************************/
	IplImage* imgCH1 = cvCreateImage( cvGetSize(image), image->depth, image->nChannels );
	for (m = 0; m < image->height; m++) 
	{
		for (n = 0; n < image->width; n++) 
		{
			((uchar *)(imgCH1->imageData + m*image->widthStep))[n*image->nChannels + 0] = Res[0].first[0];	//B
			((uchar *)(imgCH1->imageData + m*image->widthStep))[n*image->nChannels + 1] = Res[0].first[1];	//G
			((uchar *)(imgCH1->imageData + m*image->widthStep))[n*image->nChannels + 2] = Res[0].first[2];	//R
		}
	}
	
	cvSetImageROI( correspond, cvRect( image->width, image->height, image->width, image->height ) );
	cvCopy( imgCH1, correspond );

	/*****************************cvSmooth Img*****************************/
	gaussianFilter( bgr, width, height, channel );

	/*****************************Change Img Format*****************************/
	IplImage* img_smooth2 = api_commen.uchar2ipl( bgr, width, height, channel );	
	cvSetImageROI( correspond, cvRect( image->width*2, 0, image->width, image->height ) );
    cvCopy( img_smooth2, correspond );

	/*****************************predict Img*****************************/
	Res.clear();
	nRet = ColorHistogram( bgr, width, height, channel, ImageID, numColorBlock, Res );	//img_resize
	if (nRet != 0)
	{	
		cout<<"Fail to ColorHistogramExtract!! "<<endl;
		cvReleaseImage(&image);image = 0;
		
		return nRet;
	}

	/*****************************check Img*****************************/
	IplImage* imgCH2 = cvCreateImage( cvGetSize(image), image->depth, image->nChannels );
	for (m = 0; m < image->height; m++) 
	{
		for (n = 0; n < image->width; n++) 
		{
			((uchar *)(imgCH2->imageData + m*image->widthStep))[n*image->nChannels + 0] = Res[0].first[0];	//B
			((uchar *)(imgCH2->imageData + m*image->widthStep))[n*image->nChannels + 1] = Res[0].first[1];	//G
			((uchar *)(imgCH2->imageData + m*image->widthStep))[n*image->nChannels + 2] = Res[0].first[2];	//R
		}
	}
	
	cvSetImageROI( correspond, cvRect( image->width*2, image->height, image->width, image->height ) );
	cvCopy( imgCH2, correspond );

	
	/*****************************cvSmooth Img*****************************/
	gaussianFilter( bgr, width, height, channel );

	/*****************************Change Img Format*****************************/
	IplImage* img_smooth3 = api_commen.uchar2ipl( bgr, width, height, channel );	
	cvSetImageROI( correspond, cvRect( image->width*3, 0, image->width, image->height ) );
    cvCopy( img_smooth3, correspond );

	/*****************************predict Img*****************************/
	Res.clear();
	nRet = ColorHistogram( bgr, width, height, channel, ImageID, numColorBlock, Res );	//img_resize
	if (nRet != 0)
	{	
		cout<<"Fail to ColorHistogramExtract!! "<<endl;
		cvReleaseImage(&image);image = 0;
		
		return nRet;
	}

	/*****************************check Img*****************************/
	IplImage* imgCH3 = cvCreateImage( cvGetSize(image), image->depth, image->nChannels );
	for (m = 0; m < image->height; m++) 
	{
		for (n = 0; n < image->width; n++) 
		{
			((uchar *)(imgCH3->imageData + m*image->widthStep))[n*image->nChannels + 0] = Res[0].first[0];	//B
			((uchar *)(imgCH3->imageData + m*image->widthStep))[n*image->nChannels + 1] = Res[0].first[1];	//G
			((uchar *)(imgCH3->imageData + m*image->widthStep))[n*image->nChannels + 2] = Res[0].first[2];	//R
		}
	}
	
	cvSetImageROI( correspond, cvRect( image->width*3, image->height, image->width, image->height ) );
	cvCopy( imgCH3, correspond );

	/*****************************output info*****************************/
    cvResetImageROI( correspond );
	sprintf(szImgPath, "res/%.4f_%d_%d_%d_%ld.jpg",
			Res[0].second, Res[0].first[2], Res[0].first[1], Res[0].first[0], ImageID );
	cvSaveImage( szImgPath,correspond );
	
	/*****************************cvReleaseImage*****************************/
	cvReleaseImage(&correspond);correspond = 0;
	cvReleaseImage(&image);image = 0;
	cvReleaseImage(&imgCH);imgCH = 0;
	cvReleaseImage(&img_smooth1);img_smooth1 = 0;
	cvReleaseImage(&imgCH1);imgCH1 = 0;
	cvReleaseImage(&img_smooth2);img_smooth2 = 0;
	cvReleaseImage(&imgCH2);imgCH2 = 0;
	cvReleaseImage(&img_smooth3);img_smooth3 = 0;
	cvReleaseImage(&imgCH3);imgCH3 = 0;

	return nRet;
}

/***********************************Predict**********************************/
int API_MAINCOLOR::Predict_ipl(
	IplImage								*image, 			//[In]:image
	UInt64									ImageID,			//[In]:ImageID
	int 									numColorBlock,		//[In]:numColorBlock
	vector< pair< vector< int >,float > >	&Res)				//[Out]:Res
{
	if(!image || (image->width<16) || (image->height<16) || image->nChannels != 3 || image->depth != IPL_DEPTH_8U) 
	{	
		cout<<"image err!!" << endl;
		return TEC_INVALID_PARAM;
	}	

	/*****************************Init*****************************/
	int i,m,n,rWidth,rHeight,nRet = 0;
	char szImgPath[1024];

	/*****************************Resize Img*****************************/
	nRet = api_commen.GetReWH( image->width, image->height, 64.0, rWidth, rHeight );
	if (nRet != 0)
	{	
		cout<<"Fail to GetReWH!! "<<endl;	
		return nRet;
	}
	
	IplImage *img_resize = cvCreateImage(cvSize(rWidth, rHeight), image->depth, image->nChannels);
	cvResize(image, img_resize);

	IplImage* correspond = cvCreateImage( cvSize(img_resize->width*4, img_resize->height*2), img_resize->depth, img_resize->nChannels );
	
	/*****************************cvSmooth Img*****************************/
	IplImage *img_smooth = cvCreateImage(cvGetSize(img_resize), img_resize->depth, img_resize->nChannels);
	cvSmooth(img_resize,img_smooth,CV_GAUSSIAN,3,3);

	/*****************************predict Img*****************************/
	Res.clear();
	nRet = ColorHistogram_ipl( img_resize, ImageID, numColorBlock, Res );	//img_resize
	if (nRet != 0)
	{	
		cout<<"Fail to ColorHistogramExtract!! "<<endl;
		cvReleaseImage(&img_smooth);img_smooth = 0;
		cvReleaseImage(&img_resize);img_resize = 0;
		cvReleaseImage(&correspond);correspond = 0;
		
		return nRet;
	}

	/*****************************check Img*****************************/
	IplImage* imgCHResize = cvCreateImage( cvGetSize(img_resize), img_resize->depth, img_resize->nChannels );
	for (m = 0; m < img_resize->height; m++) 
	{
		for (n = 0; n < img_resize->width; n++) 
		{
			((uchar *)(imgCHResize->imageData + m*img_resize->widthStep))[n*img_resize->nChannels + 0] = Res[0].first[0];	//B
			((uchar *)(imgCHResize->imageData + m*img_resize->widthStep))[n*img_resize->nChannels + 1] = Res[0].first[1];	//G
			((uchar *)(imgCHResize->imageData + m*img_resize->widthStep))[n*img_resize->nChannels + 2] = Res[0].first[2];	//R
		}
	}
	cvSetImageROI( correspond, cvRect( 0, 0, img_resize->width, img_resize->height ) );
    cvCopy( img_resize, correspond );
	cvSetImageROI( correspond, cvRect( 0, img_resize->height, img_resize->width, img_resize->height ) );
	cvCopy( imgCHResize, correspond );

	/*****************************predict Img*****************************/
	Res.clear();
	nRet = ColorHistogram_ipl( img_smooth, ImageID, numColorBlock, Res );	//img_smooth
	if (nRet != 0)
	{	
		cout<<"Fail to ColorHistogramExtract!! "<<endl;
		cvReleaseImage(&img_smooth);img_smooth = 0;
		cvReleaseImage(&img_resize);img_resize = 0;
		
		return nRet;
	}

	/*****************************check Img*****************************/
	IplImage* imgCHSmooth1 = cvCreateImage( cvGetSize(img_resize), img_resize->depth, img_resize->nChannels );
	for (m = 0; m < img_resize->height; m++) 
	{
		for (n = 0; n < img_resize->width; n++) 
		{
			((uchar *)(imgCHSmooth1->imageData + m*img_resize->widthStep))[n*img_resize->nChannels + 0] = Res[0].first[0];	//B
			((uchar *)(imgCHSmooth1->imageData + m*img_resize->widthStep))[n*img_resize->nChannels + 1] = Res[0].first[1];	//G
			((uchar *)(imgCHSmooth1->imageData + m*img_resize->widthStep))[n*img_resize->nChannels + 2] = Res[0].first[2];	//R
		}
	}
	cvSetImageROI( correspond, cvRect( img_resize->width, 0, img_resize->width, img_resize->height ) );
	cvCopy( img_smooth, correspond );
	cvSetImageROI( correspond, cvRect( img_resize->width, img_resize->height, img_resize->width, img_resize->height ) );
	cvCopy( imgCHSmooth1, correspond );

	/*****************************cvSmooth Img*****************************/
	cvSmooth(img_smooth,img_smooth,CV_GAUSSIAN,3,3);

	/*****************************predict Img*****************************/
	Res.clear();
	nRet = ColorHistogram_ipl( img_smooth, ImageID, numColorBlock, Res );	//img_smooth
	if (nRet != 0)
	{	
		cout<<"Fail to ColorHistogramExtract!! "<<endl;
		cvReleaseImage(&img_smooth);img_smooth = 0;
		cvReleaseImage(&img_resize);img_resize = 0;
		
		return nRet;
	}

	/*****************************check Img*****************************/
	IplImage* imgCHSmooth2 = cvCreateImage( cvGetSize(img_resize), img_resize->depth, img_resize->nChannels );
	for (m = 0; m < img_resize->height; m++) 
	{
		for (n = 0; n < img_resize->width; n++) 
		{
			((uchar *)(imgCHSmooth2->imageData + m*img_resize->widthStep))[n*img_resize->nChannels + 0] = Res[0].first[0];	//B
			((uchar *)(imgCHSmooth2->imageData + m*img_resize->widthStep))[n*img_resize->nChannels + 1] = Res[0].first[1];	//G
			((uchar *)(imgCHSmooth2->imageData + m*img_resize->widthStep))[n*img_resize->nChannels + 2] = Res[0].first[2];	//R
		}
	}
	cvSetImageROI( correspond, cvRect( img_resize->width*2, 0, img_resize->width, img_resize->height ) );
    cvCopy( img_smooth, correspond );
	cvSetImageROI( correspond, cvRect( img_resize->width*2, img_resize->height, img_resize->width, img_resize->height ) );
	cvCopy( imgCHSmooth2, correspond );

	/*****************************cvSmooth Img*****************************/
	cvSmooth(img_smooth,img_smooth,CV_GAUSSIAN,3,3);

	/*****************************predict Img*****************************/
	Res.clear();
	nRet = ColorHistogram_ipl( img_smooth, ImageID, numColorBlock, Res );	//img_smooth
	if (nRet != 0)
	{	
		cout<<"Fail to ColorHistogramExtract!! "<<endl;
		cvReleaseImage(&img_smooth);img_smooth = 0;
		cvReleaseImage(&img_resize);img_resize = 0;
		
		return nRet;
	}

	/*****************************check Img*****************************/
	IplImage* imgCHSmooth3 = cvCreateImage( cvGetSize(img_resize), img_resize->depth, img_resize->nChannels );
	for (m = 0; m < img_resize->height; m++) 
	{
		for (n = 0; n < img_resize->width; n++) 
		{
			((uchar *)(imgCHSmooth3->imageData + m*img_resize->widthStep))[n*img_resize->nChannels + 0] = Res[0].first[0];	//B
			((uchar *)(imgCHSmooth3->imageData + m*img_resize->widthStep))[n*img_resize->nChannels + 1] = Res[0].first[1];	//G
			((uchar *)(imgCHSmooth3->imageData + m*img_resize->widthStep))[n*img_resize->nChannels + 2] = Res[0].first[2];	//R
		}
	}
	cvSetImageROI( correspond, cvRect( img_resize->width*3, 0, img_resize->width, img_resize->height ) );
    cvCopy( img_smooth, correspond );
	cvSetImageROI( correspond, cvRect( img_resize->width*3, img_resize->height, img_resize->width, img_resize->height ) );
	cvCopy( imgCHSmooth3, correspond );

	/*****************************output info*****************************/
    cvResetImageROI( correspond );
	sprintf(szImgPath, "res/%.4f_%d_%d_%d_%ld.jpg",
			Res[0].second, Res[0].first[2], Res[0].first[1], Res[0].first[0], ImageID );
	cvSaveImage( szImgPath,correspond );
	
	/*****************************cvReleaseImage*****************************/
	cvReleaseImage(&imgCHResize);imgCHResize = 0;
	cvReleaseImage(&imgCHSmooth1);imgCHSmooth1 = 0;
	cvReleaseImage(&imgCHSmooth2);imgCHSmooth2 = 0;
	cvReleaseImage(&imgCHSmooth3);imgCHSmooth3 = 0;
	cvReleaseImage(&correspond);correspond = 0;
	cvReleaseImage(&img_smooth);img_smooth = 0;
	cvReleaseImage(&img_resize);img_resize = 0;

	return nRet;
}

int API_MAINCOLOR::ColorHistogram(
	uchar*										bgr,				//[In]:image->bgr
	int 										width,				//[In]:image->width
	int 										height, 			//[In]:image->height
	int 										channel,			//[In]:image->channel
	UInt64										ImageID,			//[In]:ImageID
	int 										numColorBlock,		//[In]:numColorBlock)
	vector< pair< vector< int >,float > >		&Res)				//[Out]:Res
{
	int nRet = TOK;
	int i,j,k,count,tmpPixel;
	
	int blockPixel = int( 255.0/numColorBlock + 0.5 );
	int halfBlockPixel = int( blockPixel*0.5 + 0.5 );
	double size = 1.0/ (height * width) ;

	vector< int > vecColor;
	map< vector< int >,long > mapMainColor;
	map< vector< int >,long >::iterator itMainColor;
	vector< pair< vector< int >,float > > ResMainColor;
	
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) 
		{
			vecColor.clear();
			for(k=0;k<channel;k++)
			{
				tmpPixel = bgr[i*width*channel+j*channel+k];
				tmpPixel = int(tmpPixel*1.0/blockPixel)*blockPixel + halfBlockPixel;	//norm	
				vecColor.push_back( tmpPixel );	//[B-G-R],[gray]
			}
			
			itMainColor = mapMainColor.find(vecColor);
			if (itMainColor == mapMainColor.end()) // not find
			{
				mapMainColor[vecColor] = 1;
			}
			else
			{
				count = itMainColor->second; 
	            itMainColor->second = count+1; 
			}
		}
	}

	for(itMainColor = mapMainColor.begin(); itMainColor != mapMainColor.end(); itMainColor++)
	{
		ResMainColor.push_back( std::make_pair( itMainColor->first, itMainColor->second*size ) );
	}

	/*****************************Sort*****************************/
	//sort label result
	sort( ResMainColor.begin(), ResMainColor.end(), ImgSortComp );

	//get max score
	int resNum = (ResMainColor.size()>2)?2:ResMainColor.size();
	for( i=0;i<resNum;i++ )
	{
		Res.push_back( std::make_pair( ResMainColor[i].first, ResMainColor[i].second ) );
		printf( "ImageID:%lld,i-%d,R-%d,G-%d,B-%d,score-%.8f\n",
			ImageID,i,Res[i].first[2],Res[i].first[1],Res[i].first[0],Res[i].second );
	}
	
	return nRet;
}

int API_MAINCOLOR::ColorHistogram_ipl(
	IplImage 									*image, 			//[In]:image
	UInt64										ImageID,			//[In]:ImageID
	int 										numColorBlock,		//[In]:numColorBlock)
	vector< pair< vector< int >,float > >		&Res)				//[Out]:Res
{
	int nRet = TOK;
	int i,j,k,count,tmpPixel;
	
	int height     = image->height;
	int width      = image->width;
	int step       = image->widthStep;
	int channels   = image->nChannels;
	int blockPixel = int( 255.0/numColorBlock + 0.5 );
	int halfBlockPixel = int( blockPixel*0.5 + 0.5 );
	double size = 1.0/ (height * width) ;

	vector< int > vecColor;
	map< vector< int >,long > mapMainColor;
	map< vector< int >,long >::iterator itMainColor;
	vector< pair< vector< int >,float > > ResMainColor;
	
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) 
		{
			vecColor.clear();
			for(k=0;k<channels;k++)
			{
				tmpPixel = ((uchar *)(image->imageData + i*step))[j*channels + k];
				tmpPixel = int(tmpPixel*1.0/blockPixel)*blockPixel + halfBlockPixel;	//norm	
				vecColor.push_back( tmpPixel );	//B-G-R
			}
			
			itMainColor = mapMainColor.find(vecColor);
			if (itMainColor == mapMainColor.end()) // not find
			{
				mapMainColor[vecColor] = 1;
			}
			else
			{
				count = itMainColor->second; 
	            itMainColor->second = count+1; 
			}
		}
	}

	for(itMainColor = mapMainColor.begin(); itMainColor != mapMainColor.end(); itMainColor++)
	{
		ResMainColor.push_back( std::make_pair( itMainColor->first, itMainColor->second*size ) );
	}

	/*****************************Sort*****************************/
	//sort label result
	sort( ResMainColor.begin(), ResMainColor.end(), ImgSortComp );

	//get max score
	int resNum = (ResMainColor.size()>2)?2:ResMainColor.size();
	for( i=0;i<resNum;i++ )
	{
		Res.push_back( std::make_pair( ResMainColor[i].first, ResMainColor[i].second ) );
		//printf( "ImageID:%lld,i-%d,R-%d,G-%d,B-%d,score-%.8f\n",
		//	ImageID,i,Res[i].first[2],Res[i].first[1],Res[i].first[0],Res[i].second );
	}
	
	return nRet;
}


void API_MAINCOLOR::gaussianFilter(uchar* data, int width, int height, int channel)
{
    int i, j, k, m, n, index, sum;
    int templates[9] = { 1, 2, 1,
                         2, 4, 2,
                         1, 2, 1 };
    sum = height * width * channel * sizeof(uchar);
    uchar *tmpdata = (uchar*)malloc(sum);
    memcpy((int*)tmpdata,(int*)data, sum);

	for(k = 0;k < channel;k++)
    {
	    for(i = 1;i < height - 1;i++)
	    {
	        for(j = 1;j < width - 1;j++)
	        {
	        	sum = 0;
	            index = 0;		
	            for(m = i - 1;m < i + 2;m++)
	            {
	                for(n = j - 1; n < j + 2;n++)
	                {
	                    sum += tmpdata[m*width*channel+n*channel+k] * templates[index];
						index++;
	                }
	            }
	            data[i*width*channel+j*channel+k] = int(sum*1.0/16+0.5);
        	}
        }
    }
	
    free(tmpdata);
}




