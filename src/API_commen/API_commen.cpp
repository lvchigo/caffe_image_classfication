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

#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>
#include <opencv/ml.h>
#include <opencv/cvaux.h>

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

#include "API_commen.h"
#include "TErrorCode.h"

using namespace cv;
using namespace std;


/***********************************Init*************************************/
/// construct function 
API_COMMEN::API_COMMEN()
{
}

/// destruct function 
API_COMMEN::~API_COMMEN(void)
{
}

/***********************************PadEnd***********************************/
void API_COMMEN::PadEnd(char *szPath)
{
	int iLength = strlen(szPath);
	if (szPath[iLength-1] != '/')
	{
		szPath[iLength] = '/';
		szPath[iLength+1] = 0;
	}
}

/***********************************GetIDFromFilePath***********************************/
long API_COMMEN::GetIDFromFilePath(const char *filepath)
{
	long ID = 0;
	int  atom =0;
	string tmpPath = filepath;
	for (int i=tmpPath.rfind('/')+1;i<tmpPath.rfind('.');i++)
	{
		atom = filepath[i] - '0';
		if (atom < 0 || atom >9)
			break;
		ID = ID * 10 + atom;
	}
	return ID;
}

/***********************************GetIDFromFilePath***********************************/
string API_COMMEN::GetStringIDFromFilePath(const char *filepath)
{
	long ID = 0;
	int  atom =0;
	string tmpPath = filepath;
	string iid;

	long start = tmpPath.find_last_of('/');
	long end = tmpPath.find_last_of('.');

	if ( (start>0) && (end>0) && ( end>start ) )
		iid = tmpPath.substr(start+1,end-start-1);
	
	return iid;
}

static bool ImgSortComp(const pair <int, float> elem1, const pair <int, float> elem2)
{
	return (elem1.second > elem2.second);
}

/***********************************split***********************************/
void API_COMMEN::split(const string& src, const string& separator, vector<string>& dest)
{
    string str = src;
    string substring;
    string::size_type start = 0, index;

    do
    {
        index = str.find_first_of(separator,start);
        if (index != string::npos)
        {   
            substring = str.substr(start,index-start);
            dest.push_back(substring);
            start = str.find_first_not_of(separator,index);
            if (start == string::npos) return;
        }
    }while(index != string::npos);
   
    //the last token
    substring = str.substr(start);
    dest.push_back(substring);
}


/***********************************getRandomID***********************************/
void API_COMMEN::getRandomID( UInt64 &randomID )
{
	int i,atom =0;
	randomID = 0;
	
	//time
	time_t nowtime = time(NULL);
	tm *now = localtime(&nowtime);

	char szTime[1024];
	sprintf(szTime, "%04d%02d%02d%02d%02d%02d%d",
			now->tm_year+1900, now->tm_mon+1, now->tm_mday,now->tm_hour, now->tm_min, now->tm_sec,(rand()%100000) );
	//printf("szTime Name:%s\n",szTime);

	string tmpID = szTime;
	for ( i=0;i<tmpID.size();i++)
	{
		atom = szTime[i] - '0';
		if (atom < 0 || atom >9)
			break;
		randomID = randomID * 10 + atom;
	}
}


/***********************************loadWordDict***********************************/
void API_COMMEN::loadWordDict(const char *filePath, vector< string > &labelWords)
{
	ifstream ifs(filePath);
	if(!ifs)
	{
		printf("open %s failed.\n", filePath);
		assert(false);
	}
	string line ;
	while(getline(ifs, line))
	{
		if(line.length() != 0)
			labelWords.push_back(line);
	}
	assert(labelWords.size());
}

/***********************************CountLines***********************************/
long API_COMMEN::doc2vec_CountLines(char *filename)
{
	ifstream ReadFile;
	long n=0;
	char line[4096];
	string temp;
	ReadFile.open(filename,ios::in);
	if(ReadFile.fail())
	{
	   return 0;
	}
	else
	{
		while(getline(ReadFile,temp))
		{
		   	n++;
		}
	}

	ReadFile.close();

	return n;
}

void API_COMMEN::Img_hMirrorTrans(const Mat &src, Mat &dst)
{
    CV_Assert(src.depth() == CV_8U);
    dst.create(src.rows, src.cols, src.type());

    int rows = src.rows;
    int cols = src.cols;

    switch (src.channels())
    {
    case 1:
        const uchar *origal;
        uchar *p;
        for (int i = 0; i < rows; i++){
            origal = src.ptr<uchar>(i);
            p = dst.ptr<uchar>(i);
            for (int j = 0; j < cols; j++){
                p[j] = origal[cols - 1 - j];
            }
        }
        break;
    case 3:
        const Vec3b *origal3;
        Vec3b *p3;
        for (int i = 0; i < rows; i++) {
            origal3 = src.ptr<Vec3b>(i);
            p3 = dst.ptr<Vec3b>(i);
            for(int j = 0; j < cols; j++){
                p3[j] = origal3[cols - 1 - j];
            }
        }
        break;
    default:
        break;
    }
}

int API_COMMEN::Img_GetMutiRoi( IplImage *MainBody, UInt64 ImageID, vector<Mat_<Vec3f> > &OutputImg )
{
	if(!MainBody || (MainBody->width<16) || (MainBody->height<16) || MainBody->nChannels != 3 || MainBody->depth != IPL_DEPTH_8U) 
	{	
		cout<<"image err!!" << endl;
		return TEC_INVALID_PARAM;
	}	
	
	int i,rWidth,rHeight,nRet=TOK;
	int MaxLen = 720;	//maxlen:720-same with online
	char imgPath[1024] = {0};
	OutputImg.clear();

	/***********************************Resize Image width && height*****************************/
	IplImage *imgResize;
	if( ( MainBody->width>MaxLen ) || ( MainBody->height>MaxLen ) )
	{
		nRet = GetReWH( MainBody->width, MainBody->height, MaxLen, rWidth, rHeight );	
		if (nRet != 0)
		{
		   	cout<<"GetReWH err!!" << endl;
			return TEC_INVALID_PARAM;
		}

		/*****************************Resize Img*****************************/
		imgResize = cvCreateImage(cvSize(rWidth, rHeight), MainBody->depth, MainBody->nChannels);
		cvResize( MainBody, imgResize );
	}
	else
	{
		imgResize = cvCreateImage(cvSize(MainBody->width, MainBody->height), MainBody->depth, MainBody->nChannels);
		cvCopy( MainBody, imgResize, NULL );
	}

	/*****************************Remove White Part*****************************/
	//If do "Remove White Part", Img_GetMutiRoi_ImageQuality's feat will be useless.
/*	IplImage* ImgRemoveWhite = RemoveWhitePart( imgResize, ImageID );
	if(!ImgRemoveWhite || (ImgRemoveWhite->width<16) || (ImgRemoveWhite->height<16) || ImgRemoveWhite->nChannels != 3 || ImgRemoveWhite->depth != IPL_DEPTH_8U) 
	{	
		cout<<"RemoveWhitePart err!!" << endl;
		cvReleaseImage(&ImgRemoveWhite);ImgRemoveWhite = 0;
		cvReleaseImage(&imgResize);imgResize = 0;
		return TEC_INVALID_PARAM;
	}	
	cvReleaseImage(&imgResize);imgResize = 0;*/

	/*****************************Init Crop Img*****************************/
	rWidth = imgResize->width;
	rHeight = imgResize->height;
	float minWH = (rWidth>rHeight)?rHeight:rWidth;

	/*****************************Get Muti ROI***************************/
	for( i=0;i<2;i++)
	{
		if ( i == 0)
			cvSetImageROI( imgResize,cvRect(0,0, imgResize->width,imgResize->height) );	//for in36class && ads6class
		else if ( i == 1)
			cvSetImageROI( imgResize,cvRect((rWidth-minWH)*0.5,(rHeight-minWH)*0.5, minWH,minWH) );	//for imagequality
		IplImage* MutiROI = cvCreateImage(cvGetSize(imgResize),imgResize->depth,imgResize->nChannels);
		cvCopy( imgResize, MutiROI, NULL );
		cvResetImageROI(imgResize);	
		
		/*****************************Resize Img*****************************/
		IplImage *MutiROIResize = cvCreateImage(cvSize(BLOB_WIDTH, BLOB_HEIGHT), imgResize->depth, imgResize->nChannels);
		cvResize( MutiROI, MutiROIResize );
		
		/*****************************push data*****************************/
		Mat matResize( MutiROIResize );
		OutputImg.push_back( matResize );
		
		/*****************************save tmp data*****************************/		
		//sprintf(imgPath, "res/Img_GetMutiRoi_%lld_%d.jpg",ImageID,i);
		//imwrite( imgPath,matResize );
		
		/*****************************Release********************************/
		cvReleaseImage(&MutiROI);MutiROI = 0;
		cvReleaseImage(&MutiROIResize);MutiROIResize = 0;
	}

	/*****************************Get Muti ImageQuality ROI***************************/
	vector<Mat_<Vec3f> > ImageQualityROI;
	nRet = Img_GetMutiRoi_ImageQuality( imgResize, ImageID, ImageQualityROI );
	if (nRet != 0)
	{	
		cout<<"Img_GetMutiRoi_ImageQuality err!!" << endl;
		cvReleaseImage(&imgResize);imgResize = 0;
		return TEC_INVALID_PARAM;
	}

	/*****************************push data*****************************/
	for( i=0;i<ImageQualityROI.size();i++ )
	{
		OutputImg.push_back( ImageQualityROI[i] );
	}
	
	/*****************************Release********************************/
	cvReleaseImage(&imgResize);imgResize = 0;

	return TOK;
}

int API_COMMEN::Img_GetMutiRoi_ImageQuality( IplImage *MainBody, UInt64 ImageID, vector<Mat_<Vec3f> > &OutputImg )
{
	if(!MainBody || (MainBody->width<16) || (MainBody->height<16) || MainBody->nChannels != 3 || MainBody->depth != IPL_DEPTH_8U) 
	{	
		cout<<"image err!!" << endl;
		return TEC_INVALID_PARAM;
	}
	
	int i,x,y,width,height,rWidth,rHeight,nRet=TOK;
	int MaxLen,MinLen = 256;	//MinLen:256-for move box
	char imgPath[1024] = {0};
	vector< pair< int, float > > blobScore;
	OutputImg.clear();
	CvScalar Bl_mean[9];
	CvScalar Bl_dev[9];

	width = MainBody->width;
	height = MainBody->height;

	/***********************************Resize Image width && height*****************************/
	IplImage *imgResize;
	if( ( width<MinLen ) || ( height<MinLen ) )
	{
		nRet = GetReWH_Minlen( width, height, MinLen, rWidth, rHeight );	
		if (nRet != 0)
		{
		   	cout<<"GetReWH err!!" << endl;
			return TEC_INVALID_PARAM;
		}

		/*****************************Resize Img*****************************/
		imgResize = cvCreateImage(cvSize(rWidth, rHeight), MainBody->depth, MainBody->nChannels);
		cvResize( MainBody, imgResize );
	}
	else
	{
		imgResize = cvCreateImage(cvSize(width, height), MainBody->depth, MainBody->nChannels);
		cvCopy( MainBody, imgResize, NULL );
	}

	rWidth = imgResize->width;
	rHeight = imgResize->height;

	//printf("Img_GetMutiRoi_ImageQuality_bak:ImageID:%ld,w:%d,h%d,rw:%d,rh%d,MinLen:%d!\n",
	//	ImageID,MainBody->width,MainBody->height,imgResize->width,imgResize->height,MinLen);

	/*****************************Get Muti ImageQuality ROI***************************/
	int blob_x[3] = {0, (rWidth-BLOB_WIDTH)/2, rWidth - BLOB_WIDTH};
	int blob_y[3] = {0, (rHeight-BLOB_HEIGHT)/2, rHeight - BLOB_HEIGHT};
	for( x = 0; x<3; x++){
		for( y = 0; y<3; y++){
			cvSetImageROI( imgResize, cvRect(blob_x[x], blob_y[y], BLOB_WIDTH, BLOB_HEIGHT ) );
			IplImage* MutiROI = cvCreateImage(cvGetSize(imgResize), imgResize->depth, imgResize->nChannels);
			cvCopy(imgResize, MutiROI, NULL );
			cvResetImageROI(imgResize);	

			/*****************************push data*****************************/
			//if ( (x+y)%2 == 0 )
			if ( ( (x+y)%2 == 1 ) || ( (x==1) && (y==1) ) )
			{
				Mat matResize( MutiROI );
				OutputImg.push_back( matResize );
			}

			/**********************count blob dev*************************************/
			cvAvgSdv(MutiROI,&Bl_mean[3*x+y],&Bl_dev[3*x+y],NULL);
			blobScore.push_back( make_pair( 3*x+y, Bl_dev[3*x+y].val[0] ) );

			/*****************************save tmp data*****************************/		
			//sprintf(imgPath, "res/blob_%lld_%d_%d.jpg",ImageID,x,y);
			//cvSaveImage( imgPath,MutiROI );

			/*****************************Release********************************/
			cvReleaseImage(&MutiROI);MutiROI = 0;
		}
	}

	//sort label result
	sort(blobScore.begin(), blobScore.end(),ImgSortComp);	

	/*****************************Get Muti ImageQuality ROI***************************/
	for ( i = 0; i < 0; ++i)
	{
		if ( blobScore[0].second == 0 )
		{
			cout<<"blobScore err!!" << endl;
			return TEC_INVALID_PARAM;
		}
		
		x = blobScore[i].first/3;
		y = blobScore[i].first%3;

		cvSetImageROI( imgResize, cvRect(blob_x[x], blob_y[y], BLOB_WIDTH, BLOB_HEIGHT ) );
		IplImage* MutiROI = cvCreateImage(cvGetSize(imgResize), imgResize->depth, imgResize->nChannels);
		cvCopy(imgResize, MutiROI, NULL );
		cvResetImageROI(imgResize);	

		/*****************************push data*****************************/
		Mat matResize( MutiROI, 1 );
		OutputImg.push_back( matResize );

		/*****************************save tmp data*****************************/		
		//sprintf(imgPath, "res/top3blob_%lld_%d_%d_%.4f.jpg",ImageID,i,blobScore[i].first,blobScore[i].second);
		//cvSaveImage( imgPath,MutiROI );

		/*****************************Release********************************/
		cvReleaseImage(&MutiROI);MutiROI = 0;

		//printf("Img_GetMutiRoi_ImageQuality_bak:do sort && save feat");
			
	}

	/*****************************Release********************************/
	cvReleaseImage(&imgResize);imgResize = 0;

	return TOK;
}

void API_COMMEN::Img_Get10MutiRoi( IplImage *MainBody, UInt64 ImageID, vector<Mat_<Vec3f> > &OutputImg )
{
	int i;
	int xROI = int( 0.1*MainBody->width + 0.5 );
	int yROI = int( 0.1*MainBody->height + 0.5 );
	int widthROI = int( 0.9*MainBody->width + 0.5 );
	int heightROI = int( 0.9*MainBody->height + 0.5 );
	char imgPath[1024] = {0};

	/*****************************Get Muti ROI***************************/
	for( i=0;i<5;i++)
	{
		if ( i == 0)
			cvSetImageROI( MainBody,cvRect(0,0, MainBody->width,MainBody->height) );	
		else if ( i == 1)
			cvSetImageROI( MainBody,cvRect(0,0, widthROI,heightROI) );	
		else if ( i == 2)
			cvSetImageROI( MainBody,cvRect(xROI,0, widthROI,heightROI) );	
		else if ( i == 3)
			cvSetImageROI( MainBody,cvRect(0,yROI, widthROI,heightROI) );	
		else if ( i == 4)
			cvSetImageROI( MainBody,cvRect(xROI,yROI, widthROI,heightROI) );	
		IplImage* MutiROI = cvCreateImage(cvGetSize(MainBody),MainBody->depth,MainBody->nChannels);
		cvCopy( MainBody, MutiROI, NULL );
		cvResetImageROI(MainBody);	
		
		/*****************************Resize Img*****************************/
		IplImage *MutiROIResize = cvCreateImage(cvSize(BLOB_WIDTH, BLOB_HEIGHT), MainBody->depth, MainBody->nChannels);
		cvResize( MutiROI, MutiROIResize );
		
		/*****************************push data*****************************/
		Mat matResize( MutiROIResize );
		OutputImg.push_back( matResize );
		
		Mat mat_hMirrorTrans;
		Img_hMirrorTrans(matResize, mat_hMirrorTrans);
		OutputImg.push_back( mat_hMirrorTrans );
		
		/*****************************save tmp data*****************************/		
		//sprintf(imgPath, "%s%lld_%d_1.jpg",tmpSavePath.c_str(),ImageID,i);
		//imwrite( imgPath,matResize );

		//sprintf(imgPath, "%s%lld_%d_2.jpg",tmpSavePath.c_str(),ImageID,i);
		//imwrite( imgPath,mat_hMirrorTrans );
		
		/*****************************Release********************************/
		cvReleaseImage(&MutiROI);MutiROI = 0;
		cvReleaseImage(&MutiROIResize);MutiROIResize = 0;
	}
}

IplImage* API_COMMEN::ResizeImg( IplImage *img, int MaxLen )
{
	int rWidth, rHeight, nRet = 0;

	IplImage *imgResize;
	if( ( img->width>MaxLen ) || ( img->height>MaxLen ) )
	{
		nRet = GetReWH( img->width, img->height, MaxLen, rWidth, rHeight );	
		if (nRet != 0)
		{
			printf("GetReWH err!!\n");
			return NULL;
		}

		/*****************************Resize Img*****************************/
		imgResize = cvCreateImage(cvSize(rWidth, rHeight), img->depth, img->nChannels);
		cvResize( img, imgResize );
	}
	else
	{
		imgResize = cvCreateImage(cvSize(img->width, img->height), img->depth, img->nChannels);
		cvCopy( img, imgResize, NULL );
	}

	return imgResize;
}

IplImage* API_COMMEN::ResizeImg( IplImage *img, float &ratio, int MaxLen )
{
	int rWidth, rHeight, nRet = 0;
	ratio = 1.0;

	//Resize
	if (img->width > img->height) {
		if (img->width > MaxLen)
			ratio = MaxLen*1.0 / img->width;
	} 
	else 
	{	
		if (img->height > MaxLen)
			ratio = MaxLen*1.0 / img->height;
	}
	rWidth =  (int )img->width * ratio;
	rHeight = (int )img->height * ratio;

	IplImage *imgResize = cvCreateImage(cvSize(rWidth, rHeight), img->depth, img->nChannels);
	cvResize( img, imgResize );

	return imgResize;
}

int API_COMMEN::Normal_MinMax( vector<float> inFeat, vector<float> &NormFeat )
{
	if (inFeat.size()==0)
		return TEC_INVALID_PARAM;
		
	int j;
	float tmpMin = 1000000.0;
	float tmpMax = -1000000.0;

	/************************Normalization*****************************/
	for ( j=0;j < inFeat.size();j++ )
	{
		if ( inFeat[j] < tmpMin )
			tmpMin = inFeat[j];
		if ( inFeat[j] > tmpMax )
			tmpMax = inFeat[j];
	}

	/************************Normalization*****************************/
	NormFeat.clear();
	for (j = 0; j < inFeat.size(); j++) 
	{
		if ( tmpMax == tmpMin )
			NormFeat.push_back( 0 );
		else
	  		NormFeat.push_back( (inFeat[j]-tmpMin)*1.0/(tmpMax-tmpMin) );
	}

	return TOK;
}

int API_COMMEN::Normal_L1( vector<float> inFeat, vector<float> &NormFeat )
{
	if (inFeat.size()==0)
		return TEC_INVALID_PARAM;
		
	int j;
	double Sum = 0.0;

	/************************Normalization*****************************/
	for ( j=0;j < inFeat.size();j++ )
	{
		Sum += fabs(inFeat[j]);
	}

	/************************Normalization*****************************/
	NormFeat.clear();
	for (j = 0; j < inFeat.size(); j++) 
	{
		if ( Sum == 0 )
			NormFeat.push_back( 0 );
		else
	  		NormFeat.push_back( inFeat[j]*1.0/Sum );
	}

	return TOK;
}


int API_COMMEN::Normal_L2( vector<float> inFeat, vector<float> &NormFeat )
{
	if (inFeat.size()==0)
		return TEC_INVALID_PARAM;
		
	int j;
	double Sum = 0.0;

	/************************Normalization*****************************/
	for ( j=0;j < inFeat.size();j++ )
	{
		Sum += pow(inFeat[j],2);
	}
	Sum = sqrt(Sum);

	/************************Normalization*****************************/
	NormFeat.clear();
	for (j = 0; j < inFeat.size(); j++) 
	{
		if ( Sum == 0 )
			NormFeat.push_back( 0 );
		else
	  		NormFeat.push_back( inFeat[j]*1.0/Sum );
	}

	return TOK;
}

// calculate entropy of an image
int API_COMMEN::ExtractFeat_Entropy( vector<float> inFeat, float &Entropy )
{
    if(inFeat.size()==0)
    {
        return TEC_INVALID_PARAM;
    }

    int i,ret = TOK;
	Entropy = 0;
	
	vector<float> NormFeat;
	ret = Normal_L2( inFeat, NormFeat );
	if (ret != 0)
	{
	   printf("Fail to Normal_L2!!\n");
	   return ret;
	}

    for(i =0;i<inFeat.size();i++)
    {
        if(NormFeat[i]>0)
            Entropy = Entropy-NormFeat[i]*(log(NormFeat[i])/log(2.0));
    }

    return ret;
}

/***********************************Image Format Change*************************************/
uchar* API_COMMEN::ipl2mat(IplImage* image)			//for matlab
{
    //
    if(NULL == image)
    {
        return NULL;
    }
    //
    int width =  image->width;
    int height = image->height;
    int channel = image->nChannels;
    uchar *mat = NULL;

    int step = image->widthStep;
    uchar* data;
    data = (uchar*) image->imageData;
    
    switch(channel)
    {
    case 1:
        mat = new uchar[width*height];
        for(int i = 0; i<width; i++)
        {
            uchar *p = mat + height*i;
            for(int j = 0; j<height; j++, p++)
            {
                *p = data[j*step+i];
            }
        }
        break;
    case 3:
        mat = new uchar[width*height*3];
        uchar *r, *g, *b;
        r = mat;
        g = mat+width*height;
        b = mat+width*height*2;
        for(int i=0; i<width; i++)
        {
            uchar *pr = r + height*i;
            uchar *pg = g + height*i;
            uchar *pb = b + height*i;
            for(int j = 0; j<height; j++, pr++, pb++, pg++)
            {
                *pr = data[j*step + 3*i +2]; // r
                *pg = data[j*step + 3*i +1]; // g
                *pb = data[j*step + 3*i ];  // b
            }
        }
        break;
    default:
        break;
    }
    
    //
    return mat;
}

///
uchar* API_COMMEN::ipl2uchar(IplImage* image)		//for c/c++
{
    //
    if(NULL == image)
    {
        return NULL;
    }
    //
    int width =  image->width;
    int height = image->height;
    int channel = image->nChannels;
    uchar *bgr = NULL;

    int step = image->widthStep;
    uchar* data;
    data = (uchar*) image->imageData;

    switch(channel)
    {
    case 1:
        bgr = new uchar[width*height];
        for(int i = 0; i<height; i++)
        {
            memcpy(bgr+i*width, data+i*step, width*sizeof(uchar));
        }
        break;
    case 3:
        bgr = new uchar[width*height*3];
        for(int i=0; i<height; i++)
        {
            memcpy(bgr+i*width*3, data+i*step, width*3*sizeof(uchar));
        }
        break;
    default:
        break;
    }
    
    //
    return bgr;
}

///
uchar* API_COMMEN::ipl2rgb(IplImage* image)
{
    //
    if(NULL == image)
    {
        return NULL;
    }
    //
    int width =  image->width;
    int height = image->height;
    int channel = image->nChannels;
    uchar *rgb = NULL;

    int step = image->widthStep;
    uchar* data;
    data = (uchar*) image->imageData;
    
    switch(channel)
    {
    case 1:
        rgb = new uchar[width*height*3];
        for(int i = 0; i<height; i++)
        {
            uchar *p = rgb + 3*width*i;
            for(int j = 0; j<width; j++, p+=3)
            {
                *p = *(p+1) = *(p+2) = data[i*step+j];
            }
        }
        break;
    case 3:
        rgb = new uchar[width*height*3];
        for(int i=0; i<height; i++)
        {
            uchar *p = rgb + 3*width*i;
            for(int j = 0; j<width; j++, p+=3)
            {
                *p = data[i*step + 3*j +2]; // r
                *(p+1) = data[i*step + 3*j +1]; // g
                *(p+2) = data[i*step + 3*j ];  // b
            }
        }
        break;
    default:
        break;
    }
    
    //
    return rgb;
}

///
uchar* API_COMMEN::ipl2gray(IplImage* image)
{
    //
    if(NULL == image)
    {
        return NULL;
    }
    //
    int width =  image->width;
    int height = image->height;
    int channel = image->nChannels;
    uchar *gray = NULL;

    int step = image->widthStep;
    uchar* data;
    data = (uchar*) image->imageData;
    gray = new uchar[width*height];
    
    switch(channel)
    {
    case 1:
        
        for(int i = 0; i<height; i++)
        {

            for(int j = 0; j<width; j++)
            {
                gray[i*width + j]= data[i*step+j];
            }
        }
        break;
    case 3:

        for(int i=0; i<height; i++)
        {

            for(int j = 0; j<width; j++)
            {
                //Gray = (R*38 + G*75 + B*15) >> 7
                gray[i*width + j] =(uchar) ((data[i*step + 3*j +2]*38 +
                    data[i*step + 3*j +1]*75 +
                    data[i*step + 3*j ]*15) >> 7);  // b
            }
        }
        break;
    default:
        break;
    }
    
    //
    return gray;
}

///
float* API_COMMEN::ipl2gray_f(IplImage* image)
{
    //
    if(NULL == image)
    {
        return NULL;
    }
    //
    int width =  image->width;
    int height = image->height;
    int channel = image->nChannels;
    float *gray = NULL;

    int step = image->widthStep;
    uchar* data;
    data = (uchar*) image->imageData;
    gray = new float[width*height];
    
    switch(channel)
    {
    case 1:
        
        for(int i = 0; i<height; i++)
        {

            for(int j = 0; j<width; j++)
            {
                gray[i*width + j]= data[i*step+j];
            }
        }
        break;
    case 3:

        for(int i=0; i<height; i++)
        {

            for(int j = 0; j<width; j++)
            {
                //Gray = (R*38 + G*75 + B*15) >> 7
                gray[i*width + j] =  ((data[i*step + 3*j +2]*38 +
                    data[i*step + 3*j +1]*75 +
                    data[i*step + 3*j ]*15) >> 7);  // b
            }
        }
        break;
    default:
        break;
    }
    
    //
    return gray;
}

///
IplImage* API_COMMEN::uchar2ipl(uchar* bgr, int width, int height, int channel)		//for check img
{
    if(NULL == bgr)
    {
        return NULL;
    }
    
    IplImage* image = NULL;
    switch(channel)
    {
    case 3:
    {
        image = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);
        int step = image->widthStep;
        uchar* data = (uchar*) image->imageData;
        for(int i=0; i<height; i++)
        {
            memcpy(data+i*step, bgr+i*width*3, 3*width*sizeof(uchar));
        }
        break;
    }
    case 1:
    {
        image = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
        int step = image->widthStep;
        uchar* data = (uchar*) image->imageData;
        for(int i = 0; i<height; i++)
        {
            memcpy(data+i*step, bgr+i*width, width*sizeof(uchar));
        }
        break;
    }
	case 4:
    {
        image = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);
        int step = image->widthStep;
        uchar* data = (uchar*) image->imageData;
        for(int i=0; i<height; i++)
        {
            //memcpy(data+i*step, bgr+i*width*3, 3*width*sizeof(uchar));
			for(int j = 0; j<width; j++)
            {
                data[i*step + 3*j +2] = bgr[i*width + j]; // r
                data[i*step + 3*j +1] = bgr[i*width + j]; // g
                data[i*step + 3*j ] = bgr[i*width + j];  // b
            }
        }
        break;
    }
    default:
        break;
    }

    //
    return image;
}

int API_COMMEN::GetReWH( int width, int height, float maxLen, int &rWidth, int &rHeight )
{
	if( (width<16) || (height<16) ) 
	{	
		cout<<"image err!!" << endl;
		return TEC_INVALID_PARAM;
	}	

	float ratio = 1.0f;
	maxLen = (maxLen<16.0)?64.0:maxLen;
	/*****************************Resize*****************************/
	if (width > height) {
		if (width > maxLen)
			ratio = maxLen / width;
	} 
	else 
	{	
		if (height > maxLen)
			ratio = maxLen / height;
	}

	rWidth =  (int )width * ratio;
	rWidth = (((rWidth + 3) >> 2) << 2);	//width 4 char
	rHeight = (int )height * ratio;

	return TOK;
}

int API_COMMEN::GetReWH_Minlen( int width, int height, float minLen, int &rWidth, int &rHeight )
{
	if( (width<16) || (height<16) ) 
	{	
		cout<<"image err!!" << endl;
		return TEC_INVALID_PARAM;
	}	

	float ratio = 1.0f;
	minLen = (minLen<16.0)?64.0:minLen;
	/*****************************Resize*****************************/
	if (width < height) {
		if (width < minLen)
			ratio = minLen / width;
	} 
	else 
	{	
		if (height < minLen)
			ratio = minLen / height;
	}

	rWidth =  (int )width * ratio;
	rWidth = (((rWidth + 3) >> 2) << 2);	//width 4 char
	rHeight = (int )height * ratio;

	return TOK;
}


void API_COMMEN::ImgProcess_gaussianFilter(uchar* data, int width, int height, int channel)
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

/***********************************RemoveWhitePart**********************************/
IplImage* API_COMMEN::RemoveWhitePart(
	IplImage								*image, 			//[In]:image
	UInt64									ImageID)			//[In]:ImageID
{
	if(!image || (image->width<16) || (image->height<16) || image->nChannels != 3 || image->depth != IPL_DEPTH_8U) 
	{	
		cout<<"image err!!" << endl;
		return NULL;
	}

	/*****************************Init*****************************/
	int i,x,y,k,m,n,rWidth,rHeight,nRet = 0;
	char szImgPath[1024];
	int point[4] = {0};
	int val[3] = {0};
	int tmpPixel = 0;

	/*****************************find Img*****************************/  
	//left
	point[0] = 0;
	for (x = 0; x < image->width; x++)  
    {    
    	tmpPixel = 0;
        for( y = 0; y < image->height; y++)   
        {    
        	val[0] = ((uchar*)(image->imageData + image->widthStep*y))[x*image->nChannels]; 
			val[1] = ((uchar*)(image->imageData + image->widthStep*y))[x*image->nChannels+1]; 
			val[2] = ((uchar*)(image->imageData + image->widthStep*y))[x*image->nChannels+2]; 
			if ( ( ( val[0]==255 ) && ( val[1]==255 ) && ( val[2]==255 ) ) ||
				 ( ( val[0]==0 ) && ( val[1]==0 ) && ( val[2]==0 ) ) )
				tmpPixel++;
			else
				break;
        }    
		
		if ( tmpPixel == image->height )
			point[0] = x;
		else
			break;
    }    

	//right
	point[2] = image->width-1;
	for (x = image->width-1; x >=0; x--) 
    {    
    	tmpPixel = 0;
        for( y = 0; y < image->height; y++)    
        {    
        	val[0] = ((uchar*)(image->imageData + image->widthStep*y))[x*image->nChannels]; 
			val[1] = ((uchar*)(image->imageData + image->widthStep*y))[x*image->nChannels+1]; 
			val[2] = ((uchar*)(image->imageData + image->widthStep*y))[x*image->nChannels+2]; 
			if ( ( ( val[0]==255 ) && ( val[1]==255 ) && ( val[2]==255 ) ) ||
				 ( ( val[0]==0 ) && ( val[1]==0 ) && ( val[2]==0 ) ) )
				tmpPixel++;
			else
				break;
        }    
		
		if ( tmpPixel == image->height )
			point[2] = x+1;
		else
			break;
    } 
	
	//up
	point[1] = 0;
    for( y = 0; y < image->height; y++)    
    {    
    	tmpPixel = 0;
        for (x = 0; x < image->width; x++)    
        {    
        	val[0] = ((uchar*)(image->imageData + image->widthStep*y))[x*image->nChannels]; 
			val[1] = ((uchar*)(image->imageData + image->widthStep*y))[x*image->nChannels+1]; 
			val[2] = ((uchar*)(image->imageData + image->widthStep*y))[x*image->nChannels+2]; 
			if ( ( ( val[0]==255 ) && ( val[1]==255 ) && ( val[2]==255 ) ) ||
				 ( ( val[0]==0 ) && ( val[1]==0 ) && ( val[2]==0 ) ) )
				tmpPixel++;
			else
				break;
        }    
		
		if ( tmpPixel == image->width )
			point[1] = y;
		else
			break;
    }    

	//down
	point[3] = image->height-1;
	for( y = image->height-1; y>=0 ; y-- )    
    {    
    	tmpPixel = 0;
        for (x = 0; x < image->width; x++)    
        {    
        	val[0] = ((uchar*)(image->imageData + image->widthStep*y))[x*image->nChannels]; 
			val[1] = ((uchar*)(image->imageData + image->widthStep*y))[x*image->nChannels+1]; 
			val[2] = ((uchar*)(image->imageData + image->widthStep*y))[x*image->nChannels+2]; 
			if ( ( ( val[0]==255 ) && ( val[1]==255 ) && ( val[2]==255 ) ) ||
				 ( ( val[0]==0 ) && ( val[1]==0 ) && ( val[2]==0 ) ) )
				tmpPixel++;
			else
				break;
        }    
		
		if ( tmpPixel == image->width )
			point[3] = y+1;
		else
			break;
    } 

	point[2] = point[2]-point[0]; 
	point[3] = point[3]-point[1]; 

	if( (point[2]<16) || (point[3]<16) ) 
	{	
		cout<<"image err!!" << endl;
		return NULL;
	}

	/*****************************Crop Img*****************************/
	cvSetImageROI( image,cvRect(point[0],point[1], point[2],point[3]) );	
	IplImage* MutiROI = cvCreateImage(cvGetSize(image),image->depth,image->nChannels);
	cvCopy( image, MutiROI, NULL );
	cvResetImageROI(image);	
	
	/*****************************output info*****************************/
	//sprintf(szImgPath, "res/src_%ld.jpg", ImageID );
	//cvSaveImage( szImgPath,image );

	//sprintf(szImgPath, "res/cs_RemoveWhite_%ld.jpg", ImageID );
	//cvSaveImage( szImgPath,MutiROI );

	/*****************************cvReleaseImage*****************************/
	//cvReleaseImage(&MutiROI);MutiROI = 0;

	return MutiROI; 
}


/***********************************Creat Sample**********************************/
int API_COMMEN::CreatSample_LowContrast(
	IplImage								*image, 			//[In]:image
	UInt64									ImageID,			//[In]:ImageID
	double									gamma)				//[In]:gamma
{
	if(!image || (image->width<16) || (image->height<16) || image->nChannels != 3 || image->depth != IPL_DEPTH_8U) 
	{	
		cout<<"image err!!" << endl;
		return TEC_INVALID_PARAM;
	}

	/*****************************Init*****************************/
	int i,m,n,rWidth,rHeight,nRet = 0;
	char szImgPath[1024];

	/*****************************gamma adjust*****************************/
	IplImage *img_ImageAdjust1 = cvCloneImage(image); 
	nRet = CreatSample_ImageAdjust( image, img_ImageAdjust1, 0, 1, 0, 1, gamma);
	if (nRet != 0)
	{	
		cout<<"Fail to img_ImageAdjust1!! "<<endl;
		cvReleaseImage(&img_ImageAdjust1);img_ImageAdjust1 = 0;
		
		return nRet;
	}
	
	/*****************************output info*****************************/
	sprintf(szImgPath, "res/cs_LowContrast_%.2f_%ld.jpg", gamma, ImageID );
	cvSaveImage( szImgPath,img_ImageAdjust1 );

	/*****************************cvReleaseImage*****************************/
	cvReleaseImage(&img_ImageAdjust1);img_ImageAdjust1 = 0;

	return nRet; 
}

int API_COMMEN::CreatSample_ImageAdjust(
	IplImage* src, IplImage* dst,    
    double low, double high,
    double bottom, double top,
    double gamma )    
{      
    double err_in = high - low;    
    double err_out = top - bottom;    
    int x,y,k;    
    double val;    
    if(low<0 && low>1 && high <0 && high>1&&    
    bottom<0 && bottom>1 && top<0 && top>1 && low>high)    
        return -1; 
	
    // intensity transform    
    for( y = 0; y < src->height; y++)    
    {    
        for (x = 0; x < src->width; x++)    
        {    
        	for ( k=0;k<src->nChannels;k++ )
        	{
	            val = ((uchar*)(src->imageData + src->widthStep*y))[x*src->nChannels + k];    
				val = ( pow( (val*1.0/255.0 - low)/err_in, gamma )*err_out+bottom )*255.0;
	            if(val>255)    
	                val = 255;    
	            if(val<0)    
	                val = 0; // Make sure src is in the range [low,high]    
	            ((uchar*)(dst->imageData + dst->widthStep*y))[x*dst->nChannels + k] = (uchar) val;    
        	}
        }    
    }    
	
    return 0;    
}   


int API_COMMEN::CreatSample_LowResolution(
	IplImage								*image, 			//[In]:image
	UInt64									ImageID)			//[In]:ImageID
{
	if(!image || (image->width<16) || (image->height<16) || image->nChannels != 3 || image->depth != IPL_DEPTH_8U) 
	{	
		cout<<"image err!!" << endl;
		return TEC_INVALID_PARAM;
	}

	/*****************************Init*****************************/
	int i,m,n,rWidth,rHeight,nRet = 0;
	char szImgPath[1024];

	/*****************************GetReWH*****************************/
	float maxWH = ( image->width > image->height )?image->width:image->height;
	int ratio = 0;
	if ( maxWH > 512 )
		ratio = 16;
	else
		ratio = 8;

	nRet = GetReWH( image->width, image->height, (maxWH*1.0/ratio) , rWidth, rHeight );
	if (nRet != 0)
	{	
		cout<<"Fail to GetReWH!! "<<endl;	
		return nRet;
	}

	/*****************************Resize Img*****************************/
	IplImage *img_resize_down = cvCreateImage(cvSize(rWidth, rHeight), image->depth, image->nChannels);
	cvResize(image, img_resize_down);

	IplImage *img_resize_up = cvCreateImage(cvSize(image->width, image->height), image->depth, image->nChannels);
	cvResize(img_resize_down, img_resize_up);

	/*****************************output info*****************************/
	sprintf(szImgPath, "res/cs_LowResolution_%ld.jpg", ImageID );
	cvSaveImage( szImgPath,img_resize_up );

	/*****************************cvReleaseImage*****************************/
	cvReleaseImage(&img_resize_down);img_resize_down = 0;
	cvReleaseImage(&img_resize_up);img_resize_up = 0;

	return nRet;
}

int API_COMMEN::CreatSample_smooth(
	IplImage								*image, 			//[In]:image
	UInt64									ImageID)			//[In]:ImageID
{
	if(!image || (image->width<16) || (image->height<16) || image->nChannels != 3 || image->depth != IPL_DEPTH_8U) 
	{	
		cout<<"image err!!" << endl;
		return TEC_INVALID_PARAM;
	}

	/*****************************Init*****************************/
	int i,m,n,rWidth,rHeight,nRet = 0;
	char szImgPath[1024];

	/*****************************cvSmooth Img*****************************/
	float maxWH = ( image->width > image->height )?image->width:image->height;
	int smoothSize = 0;
	if ( maxWH > 512 )
		smoothSize = 19;
	else if ( maxWH > 256 )
		smoothSize = 13;
	else if ( maxWH > 128 )
		smoothSize = 9;
	else
		smoothSize = 7;
	
	IplImage *img_smooth = cvCreateImage(cvGetSize(image), image->depth, image->nChannels);
	cvSmooth(image,img_smooth,CV_GAUSSIAN,smoothSize,smoothSize);
	
	/*****************************output info*****************************/
	sprintf(szImgPath, "res/cs_Smooth_%ld.jpg", ImageID );
	cvSaveImage( szImgPath,img_smooth );

	/*****************************cvReleaseImage*****************************/
	cvReleaseImage(&img_smooth);img_smooth = 0;

	return nRet;
}

int API_COMMEN::CreatSample_addGaussNoise(
	IplImage								*image, 			//[In]:image
	UInt64									ImageID)			//[In]:ImageID
{
	if(!image || (image->width<16) || (image->height<16) || image->nChannels != 3 || image->depth != IPL_DEPTH_8U) 
	{	
		cout<<"image err!!" << endl;
		return TEC_INVALID_PARAM;
	}

	/*****************************Init*****************************/
	int i,m,n,rWidth,rHeight,nRet = 0;
	char szImgPath[1024];
	float delta = 50.0;

	CvRNG rng = cvRNG(-1);

	/*****************************cvSmooth Img*****************************/
	IplImage *noise = cvCreateImage(cvGetSize(image), image->depth, image->nChannels);
	IplImage *dstImage = cvCreateImage(cvGetSize(image), image->depth, image->nChannels);
	
	cvRandArr(&rng,noise,CV_RAND_NORMAL,cvScalarAll(0),cvScalarAll(delta));
	cvAdd(image,noise,dstImage);
	
	/*****************************output info*****************************/
	sprintf(szImgPath, "res/cs_AddGaussNoise_%ld.jpg", ImageID );
	cvSaveImage( szImgPath,dstImage );

	/*****************************cvReleaseImage*****************************/
	cvReleaseImage(&noise);noise = 0;
	cvReleaseImage(&dstImage);dstImage = 0;

	return nRet;
}


/***********************************ExtractFeat*************************************/

//blur 
//reference No-reference Image Quality Assessment using blur and noisy
//write by Min Goo Choi, Jung Hoon Jung  and so on 

int API_COMMEN::ExtractFeat_Blur(
	IplImage* 							pSrcImg,
	vector< float > 					&fBlur)		//5D
{
	if(!pSrcImg) 
		return -1;

	IplImage* pGrayImg = NULL;
	pGrayImg = cvCreateImage(cvGetSize(pSrcImg),pSrcImg->depth,1);
	//for mean filter
	IplImage* pNoisyImg = cvCreateImage(cvGetSize(pSrcImg),pSrcImg->depth,1);

	if(pSrcImg->nChannels == 3) cvCvtColor(pSrcImg,pGrayImg,CV_BGR2GRAY);
	else cvCopy(pSrcImg,pGrayImg);

	//something different form paper i use opencv median filter here
	cvSmooth(pGrayImg,pNoisyImg,CV_MEDIAN);

	int nHei = pGrayImg->height; int nWid = pGrayImg->width;

	int total = (nWid)*(nHei);

	int iLineBytes = pGrayImg->widthStep;
	uchar* pData = (uchar*)pGrayImg->imageData;

	int iNoisyBytes = pNoisyImg->widthStep;
	uchar* pNoisyData = (uchar*)pNoisyImg->imageData;

	int steps = 0;
	//result
	//blur
	double blur_mean = 0;
	double blur_ratio = 0;
	//noisy
	double nosiy_mean = 0;
	double nosiy_ratio = 0;

	//means DhMean and DvMean in paper
	//for edge
	// it is global mean in paper i will try local later
	double ghMean = 0; 
	double gvMean = 0;
	//for noisy
	double gNoisyhMean = 0;
	double gNoisyvMean = 0;
	//Nccand-mean
	double gNoisyMean = 0;

	//tmp color value for h v
	double ch = 0;	
	double cv = 0;
	//The Thresh blur value best detected
	const double blur_th = 0.1;
	//blur value sum
	double blurvalue = 0;
	//blur count
	int blur_cnt = 0;
	//edge count
	int h_edge_cnt = 0;
	int v_edge_cnt = 0;
	//noisy count
	int noisy_cnt = 0;
	// noisy value
	double noisy_value = 0;
	
	//mean Dh(x,y) in the paper 
	// in code it means Dh(x,y) and Ax(x,y)
	double* phEdgeMatric = new double[total];
	double* pvEdgeMatric = new double[total];
	// for noisy
	//Dh Dv in the paper
	double* phNoisyMatric = new double[total];
	double* pvNoisyMatric = new double[total];
	//Ncond in the paper
	double * NoisyM = new double[total];

	//means Ch(x,y) Cv(x,y) in the paper
	double* tmpH = new double[total];
	double* tmpV = new double[total];
	

	//for blur and noisy
	//loop 1
	for(int i = 0; i < nHei; i ++)
	{
		uchar* pOffset = pData;
		uchar* pNoisyOff = pNoisyData;
		steps = i*nWid;	

		for(int j = 0; j < nWid; j ++)
		{	
			int nSteps = steps + j;
			if(i == 0 || i == nHei -1)
			{
				//for edge
				phEdgeMatric[nSteps] = 0;
				pvEdgeMatric[nSteps] = 0;
				//for noisy
				phNoisyMatric[nSteps] = 0;
				pvNoisyMatric[nSteps] = 0;
			}
			else if(j == 0 || j == nWid -1)
			{
				//for edge
				phEdgeMatric[nSteps] = 0;
				pvEdgeMatric[nSteps] = 0;
				//for noisy
				phNoisyMatric[nSteps] = 0;
				pvNoisyMatric[nSteps] = 0;
			}
			else
			{
				//for edge
				ch = abs(*(pOffset-1) - *(pOffset+1)) * 1.0 / 255.0;
				phEdgeMatric[nSteps] = ch;
				ghMean += ch;

				cv = abs(*(pOffset-nWid) - *(pOffset+nWid)) * 1.0 / 255.0;
				pvEdgeMatric[nSteps] = cv;
				gvMean += cv;

				//for noisy
				ch = abs(*(pNoisyOff-1) - *(pNoisyOff+1)) * 1.0 / 255.0;
				phNoisyMatric[nSteps] = ch;
				gNoisyhMean += ch;
				cv = abs(*(pNoisyOff-nWid) - *(pNoisyOff+nWid)) * 1.0 / 255.0;
				pvNoisyMatric[nSteps] = cv;
				gNoisyvMean += cv;
			}
			
			double tmp_blur_value = 0;
			double tmp_ch = 0;
			double tmp_cv = 0;
			ch = (phEdgeMatric[nSteps] / 2);
			if(ch != 0)
				tmp_ch = abs((*pOffset) * 1.0 / 255 - ch) * 1.0 / ch;	
			cv = (pvEdgeMatric[nSteps] / 2);
			if(cv != 0)
				tmp_cv = abs((*pOffset) * 1.0 / 255 - cv) * 1.0 / cv;

			tmp_blur_value = max(tmp_ch,tmp_cv);
		//	blurvalue += tmp_blur_value;
			if(tmp_blur_value > blur_th) 
			{
				blur_cnt ++;
				blurvalue += tmp_blur_value;
			}

			pOffset ++;
			pNoisyOff ++;
		}
		pData += iLineBytes;
		pNoisyData += iNoisyBytes;
	}

	//for edge and noisy
	//for edge
	ghMean /= (total);
	gvMean /= (total);	
	//noisy
	gNoisyhMean /= total;
	gNoisyvMean /= total;

	//loop 2
	for(int i = 0; i < nHei; i ++)
	{
		steps = i*nWid;
		for(int j = 0; j < nWid; j ++)
		{
			int nSteps = steps + j;
			ch = phEdgeMatric[nSteps];
			tmpH[nSteps] = ch > ghMean ?  ch : 0;
			cv = pvEdgeMatric[nSteps];
			tmpV[nSteps] = cv > gvMean ?  cv : 0;

			ch = phNoisyMatric[nSteps];
			cv = pvNoisyMatric[nSteps];
			if(ch <= gNoisyhMean && cv <= gNoisyvMean)
			{
				NoisyM[nSteps] = max(ch,cv);
			}
			else
				NoisyM[nSteps] = 0;

			gNoisyMean += NoisyM[nSteps];
		}
	}
	gNoisyMean /= total;

	//loop 3
	for(int i = 0; i < nHei; i ++)
	{
		steps = i*(nWid);
		for(int j = 0; j < nWid; j ++)
		{
			int nSteps = steps + j;
			//for edge
			if(i == 0 || i == nHei -1)
			{
			//	phEdge[steps+j] = 0;
			//	pvEdge[steps+j] = 0;
			}
			else if(j == 0 || j == nWid -1)
			{
			//	phEdge[steps+j] = 0;
			//	pvEdge[steps+j] = 0;
			}
			else
			{
				//for edge
				if(tmpH[nSteps] > tmpH[nSteps-1] && tmpH[nSteps] > tmpH[nSteps+1])
				{
				//	phEdge[steps+j] = 1;
					h_edge_cnt ++;
				}
				//else phEdge[steps+j] = 0;

				if(tmpV[nSteps] > tmpV[steps-nWid] && tmpV[nSteps] > tmpV[steps+nWid])
				{
				//	pvEdge[steps+j] = 1;
					v_edge_cnt ++;
				}
			//	else pvEdge[steps+j] = 0;
				
				if(NoisyM[nSteps] > gNoisyMean)
				{
					noisy_cnt++;
					noisy_value += NoisyM[nSteps];
				}

			}
		
		}
	}

	if(pGrayImg){cvReleaseImage(&pGrayImg);pGrayImg = NULL;}
	if(pNoisyImg){cvReleaseImage(&pNoisyImg);pNoisyImg = NULL;}
	if(phEdgeMatric){delete []phEdgeMatric; phEdgeMatric = NULL;}
	if(pvEdgeMatric){delete []pvEdgeMatric; pvEdgeMatric = NULL;}
	if(phNoisyMatric){delete []phNoisyMatric; phNoisyMatric = NULL;}
	if(pvNoisyMatric){delete []pvNoisyMatric; pvNoisyMatric = NULL;}
	if(NoisyM){delete []NoisyM; NoisyM = NULL;}
	if(tmpH){delete []tmpH; tmpH = NULL;}
	if(tmpV){delete []tmpV; tmpV = NULL;}

	if ( ( blur_cnt == 0) || ( (h_edge_cnt+v_edge_cnt) == 0) || ( noisy_cnt == 0) || ( total == 0) )
		return -1;

	blur_mean = blurvalue * 1.0 / blur_cnt;
	blur_ratio = blur_cnt * 1.0 / (h_edge_cnt+v_edge_cnt);

	nosiy_mean = noisy_value * 1.0 / noisy_cnt;
	nosiy_ratio = noisy_cnt * 1.0 / total;

	//the para is provided by paper
	//another para 1.55 0.86 0.24 0.66
	double gReulst = 1 -(blur_mean + 0.95 * blur_ratio + \
						 nosiy_mean * 0.3 + 0.75 * nosiy_ratio);

	fBlur.push_back( blur_mean );
	fBlur.push_back( blur_ratio );
	fBlur.push_back( nosiy_mean );
	fBlur.push_back( nosiy_ratio );
	fBlur.push_back( gReulst );

	return TOK;
}

int API_COMMEN::ExtractFeat_Constract( IplImage* pSrcImg, vector< float > &fContrast )		//4D
{
	if(pSrcImg->nChannels != 3) {
		printf("preprocess fun ImageQuality_constract img is gray\n");
		return -1;
	}

	int nWid = pSrcImg->width; int nHei = pSrcImg->height;

	IplImage* pRGBImg[3];
	IplImage* pGrayImg = NULL;
	for(int i =0; i < 3; i ++)
	{
		pRGBImg[i] = cvCreateImage(cvSize(nWid,nHei),pSrcImg->depth,1);
	}
	pGrayImg = cvCreateImage(cvSize(nWid,nHei),pSrcImg->depth,1);
	cvCvtColor(pSrcImg,pGrayImg,CV_BGR2GRAY);
	cvSplit(pSrcImg,pRGBImg[0],pRGBImg[1],pRGBImg[2],0);

	double rVal = 0;
	double gVal = 0;
	double bVal = 0;
	double grayVal = 0;

	rVal = ExtractFeat_Constract_GetBlockDev(pRGBImg[0]);
	gVal = ExtractFeat_Constract_GetBlockDev(pRGBImg[1]);
	bVal = ExtractFeat_Constract_GetBlockDev(pRGBImg[2]);
	grayVal = ExtractFeat_Constract_GetBlockDev(pGrayImg);

	fContrast.push_back( rVal );
	fContrast.push_back( gVal );
	fContrast.push_back( bVal );
	fContrast.push_back( grayVal );

	if(pGrayImg){cvReleaseImage(&pGrayImg);pGrayImg = NULL;}
	for(int i = 0; i < 3; i ++)
	{
		cvReleaseImage(&pRGBImg[i]);
		pRGBImg[i] = NULL;
	}
	
	return 0;
}

double API_COMMEN::ExtractFeat_Constract_GetBlockDev(IplImage* ScaleImg)
{
	int Imgwidth = ScaleImg->width;
	int Imgheight = ScaleImg->height;
	int Bl_width = Imgwidth/3, Bl_height = Imgheight/3;
	CvRect BlockImg[BLOCKNUM] = {
		cvRect(0,0,Bl_width,Bl_height),
		cvRect(Imgwidth*1/3,0,Bl_width,Bl_height),
		cvRect(Imgwidth*2/3,0,Bl_width,Bl_height),
		cvRect(0,Imgheight/3,Bl_width,Bl_height),
		cvRect(Imgwidth/3,Imgheight/3,Bl_width,Bl_height),
		cvRect(Imgwidth*2/3,Imgheight/3,Bl_width,Bl_height),
		cvRect(0,Imgheight*2/3,Bl_width,Bl_height),
		cvRect(Imgwidth*1/3,Imgheight*2/3,Bl_width,Bl_height),
		cvRect(Imgwidth*2/3,Imgheight*2/3,Bl_width,Bl_height)
	};
	//	IplImage *blockimage[BLOCKNUM];
	IplImage* blockimage = NULL;
	blockimage = cvCreateImage(cvSize(Bl_width,Bl_height),IPL_DEPTH_8U,1);
	vector <float> dev;
	CvScalar Bl_mean[BLOCKNUM];
	CvScalar Bl_dev[BLOCKNUM];

	for (int i=0;i<BLOCKNUM;i++)
	{
		cvResetImageROI(ScaleImg);
		cvSetImageROI(ScaleImg,BlockImg[i]);
		cvCopy(ScaleImg,blockimage);

		cvAvgSdv(blockimage,&Bl_mean[i],&Bl_dev[i],NULL);
		dev.push_back(Bl_dev[i].val[0]);
		
	}

	cvResetImageROI(ScaleImg);
	cvReleaseImage(&blockimage);

	CvMat *dev_data;
	dev_data = cvCreateMatHeader(BLOCKNUM,1,CV_32FC1);
	cvSetData(dev_data,&dev[0],sizeof(float));

	CvScalar Bl_Ldev;   //对比度名明显度参数；
	cvAvgSdv(dev_data,NULL,&Bl_Ldev,NULL);

//	cvReleaseMatHeader(&dev_data);
	cvReleaseMat(&dev_data);
	return Bl_Ldev.val[0];
}

/***********************************Count_FaceRectFromPoint*************************************/
int API_COMMEN::Count_FaceRectFromPoint( int loadPoint[], int width, int height, vector< pair<string, Vec4i> > &FacePoint )
{
	int fx,fy,fw,fh;
	float tmp,max_h,w_eye,w_mouse,h_eye2nose,h_nose2mouse,center_eye_x;

	//string Dict_6class[6] = {"face","eye","mouse","nose","hair","bread"};
	string Dict_7class[7] = {"face","halfface","eye","mouse","nose","hair","bread"};

	FacePoint.clear();
	
	//commen
	{
		w_eye = fabs(loadPoint[2]-loadPoint[0]);
		h_eye2nose = fabs(loadPoint[5]-(loadPoint[1]+loadPoint[3])*0.5);  //eye to nose
		h_nose2mouse = fabs((loadPoint[7]+loadPoint[9])*0.5-loadPoint[5]);  //nose to mouse
		max_h = (h_eye2nose>h_nose2mouse)?h_eye2nose:h_nose2mouse;
	}

	//unnormal face	
	tmp = fabs(loadPoint[5]-(loadPoint[1]+loadPoint[3])*0.5);
	if ( (w_eye*1.25<tmp) || (w_eye<10) )
	{
		return -1;
	}
	
	//count face
	{
		fw = int(w_eye*2.5+0.5);
		fx = int(fabs(loadPoint[2]+loadPoint[0]-fw)*0.5+0.5);			
		fh = int(max_h*2.5+h_eye2nose+h_nose2mouse+0.5);
		fy = int(fabs((loadPoint[1]+loadPoint[3])*0.5-max_h*1.5)+0.5);

		if ( (fx>0) && (fy>0) && (fw>0) && (fh>0) && (fx+fw<=width) && (fy+fh<=height) && (fx<fx+fw) && (fy<fy+fh) )
		{
			Vec4i face(fx,fy,fx+fw,fy+fh);
			FacePoint.push_back(make_pair(Dict_7class[0],face));
		}
		else
		{
			printf("face:w_%d,h_%d,fx_%d,fy_%d,fw_%d,fh_%d\n",width,height,fx,fy,fw,fh);
		}
	}

	//count half face
	{
		fh = int(max_h*2.5+h_eye2nose+h_nose2mouse+0.5);
		fy = int(fabs((loadPoint[1]+loadPoint[3])*0.5-max_h*1.5)+0.5);

		center_eye_x = (loadPoint[0]+loadPoint[2])*0.5;
		if (loadPoint[4]<=center_eye_x)
		{
			fw = int(w_eye*2.5+0.5);	//face-w
			tmp = int(fabs(loadPoint[2]+loadPoint[0]-fw)*0.5+0.5);	//face-x1
			fx = int(loadPoint[4]-0.1*w_eye+0.5);	//half face x1
			fw = int(w_eye*2.5-(fx-tmp)+0.5);
		}
		else
		{
			fw = int(w_eye*2.5+0.5);	//face-w
			tmp = int(loadPoint[4]+0.1*w_eye+0.5);	//half face x1
			fx = int(fabs(loadPoint[2]+loadPoint[0]-fw)*0.5+0.5);	//face-x1
			fw = int(tmp-fx);			
		}

		if ( (fx>0) && (fy>0) && (fw>0) && (fh>0) && (fx+fw<=width) && (fy+fh<=height) && (fx<fx+fw) && (fy<fy+fh) )
		{
			Vec4i halfface(fx,fy,fx+fw,fy+fh);
			FacePoint.push_back(make_pair(Dict_7class[1],halfface));
		}
		else
		{
			printf("halfface:w_%d,h_%d,fx_%d,fy_%d,fw_%d,fh_%d\n",width,height,fx,fy,fw,fh);
		}
	}

	//count eye
	{
		fw = int(w_eye*2+0.5);
		fx = int(fabs(loadPoint[0]*3-loadPoint[2])*0.5+0.5);
		fh = int(max_h*1.5+0.5);
		fy = int(fabs((loadPoint[1]+loadPoint[3])*0.5-max_h*0.75)+0.5);

		if ( (fx>0) && (fy>0) && (fw>0) && (fh>0) && (fx+fw<=width) && (fy+fh<=height) && (fx<fx+fw) && (fy<fy+fh) )
		{
			Vec4i eye(fx,fy,fx+fw,fy+fh);
			FacePoint.push_back(make_pair(Dict_7class[2],eye));
		}
		else
		{
			printf("eye:w_%d,h_%d,fx_%d,fy_%d,fw_%d,fh_%d\n",width,height,fx,fy,fw,fh);
		}
	}

	//count mouse
	{
		w_mouse = fabs(loadPoint[8]-loadPoint[6]);
		fw = int(w_mouse*2+0.5);
		fx = int(fabs(loadPoint[6]*3-loadPoint[8])*0.5+0.5);
		fh = int(max_h*1.5+0.5);
		fy = int(fabs((loadPoint[7]+loadPoint[9])*0.5-max_h*0.5)+0.5);

		if ( (fx>0) && (fy>0) && (fw>0) && (fh>0) && (fx+fw<=width) && (fy+fh<=height) && (fx<fx+fw) && (fy<fy+fh) )
		{
			Vec4i mouse(fx,fy,fx+fw,fy+fh);
			FacePoint.push_back(make_pair(Dict_7class[3],mouse));
		}
		else
		{
			printf("mouse:w_%d,h_%d,fx_%d,fy_%d,fw_%d,fh_%d\n",width,height,fx,fy,fw,fh);
		}
	}

	//count nose
	{
		fw = (w_eye>w_mouse)?w_eye:w_mouse;
		fx = int(fabs(loadPoint[4]-fw*0.5)+0.5);
		fh = int((h_eye2nose+h_nose2mouse)*0.8+0.5);
		fy = int(fabs(loadPoint[5]-h_eye2nose*0.8)+0.5);

		if ( (fx>0) && (fy>0) && (fw>0) && (fh>0) && (fx+fw<=width) && (fy+fh<=height) && (fx<fx+fw) && (fy<fy+fh) )
		{
			Vec4i nose(fx,fy,fx+fw,fy+fh);
			FacePoint.push_back(make_pair(Dict_7class[4],nose));
		}
		else
		{
			printf("nose:w_%d,h_%d,fx_%d,fy_%d,fw_%d,fh_%d\n",width,height,fx,fy,fw,fh);
		}
	}

	//count hair
	{
		fw = int(w_eye*2.5+0.5);
		fx = int(fabs(loadPoint[2]+loadPoint[0]-fw)*0.5+0.5);
		tmp = (loadPoint[1]+loadPoint[3])*0.5-max_h*3;
		fy = (tmp<1)?1:(int(tmp+0.5));
		fh = (tmp<1)?(int(max_h*2+0.5)):(int(max_h*2.5+0.5));

		if ( (fx>0) && (fy>0) && (fw>0) && (fh>0) && (fx+fw<=width) && (fy+fh<=height) && (fx<fx+fw) && (fy<fy+fh) )
		{
			Vec4i hair(fx,fy,fx+fw,fy+fh);
			FacePoint.push_back(make_pair(Dict_7class[5],hair));
		}
		else
		{
			printf("hair:w_%d,h_%d,fx_%d,fy_%d,fw_%d,fh_%d\n",width,height,fx,fy,fw,fh);
		}
	}

	//count bread
	{
		fw = int(w_eye*2+0.5);
		fx = int(fabs(loadPoint[6]+loadPoint[8]-fw)*0.5+0.5);
		fh = int(max_h*1.5+0.5);
		tmp = (loadPoint[7]+loadPoint[9])*0.5+max_h*1.75;
		fy = (tmp>height)?(int(height-fh)):(int(tmp-fh+0.5));

		if ( (fx>0) && (fy>0) && (fw>0) && (fh>0) && (fx+fw<=width) && (fy+fh<=height) && (fx<fx+fw) && (fy<fy+fh) )
		{
			Vec4i bread(fx,fy,fx+fw,fy+fh);
			FacePoint.push_back(make_pair(Dict_7class[6],bread));
		}
		else
		{
			printf("bread:w_%d,h_%d,fx_%d,fy_%d,fw_%d,fh_%d\n",width,height,fx,fy,fw,fh);
		}
	}

	return 0;
}

int API_COMMEN::Rect_IOU(Vec4i rect1, Vec4i rect2, float &iou) 
{ 
	if ( (rect1[0]<0)||(rect1[1]<0)||(rect1[0]>rect1[2])||(rect1[1]>rect1[3])||
		 (rect2[0]<0)||(rect2[1]<0)||(rect2[0]>rect2[2])||(rect2[1]>rect2[3]))
		return TEC_INVALID_PARAM;
	
	float xx1,yy1,xx2,yy2,w,h,tmpArea,area1,area2;

	iou = 0;
	area1 = (rect1[2]-rect1[0])*(rect1[3]-rect1[1]);
	area2 = (rect2[2]-rect2[0])*(rect2[3]-rect2[1]);
	
	xx1=max(rect1[0],rect2[0]);
	yy1=max(rect1[1],rect2[1]);
	xx2=min(rect1[2],rect2[2]);
	yy2=min(rect1[3],rect2[3]);
	w=xx2-xx1+1;
	h=yy2-yy1+1;
	tmpArea = w*h;
	if ( (w>0)&&(h>0)&&(tmpArea>0)&&(area1>0)&&(area2>0)&&(area1+area2>tmpArea) )
	{
		iou = tmpArea*1.0/(area1+area2-tmpArea);
	}

	return 0;
}

