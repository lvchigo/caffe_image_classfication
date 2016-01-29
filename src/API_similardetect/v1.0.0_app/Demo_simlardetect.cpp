#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <dirent.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/stat.h> 

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include "cv.h"
#include "cxcore.h"
#include "highgui.h"

#include "TErrorCode.h"
#include "SD_global.h"

using namespace std;
using namespace cv;

#define _TIME_STATISTICS_
#define THREAD_NUM    8

#ifdef _TIME_STATISTICS_
#include <sys/time.h>

static double timeSum = 0.0;
static int  nCount = 0;

typedef unsigned long long UInt64;

double difftimeval(const struct timeval *tv1, const struct timeval *tv2)
{
        double d;
        time_t s;
        suseconds_t u;

        s = tv1->tv_sec - tv2->tv_sec;
        u = tv1->tv_usec - tv2->tv_usec;
        d = 1000000.0 * s  + u ;
        return d;
}
#endif //_TIME_STATISTICS_

inline long GetIDFromFilePath(const char *filepath)
{
	long ID = 0;
	int atom =0;
	string tmpPath = filepath;
	for (int i=tmpPath.rfind("/")+1;i<tmpPath.rfind(".");i++)
	{
		atom = filepath[i] - '0';
		if (atom < 0 || atom >9)
			break;
		ID = ID * 10 + atom;
	}
	return ID;
}

string GetStringIDFromFilePath(const char *filepath)
{
        string tmpPath = filepath;
        string iid;

        long start = tmpPath.find_last_of('/');
        long end = tmpPath.find_last_of('.');

        if ( (start>0) && (end>0) && ( end>start ) )
                iid = tmpPath.substr(start+1,end-start-1);

        return iid;
}

/***********************************getRandomID***********************************/
void getRandomID( unsigned long long &randomID )
{
	int i,atom =0;
	randomID = 0;
	
	//time
	time_t nowtime = time(NULL);
	tm *now = localtime(&nowtime);

	char szTime[1024];
	sprintf(szTime, "%04d%02d%02d%02d%02d%02d%d",
			now->tm_year+1900, now->tm_mon+1, now->tm_mday,now->tm_hour, now->tm_min, now->tm_sec,(rand()%10000) );
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

unsigned char * Image_Change(IplImage *src, int &width, int &height, int &nChannel )
{
	if(!src || (src->width<16) || (src->height<16) || src->nChannels != 3 || src->depth != IPL_DEPTH_8U)
    {
        cvReleaseImage(&src);src = 0;
        return NULL;
    }

	int i,j,k;
	width = src->width;
	height = src->height;
	nChannel = src->nChannels;
	int nSize = width * height * nChannel;
	unsigned char *dst = new unsigned char[nSize];

	for (i=0; i<height; ++i)
    {
        for (j=0; j<width; ++j)
        {
        	for (k=0; k<nChannel; ++k)
        	{
				dst[i*width*nChannel+j*nChannel+k] = 
					((unsigned char *)(src->imageData + i*src->widthStep))[j*src->nChannels + k];
        	}
		}
	}

	return dst;
}

int Filter_Get_Image_Feat( string ImagePath, IplImage *ImageMedia, string &ImageID, string &feat )
{
	IN_IMAGE_SIMILAR_DETECT_1_0_0 api_in_image_similar_detect;
	IplImage *src = 0;
	int  i, j, width, height, nChannel, feat_len, ret = 0;

	//load image
	src = cvLoadImage(ImagePath.c_str(), 1);					//待提取特征图像文件
	if(!src || (src->width<16) || (src->height<16) || src->nChannels != 3 || src->depth != IPL_DEPTH_8U)
    {
        if (src) {cvReleaseImage(&src);src = 0;}
        return TEC_FILE_BASE;
    }

	//resize image
	cvResize(src, ImageMedia);

	//change image
	ImageID = GetStringIDFromFilePath(ImagePath.c_str());
	unsigned char *img_data = Image_Change( ImageMedia, width, height, nChannel );

	//get feat && score
	ret = api_in_image_similar_detect.Filter_Get_Feat_Score(img_data, width, height, nChannel, feat);
	if ( ret!=TOK )
	{
		printf( "Err Get_Feat_Score\n" );
		if (src) {cvReleaseImage(&src);src = 0;}
		if (img_data) {delete [] img_data;img_data = NULL;}
		return TEC_FILE_BASE;
	}

	//delete
	if (src) {cvReleaseImage(&src);src = 0;}
	if (img_data) {delete [] img_data;img_data = NULL;}

	return ret;
}

int Get_Image_Feat( string ImagePath, IplImage *ImageMedia, string &ImageID, string &feat )
{
	IN_IMAGE_SIMILAR_DETECT_1_0_0 api_in_image_similar_detect;
	IplImage *src = 0;
	int  i, j, width, height, nChannel, feat_len, ret = 0;

	//load image
	src = cvLoadImage(ImagePath.c_str(), 1);					//待提取特征图像文件
	if(!src || (src->width<16) || (src->height<16) || src->nChannels != 3 || src->depth != IPL_DEPTH_8U)
    {
        if (src) {cvReleaseImage(&src);src = 0;}
        return TEC_INVALID_PARAM;
    }

	//resize image
	cvResize(src, ImageMedia);

	//change image
	ImageID = GetStringIDFromFilePath(ImagePath.c_str());
	unsigned char *img_data = Image_Change( ImageMedia, width, height, nChannel );

	//get feat && score
	vector<double> tTime;
	//ret = api_in_image_similar_detect.Filter_Get_Feat_Score(img_data, width, height, nChannel, feat);
	ret = api_in_image_similar_detect.Album_Get_Feat_Score_Test(img_data, width, height, nChannel, feat, tTime);
	if ( ret!=TOK )
	{
		printf( "Err Get_Feat_Score\n" );
		if (src) {cvReleaseImage(&src);src = 0;}
		if (img_data) {delete [] img_data;img_data = NULL;}
		return TEC_INVALID_PARAM;
	}

	//delete
	if (src) {cvReleaseImage(&src);src = 0;}
	if (img_data) {delete [] img_data;img_data = NULL;}

	return ret;
}

int Filter_SimilarDetect( char *szFileList, char *szKeyFiles, char *savePath )
{
	IplImage *img = 0;
	int  i, j, mode_same, mode_quality, maxSingle, maxSum, ret = 0;
	long nCount = 0;
	string ImageID1, ImageID2;
	float quality1,quality2;
	double dist_cld,dist_ehd,Dist;
	double allGet_Image_Feat,tGet_Image_Feat,allSimilarDetect,tSimilarDetect;
	char szImgPath[512];
	char sv_char_Text[512];
	FILE *fpListFile = 0 ;
	vector < string > vecImagePath;
	IN_IMAGE_SIMILAR_DETECT_1_0_0 api_in_image_similar_detect;

	fpListFile = fopen(szFileList,"r");
	if (!fpListFile) 
	{
		printf("Error!!can't open %s\n",szFileList);
		if (img) {cvReleaseImage(&img);img = 0;}
		if (fpListFile) {fclose(fpListFile);fpListFile = 0;}
		return TEC_FILE_BASE;
	}

	//load path
	while( EOF != fscanf(fpListFile, "%s", szImgPath))
	{
		img = cvLoadImage(szImgPath, 1);					//待提取特征图像文件
		if(!img || (img->width<32) || (img->height<32) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U)
		{	
			cout<<"can't open " << szImgPath << ",or unsupported image format!! "<< endl;
			if (img) {cvReleaseImage(&img);img = 0;}
			continue;
		}	
		vecImagePath.push_back(string(szImgPath));

		if (img) {cvReleaseImage(&img);img = 0;}
	}
	if (img) {cvReleaseImage(&img);img = 0;}
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}

	//judge
	if (vecImagePath.size()<2)
	{
		printf("Error!!vecImagePath.size():%d\n",vecImagePath.size());
		return TEC_FILE_BASE;
	}

	//count data
	nCount = 0;
	allGet_Image_Feat = 0.0;
    allSimilarDetect = 0.0;
	printf( "iid1\tiid2\tres_mode\tdist_cld\tdist_ehd\tDist\n" );
	for(i=0;i<vecImagePath.size()-1;i++)
	{
		IplImage *ImageMedia1 = cvCreateImage(cvSize(256, 256), 8, 3);	
		string feat1;
		ret = Filter_Get_Image_Feat( vecImagePath[i], ImageMedia1, ImageID1, feat1 );
		if(ret!=TOK)
		{
			cout<< vecImagePath[i] << " ERR!! "<< endl;
			if (ImageMedia1) {cvReleaseImage(&ImageMedia1);ImageMedia1 = 0;}
			continue;
		}
		for(j=i+1;j<vecImagePath.size();j++)
		{
			IplImage *ImageMedia2 = cvCreateImage(cvSize(256, 256), 8, 3);
			string feat2;

			tGet_Image_Feat = (double)getTickCount();
			ret = Filter_Get_Image_Feat( vecImagePath[j], ImageMedia2, ImageID2, feat2 );
			tGet_Image_Feat = (double)getTickCount() - tGet_Image_Feat;
            tGet_Image_Feat = tGet_Image_Feat*1000./cv::getTickFrequency();
            allGet_Image_Feat += tGet_Image_Feat;

			if(ret!=TOK)
			{
				cout<< vecImagePath[j] << " ERR!! "<< endl;
				if (ImageMedia2) {cvReleaseImage(&ImageMedia2);ImageMedia2 = 0;}
				continue;
			}
			
			mode_same = 0;
			mode_quality = 0;
			tSimilarDetect = (double)getTickCount();
			ret = api_in_image_similar_detect.Filter_SimilarDetect_Test(feat1, feat2, mode_same, mode_quality, maxSingle, maxSum );
			//ret = api_in_image_similar_detect.Filter_SimilarDetect(feat1, feat2, mode_same, mode_quality );
			tSimilarDetect = (double)getTickCount() - tSimilarDetect;
            tSimilarDetect = tSimilarDetect*1000./cv::getTickFrequency();
            allSimilarDetect += tSimilarDetect;
			if (TOK == ret)
			{
				printf( "%s\t%s\t%d\t%d\t%d\t%d\n", 
					ImageID1.c_str(), ImageID2.c_str(), mode_same, mode_quality, maxSingle, maxSum );
			}
			else
			{
				cout<< vecImagePath[j] << " ERR!! "<< endl;
				if (ImageMedia2) {cvReleaseImage(&ImageMedia2);ImageMedia2 = 0;}
				continue;
			}

			//save check image
			IplImage* correspond = cvCreateImage( cvSize(512, 256), 8, 3 );
			cvSetImageROI( correspond, cvRect( 0, 0, 256, 256 ) );
			cvCopy( ImageMedia1, correspond );
			cvSetImageROI( correspond, cvRect( 256, 0, 256, 256 ) );
			cvCopy( ImageMedia2, correspond );
			cvResetImageROI( correspond );

			//save image
			if ( mode_same != 0 )
			{
				Mat matImg(correspond);
	            sprintf(szImgPath, "%s/%s_%s_%d_%d_%d_%d.jpg",
	                        savePath, ImageID1.c_str(), ImageID2.c_str(), mode_same, mode_quality, maxSingle, maxSum );
				imwrite(szImgPath, matImg );
			}

			nCount++;
			if( nCount%50 == 0 )
            	printf( "Loaded %ld img...\n", nCount );

			if (ImageMedia2) {cvReleaseImage(&ImageMedia2);ImageMedia2 = 0;}
			if (correspond) {cvReleaseImage(&correspond);correspond = 0;}
		}
		if (ImageMedia1) {cvReleaseImage(&ImageMedia1);ImageMedia1 = 0;}
	}

	/*********************************Print Info*********************************/
    if ( vecImagePath.size() != 0 )
    {
	    printf( "nCount:%ld,allGet_Image_Feat:%.2fms,allGet_Image_Feat:%.2fms\n", \
	    	nCount, allGet_Image_Feat*1.0/nCount, allSimilarDetect*1.0/nCount );
    }
	//print
	printf("SimilarDetect Done!\n");

	return ret;
}

int SimilarDetect( char *szFileList, char *szKeyFiles, char *savePath )
{
	IplImage *img = 0;
	int  i, j, mode_same, mode_quality, ret = 0;
	long nCount = 0;
	string ImageID1, ImageID2;
	float quality1,quality2;
	double dist_cld,dist_ehd,Dist;
	double allGet_Image_Feat,tGet_Image_Feat,allSimilarDetect,tSimilarDetect;
	char szImgPath[512];
	char sv_char_Text[512];
	FILE *fpListFile = 0 ;
	vector < string > vecImagePath;
	IN_IMAGE_SIMILAR_DETECT_1_0_0 api_in_image_similar_detect;
	int Num_Mode_Same[3] = {0};

	fpListFile = fopen(szFileList,"r");
	if (!fpListFile) 
	{
		printf("Error!!can't open %s\n",szFileList);
		if (img) {cvReleaseImage(&img);img = 0;}
		if (fpListFile) {fclose(fpListFile);fpListFile = 0;}
		return TEC_FILE_BASE;
	}

	//load path
	while( EOF != fscanf(fpListFile, "%s", szImgPath))
	{
		img = cvLoadImage(szImgPath, 1);					//待提取特征图像文件
		if(!img || (img->width<32) || (img->height<32) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U)
		{	
			cout<<"can't open " << szImgPath << ",or unsupported image format!! "<< endl;
			if (img) {cvReleaseImage(&img);img = 0;}
			continue;
		}	
		vecImagePath.push_back(string(szImgPath));

		if (img) {cvReleaseImage(&img);img = 0;}
	}
	if (img) {cvReleaseImage(&img);img = 0;}
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}

	//judge
	if (vecImagePath.size()<2)
	{
		printf("Error!!vecImagePath.size():%d\n",vecImagePath.size());
		return TEC_FILE_BASE;
	}

	//count data
	nCount = 0;
	allGet_Image_Feat = 0.0;
    allSimilarDetect = 0.0;
	printf( "iid1\tiid2\tres_mode\tdist_cld\tdist_ehd\tDist\n" );
	for(i=0;i<vecImagePath.size()-1;i++)
	{
		IplImage *ImageMedia1 = cvCreateImage(cvSize(256, 256), 8, 3);	
		string feat1;
		ret = Get_Image_Feat( vecImagePath[i], ImageMedia1, ImageID1, feat1 );
		if(ret!=TOK)
		{
			cout<< vecImagePath[i] << " ERR!! "<< endl;
			if (ImageMedia1) {cvReleaseImage(&ImageMedia1);ImageMedia1 = 0;}
			continue;
		}
		for(j=i+1;j<vecImagePath.size();j++)
		{
			IplImage *ImageMedia2 = cvCreateImage(cvSize(256, 256), 8, 3);
			string feat2;

			tGet_Image_Feat = (double)getTickCount();
			ret = Get_Image_Feat( vecImagePath[j], ImageMedia2, ImageID2, feat2 );
			tGet_Image_Feat = (double)getTickCount() - tGet_Image_Feat;
            tGet_Image_Feat = tGet_Image_Feat*1000./cv::getTickFrequency();
            allGet_Image_Feat += tGet_Image_Feat;

			if(ret!=TOK)
			{
				cout<< vecImagePath[j] << " ERR!! "<< endl;
				if (ImageMedia2) {cvReleaseImage(&ImageMedia2);ImageMedia2 = 0;}
				continue;
			}
			
			mode_same = 0;
			mode_quality = 0;
			tSimilarDetect = (double)getTickCount();
			//ret = api_in_image_similar_detect.SimilarDetect(feat1, feat2, res_mode );
			ret = api_in_image_similar_detect.Album_SimilarDetect_Test(feat1, feat2, mode_same, mode_quality, dist_cld, dist_ehd, Dist );
			tSimilarDetect = (double)getTickCount() - tSimilarDetect;
            tSimilarDetect = tSimilarDetect*1000./cv::getTickFrequency();
            allSimilarDetect += tSimilarDetect;
			if (TOK == ret)
			{
				printf( "%s\t%s\t%d\t%d\t%.2f\t%.2f\t%.2f\n", 
					ImageID1.c_str(), ImageID2.c_str(), mode_same, mode_quality, dist_cld, dist_ehd, Dist );
			}
			else
			{
				cout<< vecImagePath[j] << " ERR!! "<< endl;
				if (ImageMedia2) {cvReleaseImage(&ImageMedia2);ImageMedia2 = 0;}
				continue;
			}

			//save check image
			IplImage* correspond = cvCreateImage( cvSize(512, 256), 8, 3 );
			cvSetImageROI( correspond, cvRect( 0, 0, 256, 256 ) );
			cvCopy( ImageMedia1, correspond );
			cvSetImageROI( correspond, cvRect( 256, 0, 256, 256 ) );
			cvCopy( ImageMedia2, correspond );
			cvResetImageROI( correspond );

			//save image
			if ( ( mode_same != 0 ) || ( ( mode_same == 0 ) && ( dist_cld<35 ) && ( dist_ehd<190 ) ) )
			{
				Mat matImg(correspond);
	            sprintf(sv_char_Text, "q:%d", feat1[0] );
	            putText(matImg, string(sv_char_Text), cvPoint(1,20), FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(0,0,255), 2);
				sprintf(sv_char_Text, "q:%d", feat2[0] );
	            putText(matImg, string(sv_char_Text), cvPoint(257,20), FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(0,0,255), 2);
				sprintf(szImgPath, "%s/%s_%s_%d_%d_%.2f_%.2f_%.2f.jpg",
	                        savePath, ImageID1.c_str(), ImageID2.c_str(), mode_same, mode_quality, dist_cld, dist_ehd, Dist );
				imwrite(szImgPath, matImg );
			}

			//check data
			Num_Mode_Same;
			if ( mode_same == 2 )			
				Num_Mode_Same[2]++;
			else if ( mode_same == 1 )
			{
				Num_Mode_Same[1]++;
				if ( ( dist_cld>25 ) || ( dist_ehd>160 ) )
					Num_Mode_Same[0]++;
			}

			//save image
			if ( ( mode_same == 1 ) && ( ( dist_cld>25 ) || ( dist_ehd>160 ) ) )
			{
				Mat matImg(correspond);
	            sprintf(sv_char_Text, "q:%d", feat1[0] );
	            putText(matImg, string(sv_char_Text), cvPoint(1,20), FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(0,0,255), 2);
				sprintf(sv_char_Text, "q:%d", feat2[0] );
	            putText(matImg, string(sv_char_Text), cvPoint(257,20), FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(0,0,255), 2);
				sprintf(szImgPath, "%s/check_%s_%s_%d_%d_%.2f_%.2f_%.2f.jpg",
	                        savePath, ImageID1.c_str(), ImageID2.c_str(), mode_same, mode_quality, dist_cld, dist_ehd, Dist );
				imwrite(szImgPath, matImg );
			}

			nCount++;
			if( nCount%50 == 0 )
            	printf( "Loaded %ld img...\n", nCount );

			if (ImageMedia2) {cvReleaseImage(&ImageMedia2);ImageMedia2 = 0;}
			if (correspond) {cvReleaseImage(&correspond);correspond = 0;}
		}
		if (ImageMedia1) {cvReleaseImage(&ImageMedia1);ImageMedia1 = 0;}
	}

	/*********************************Print Info*********************************/
    if ( vecImagePath.size() != 0 )
    {
	    printf( "nCount:%ld,allGet_Image_Feat:%.2fms,allGet_Image_Feat:%.2fms\n", \
	    	nCount, allGet_Image_Feat*1.0/nCount, allSimilarDetect*1.0/nCount );
		printf( "nCount:%ld,mode_same:2_%d,mode_same:1_%d,mode_same:0_%d,%.2f%\n", \
			nCount, Num_Mode_Same[2], Num_Mode_Same[1], Num_Mode_Same[0], \
			Num_Mode_Same[0]*100.0/(Num_Mode_Same[1]+Num_Mode_Same[2]) );
    }
	//print
	printf("SimilarDetect Done!\n");

	return ret;
}


inline void PadEnd(char *szPath)
{
	int iLength = strlen(szPath);
	if (szPath[iLength-1] != '/')
	{
		szPath[iLength] = '/';
		szPath[iLength+1] = 0;
	}
}

int main(int argc, char* argv[])
{
	int  ret = 0;
	char szKeyFiles[256],szSavePath[256];

	if ( argc == 5 && strcmp(argv[1],"-simdetect") == 0)
	{
		strcpy(szKeyFiles, argv[3]);
		PadEnd(szKeyFiles);
		strcpy(szSavePath, argv[4]);
		PadEnd(szSavePath);
		SimilarDetect(argv[2], szKeyFiles, szSavePath );
	}
	else if ( argc == 5 && strcmp(argv[1],"-simdetect_filter") == 0)
	{
		strcpy(szKeyFiles, argv[3]);
		PadEnd(szKeyFiles);
		strcpy(szSavePath, argv[4]);
		PadEnd(szSavePath);
		Filter_SimilarDetect(argv[2], szKeyFiles, szSavePath );
	}
	else
	{
		cout << "usage:\n" << endl;
		cout << "\tSimilarDetect: Demo -simdetect ImageList.txt keyFilePath savePath\n" << endl;
		cout << "\tSimilarDetect: Demo -simdetect_filter ImageList.txt keyFilePath savePath\n" << endl;
	}
	return ret;
}


