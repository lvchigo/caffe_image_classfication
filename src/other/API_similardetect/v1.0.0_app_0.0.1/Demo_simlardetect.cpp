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
	for (int i=tmpPath.rfind('/')+1;i<tmpPath.rfind('.');i++)
	{
		atom = filepath[i] - '0';
		if (atom < 0 || atom >9)
			break;
		ID = ID * 10 + atom;
	}
	return ID;
}

string GetSrtingIDFromFilePath(const char *filepath)
{
	string ID;
	char szID[512] = {0};
	string tmpPath = filepath;
	for (int i=tmpPath.rfind('/')+1;i<tmpPath.rfind('.');i++)
	{
		szID[i] = filepath[i];
	}
	ID = szID;
	return ID;
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

unsigned char * ImageChange(IplImage *src, int &width, int &height, int &nChannel )
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

	for (i=0; i<width; ++i)
    {
        for (j=0; j<height; ++j)
        {
        	for (k=0; k<nChannel; ++k)
        	{
				dst[i*height*nChannel+j*nChannel+k] = 
					((unsigned char *)(src->imageData + j*src->widthStep))[i*src->nChannels + k];
        	}
		}
	}

	return dst;
}

int SimilarDetectSingle( char *szFileList, char *szKeyFiles )
{
	IplImage *img = 0;
	int  width, height, nChannel, res_mode, ret = 0;
	long ImageID;
	unsigned long long randomID;
	UInt64 res_iid;
	char szImgPath[512];
	FILE *fpListFile = 0 ;
	unsigned char *img_data = NULL;

#ifdef _TIME_STATISTICS_
    struct timeval tv1;
    struct timeval tv2;
	double timeOnce;
#endif

	fpListFile = fopen(szFileList,"r");
	if (!fpListFile) 
	{
		cout << "0.can't open" << szFileList << endl;
		goto ERR;
	}

	ret  = SD_GLOBAL_1_1_0::Init(szKeyFiles); //加载资源
	if (ret != 0)
	{
	   cout<<"can't open"<< szKeyFiles<<endl;
	   goto ERR;
	}
	
	while( EOF != fscanf(fpListFile, "%s", szImgPath))
	{
		img = cvLoadImage(szImgPath, 1);					//待提取特征图像文件
		if(!img || (img->width<32) || (img->height<32) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U)
		{	
			cout<<"can't open " << szImgPath << ",or unsupported image format!! "<< endl;
			cvReleaseImage(&img);img = 0;
			continue;
		}	

		getRandomID( randomID );
		//ImageID = GetIDFromFilePath(szImgPath);

		img_data = ImageChange( img, width, height, nChannel );
		
#ifdef _TIME_STATISTICS_	
		gettimeofday(&tv1, NULL);
#endif //_TIME_STATISTICS_
		ret = SD_GLOBAL_1_1_0::SimilarDetect(img_data, width, height, nChannel, randomID, res_mode, res_iid );
#ifdef _TIME_STATISTICS_
		gettimeofday(&tv2, NULL);
		timeOnce = difftimeval(&tv2, &tv1);
		timeSum += timeOnce; 
		nCount++;
#endif //_TIME_STATISTICS_

		if (TOK == ret)
		{
			printf("%lld\t%lld\t%d\n",randomID,res_iid,res_mode);
		}
		else
		{
			cout<< szImgPath << " ERR!! "<< endl;
			cvReleaseImage(&img);img = 0;
			if (img_data) {delete [] img_data;img_data = NULL;}
			continue;
		}
		
		cvReleaseImage(&img);img = 0;
		if (img_data) {delete [] img_data;img_data = NULL;}
	}

ERR:
	//释放资源
	SD_GLOBAL_1_1_0::Uninit();

	if (img)	
		cvReleaseImage(&img);
	if (img_data) {delete [] img_data;img_data = NULL;}

	if (fpListFile) {
		fclose(fpListFile);
		fpListFile = 0;
	}

	printf("SimilarDetect Done!\n");
#ifdef _TIME_STATISTICS_	
	if (nCount)
		printf("Average Extraction time = %lf\n",timeSum / nCount);
#endif //_TIME_STATISTICS_
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

	if ( argc == 4 && strcmp(argv[1],"-simdetect") == 0)
	{
		strcpy(szKeyFiles, argv[3]);
		PadEnd(szKeyFiles);
		SimilarDetectSingle(argv[2], szKeyFiles );
	}
	else
	{
		cout << "usage:\n" << endl;
		cout << "\tSimilarDetect: Demo -simdetect ImageList.txt keyFilePath\n" << endl;
	}
	return ret;
}


