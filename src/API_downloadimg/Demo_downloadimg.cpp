#include <string>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h> 
#include <sys/types.h> 
#include <iostream>
#include <unistd.h>
#include <pthread.h>

#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>

//img download
#include "json/json.h"
#include "downloader.h"
#include <curl/curl.h>

#include "wget_inim_ids.h"
#include "API_commen.h" 
#include "TErrorCode.h"

using namespace cv;
using namespace std;

//=================================================================================//
static pthread_mutex_t g_mutex = PTHREAD_MUTEX_INITIALIZER;
const int THREADCOUNT = 8;

typedef struct tagThreadParam
{
	int para;
	int nThreads;
	string svPath;
	int BinReSizeImg;
}ThreadParam;

typedef struct tagAdsInfo
{
	unsigned long long ImageID;
	string url;
	string label;
	float score;
}AdsInfo;

vector< pair< long, string > > gImgInfo;
vector< pair< string, string > > gImgInfo_string;
vector< AdsInfo > gAdsInfo;

//=================================================================================//

void *workThread_download (void *para)
{
	/*****************************Init*****************************/
	char inputLabel[256];
	char ImgHttpPath[256];
	char szImgPath[256];
	char rmImgPath[256];
	char charImageID[1024];
	string strImageID;
	int j, svImg, nRet = 0;
	long idx,nCount;
	unsigned long long ImageID = 0;
	/***********************************Init*************************************/
	API_COMMEN api_commen;
	API_WGET_INIM api_wget_inim;
	string url;

	ThreadParam *pParam = (ThreadParam*)para;
	
	/***********************************Start*************************************/
	for (idx = pParam->para; idx < gImgInfo.size(); idx += pParam->nThreads) 
	{
		ImageID = gImgInfo[idx].first;
		if ( ImageID < 0 )
			continue;
		if ( gImgInfo[idx].second == "null" )		//DL_DownloadImg_Num
		{
			sprintf(charImageID,"%lld",ImageID);
			strImageID = charImageID;
			pthread_mutex_lock (&g_mutex);
			url = api_wget_inim.id2url( strImageID );	
			pthread_mutex_unlock (&g_mutex);
		}
		else										//DL_DownloadImg 
		{
			url = gImgInfo[idx].second;
		}

		if( strlen(url.c_str()) == 0 )
		  	continue;
	  	cout<<url<<endl;
		
		/*****************************wget_image_file*****************************/
		sprintf( szImgPath, "%s/%lld.jpg", pParam->svPath.c_str(), ImageID );
		pthread_mutex_lock (&g_mutex);
		api_wget_inim.wget_image_file_sys( url, string(szImgPath) );
		pthread_mutex_unlock (&g_mutex);
		//cout<<"its done."<<std::endl;

		/*****************************cvLoadImage*****************************/
		IplImage *img = cvLoadImage(szImgPath);
		if(!img || (img->width<16) || (img->height<16) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
		{	
			cout<<"Can't open " << szImgPath << endl;
			
			//rm img file
			sprintf(rmImgPath, "rm %s", szImgPath );
			nRet = system(rmImgPath);

			//Release
			cvReleaseImage(&img);img = 0;
			
			continue;
		}	

		//save img data
		IplImage *ImgResize = cvCreateImage(cvSize(255, 255), img->depth, img->nChannels);
		cvResize( img, ImgResize );
		pthread_mutex_lock (&g_mutex);
		if ( pParam->BinReSizeImg == 1 )
			cvSaveImage( szImgPath,ImgResize );
		else
			cvSaveImage( szImgPath,img );
		pthread_mutex_unlock (&g_mutex);	
		cvReleaseImage(&ImgResize);ImgResize = 0;

		//Release
		cvReleaseImage(&img);img = 0;

	}

	/*******************************clr img***************************************/
	printf("thread %d Over!\n", pParam->para);
	pthread_exit (0);
}

int DL_DownloadImg_Num( long startID, long ImgNum, char *savePath, long BinReSizeImg )
{
	long i;
	gImgInfo.clear();
	
	/*********************************Load Img Data*************************************/
	for( i=startID;i<startID+ImgNum;++i )
	{
		gImgInfo.push_back( std::make_pair( i+1, "null" ) );
	}
	printf("load %d imgData...", gImgInfo.size() );

	/*********************************MutiThread*************************************/	
	{
		pthread_t *pThread = new pthread_t[THREADCOUNT];
		ThreadParam *pParam = new ThreadParam[THREADCOUNT];
		
		for(i=0; i<THREADCOUNT; ++i)
		{
			pParam[i].para = int(i);
			pParam[i].nThreads = THREADCOUNT;
			pParam[i].svPath = savePath;
			pParam[i].BinReSizeImg = BinReSizeImg;

			int rc = pthread_create(pThread+i, NULL, workThread_download,(void*)(pParam+i));
		}

		for(i=0; i<THREADCOUNT; ++i)
		{
			pthread_join(pThread[i], NULL);
		}
		sleep (1);
		delete [] pThread;
		delete [] pParam;
	}

	return 0;
}

int DL_DownloadImg_Path( char *szQueryList,char *savePath, long BinReSizeImg, int BinOnlyUrl )
{
	/***********************************Init*************************************/
	long i,nCount;
	unsigned long long ImageID;
	gImgInfo.clear();
	
	API_COMMEN api_commen;
	vector<string> cell ;
	
	/********************************Open Query List*****************************/
	std::ifstream stream_reader( szQueryList );
	std::string line;
	
	/*****************************Process one by one*****************************/
	nCount = 0;
	while( getline(stream_reader, line) )
	{
		if( strlen(line.c_str()) == 0 )
		{
		  	continue;
		}
	  	//cout<<line<<endl;

		if (BinOnlyUrl == 0)
		{
		    cell.clear() ;
			api_commen.split( line, ",", cell );
			//for(i=0;i<cell.size();i++)
			//	printf("cell.size:%d,cell[%d]:%s\n", cell.size(), i, cell[i].c_str() );

			if ( cell.size() == 2 )
			{
				gImgInfo.push_back( std::make_pair( atol(cell[0].c_str()), cell[1] ) );
			}
			else if ( cell.size() == 3 )
				gImgInfo.push_back( std::make_pair( atol(cell[0].c_str()), cell[2] ) );
			else
				continue;	
		}
		else if (BinOnlyUrl == 1)
		{
			/************************getRandomID*****************************/
			api_commen.getRandomID( ImageID );
			gImgInfo.push_back( std::make_pair( ImageID, line ) );
			
			nCount++;
			if( nCount%5000 == 0 )
			{
				printf("Loaded %ld img...\n",nCount);
				sleep(1);
			}
		}
	}
	/*********************************close file*************************************/
	stream_reader.close();
	printf("load %d imgData...", gImgInfo.size() );

	/*********************************MutiThread*************************************/	
	{
		pthread_t *pThread = new pthread_t[THREADCOUNT];
		ThreadParam *pParam = new ThreadParam[THREADCOUNT];
		
		for(i=0; i<THREADCOUNT; ++i)
		{
			pParam[i].para = int(i);
			pParam[i].nThreads = THREADCOUNT;
			pParam[i].svPath = savePath;
			pParam[i].BinReSizeImg = BinReSizeImg;

			int rc = pthread_create(pThread+i, NULL, workThread_download,(void*)(pParam+i));
		}

		for(i=0; i<THREADCOUNT; ++i)
		{
			pthread_join(pThread[i], NULL);
		}
		sleep (1);
		delete [] pThread;
		delete [] pParam;
	}

	return 0;
}

void *workThread_download_string (void *para)
{
	/*****************************Init*****************************/
	char inputLabel[256];
	char ImgHttpPath[256];
	char szImgPath[256];
	char rmImgPath[256];
	char charImageID[1024];
	string strImageID;
	int j, svImg, nRet = 0;
	long idx,nCount;
	string ImageID;
	/***********************************Init*************************************/
	API_COMMEN api_commen;
	API_WGET_INIM api_wget_inim;
	string url;

	ThreadParam *pParam = (ThreadParam*)para;
	
	/***********************************Start*************************************/
	for (idx = pParam->para; idx < gImgInfo_string.size(); idx += pParam->nThreads) 
	{
		ImageID = gImgInfo_string[idx].first;
		url = gImgInfo_string[idx].second;
		if ( ( url == "null" ) || ( strlen(url.c_str()) == 0 ) )		//DL_DownloadImg_Num
		{
			continue;
		}
	  	cout<<url<<endl;
		
		/*****************************wget_image_file*****************************/
		sprintf( szImgPath, "%s/%s.jpg", pParam->svPath.c_str(), ImageID.c_str() );
		pthread_mutex_lock (&g_mutex);
		api_wget_inim.wget_image_file_sys( url, string(szImgPath) );
		pthread_mutex_unlock (&g_mutex);
		//cout<<"its done."<<std::endl;

		/*****************************cvLoadImage*****************************/
		IplImage *img = cvLoadImage(szImgPath);
		if(!img || (img->width<16) || (img->height<16) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
		{	
			cout<<"Can't open " << szImgPath << endl;
			
			//rm img file
			sprintf(rmImgPath, "rm %s", szImgPath );
			nRet = system(rmImgPath);

			//Release
			cvReleaseImage(&img);img = 0;
			
			continue;
		}	

		//save img data
		IplImage *ImgResize = cvCreateImage(cvSize(255, 255), img->depth, img->nChannels);
		cvResize( img, ImgResize );
		pthread_mutex_lock (&g_mutex);
		if ( pParam->BinReSizeImg == 1 )
			cvSaveImage( szImgPath,ImgResize );
		else
			cvSaveImage( szImgPath,img );
		pthread_mutex_unlock (&g_mutex);	
		cvReleaseImage(&ImgResize);ImgResize = 0;

		//Release
		cvReleaseImage(&img);img = 0;

	}

	/*******************************clr img***************************************/
	printf("thread %d Over!\n", pParam->para);
	pthread_exit (0);
}

int DL_DownloadImg_Path_stringid( char *szQueryList,char *savePath, long BinReSizeImg, int BinOnlyUrl )
{
	/***********************************Init*************************************/
	long i,nCount;
	unsigned long long ImageID;
	string strImageID;
	gImgInfo_string.clear();
	
	API_COMMEN api_commen;
	vector<string> cell ;
	
	/********************************Open Query List*****************************/
	std::ifstream stream_reader( szQueryList );
	std::string line;
	
	/*****************************Process one by one*****************************/
	nCount = 0;
	while( getline(stream_reader, line) )
	{
		if( strlen(line.c_str()) == 0 )
		{
		  	continue;
		}
	  	//cout<<line<<endl;

		if (BinOnlyUrl == 0)
		{
		    cell.clear() ;
			api_commen.split( line, ",", cell );
			//for(i=0;i<cell.size();i++)
			//	printf("cell.size:%d,cell[%d]:%s\n", cell.size(), i, cell[i].c_str() );

			if ( cell.size() == 2 )
			{
				gImgInfo_string.push_back( std::make_pair( cell[0], cell[1] ) );
			}
			else if ( cell.size() == 3 )
				gImgInfo_string.push_back( std::make_pair( cell[0], cell[2] ) );
			else
				continue;	
		}
		else if (BinOnlyUrl == 1)
		{
			/************************getRandomID*****************************/
			api_commen.getRandomID( ImageID );
			strImageID = ImageID;
			gImgInfo_string.push_back( std::make_pair( strImageID, line ) );
			
			nCount++;
			if( nCount%5000 == 0 )
			{
				printf("Loaded %ld img...\n",nCount);
				sleep(1);
			}
		}
	}
	/*********************************close file*************************************/
	stream_reader.close();
	printf("load %d imgData...", gImgInfo_string.size() );

	/*********************************MutiThread*************************************/	
	{
		pthread_t *pThread = new pthread_t[THREADCOUNT];
		ThreadParam *pParam = new ThreadParam[THREADCOUNT];
		
		for(i=0; i<THREADCOUNT; ++i)
		{
			pParam[i].para = int(i);
			pParam[i].nThreads = THREADCOUNT;
			pParam[i].svPath = savePath;
			pParam[i].BinReSizeImg = BinReSizeImg;

			int rc = pthread_create(pThread+i, NULL, workThread_download_string,(void*)(pParam+i));
		}

		for(i=0; i<THREADCOUNT; ++i)
		{
			pthread_join(pThread[i], NULL);
		}
		sleep (1);
		delete [] pThread;
		delete [] pParam;
	}

	return 0;
}


int DL_DownloadImg_NoPath( char *szQueryList,char *savePath, long BinReSizeImg )
{
	/***********************************Init*************************************/
	long i,imgID=0;
	gImgInfo.clear();
	
	API_COMMEN api_commen;

	/********************************Open Query List*****************************/
	FILE *fpListFile = 0;
	fpListFile = fopen(szQueryList,"r");
	if (!fpListFile) 
	{
		cout << "0.can't open " << szQueryList << endl;
		return TEC_INVALID_PARAM;
	}
	
	/*****************************Process one by one*****************************/
	while(EOF != fscanf(fpListFile, "%ld", &imgID))
	{
		gImgInfo.push_back( std::make_pair( imgID, "null" ) );
	}
	/*********************************close file*************************************/
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	
	printf("load %d imgData...", gImgInfo.size() );

	/*********************************MutiThread*************************************/	
	{
		pthread_t *pThread = new pthread_t[THREADCOUNT];
		ThreadParam *pParam = new ThreadParam[THREADCOUNT];
		
		for(i=0; i<THREADCOUNT; ++i)
		{
			pParam[i].para = int(i);
			pParam[i].nThreads = THREADCOUNT;
			pParam[i].svPath = savePath;
			pParam[i].BinReSizeImg = BinReSizeImg;

			int rc = pthread_create(pThread+i, NULL, workThread_download,(void*)(pParam+i));
		}

		for(i=0; i<THREADCOUNT; ++i)
		{
			pthread_join(pThread[i], NULL);
		}
		sleep (1);
		delete [] pThread;
		delete [] pParam;
	}

	return 0;
}

//===============================Ads===========================================//
void *workThread_Ads (void *para)
{
	/*****************************Init*****************************/
	char inputLabel[256];
	char ImgHttpPath[256];
	char szImgPath[256];
	char saveImgPath[256];
	char rmImgPath[256];
	char charImageID[1024];
	string strImageID;
	int j, svImg, nRet = 0;
	long idx,nCount;
	unsigned long long ImageID = 0;
	/***********************************Init*************************************/
	API_COMMEN api_commen;
	API_WGET_INIM api_wget_inim;
	string url;
	string label;
	float score;

	ThreadParam *pParam = (ThreadParam*)para;
	
	/***********************************Start*************************************/
	for (idx = pParam->para; idx < gAdsInfo.size(); idx += pParam->nThreads) 
	{
		ImageID = gAdsInfo[idx].ImageID;
		url = gAdsInfo[idx].url;
		label = gAdsInfo[idx].label;
		score = gAdsInfo[idx].score;
		
		/*****************************wget_image_file*****************************/
		sprintf( szImgPath, "%s/%lld.jpg", pParam->svPath.c_str(), ImageID );
		api_wget_inim.wget_image_file_sys( url, string(szImgPath) );
		//cout<<"its done."<<std::endl;

		/*****************************cvLoadImage*****************************/
		IplImage *img = cvLoadImage(szImgPath);
		if(!img || (img->width<16) || (img->height<16) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
		{	
			printf("Can't open %s!\n label:%s,score:%.4f,url:%s!\n",szImgPath,label.c_str(),score,url.c_str());
			
			/*****************************rm img file*****************************/
			sprintf(rmImgPath, "rm %s", szImgPath );
			nRet = system(rmImgPath);

			/*********************************Release*************************************/
			cvReleaseImage(&img);img = 0;
			
			continue;
		}	

		/*****************************save img data*****************************/
		IplImage *ImgResize = cvCreateImage(cvSize(255, 255), img->depth, img->nChannels);
		cvResize( img, ImgResize );
		pthread_mutex_lock (&g_mutex);
		sprintf (saveImgPath, "%s/%s_%.4f_%lld.jpg",pParam->svPath.c_str(),label.c_str(),score,ImageID);
		if ( pParam->BinReSizeImg == 1 )
			cvSaveImage( saveImgPath,ImgResize );
		else
			cvSaveImage( saveImgPath,img );
		pthread_mutex_unlock (&g_mutex);	
		cvReleaseImage(&ImgResize);ImgResize = 0;

		/*****************************rm img file*****************************/
		sprintf(rmImgPath, "rm %s", szImgPath );
		nRet = system(rmImgPath);

		/*********************************Release*************************************/
		cvReleaseImage(&img);img = 0;
	}

	/*******************************clr img***************************************/
	printf("thread %d Over!\n", pParam->para);
	pthread_exit (0);
}

int DL_DownloadImg_OnlyUrl( char *szQueryList,char *savePath, long BinReSizeImg )
{
	/***********************************Init*************************************/
	long i;
	unsigned long long ImageID = 0;
	gAdsInfo.clear();
	
	API_COMMEN api_commen;
	vector<string> cell ;
	
	/********************************Open Query List*****************************/
	std::ifstream stream_reader( szQueryList );
	std::string line;
	
	/*****************************Process one by one*****************************/
	while( getline(stream_reader, line) )
	{
		if( strlen(line.c_str()) == 0 )
		{
		  	continue;
		}
	  	//cout<<line<<endl;
		
	    cell.clear() ;
		api_commen.split( line, ",", cell );
		//for(i=0;i<cell.size();i++)
		//	printf("cell.size:%d,cell[%d]:%s\n", cell.size(), i, cell[i].c_str() );

		/************************getRandomID*****************************/
		api_commen.getRandomID( ImageID );
		
		if ( cell.size() == 3 )
		{
			/************************push adsInfo*****************************/
			AdsInfo adsInfo;
			adsInfo.ImageID = ImageID;
			adsInfo.url = cell[0];
			adsInfo.label = cell[1];
			adsInfo.score = float(atof(cell[2].c_str()));

			/************************push adsInfo*****************************/
			if ( 	( adsInfo.ImageID < 0 ) || ( adsInfo.score < 0 ) || 
					( strlen(adsInfo.url.c_str()) == 0 ) || ( strlen(adsInfo.label.c_str()) == 0 ) )
				continue;
			gAdsInfo.push_back( adsInfo );
		}
		else if ( cell.size() == 4 )
		{
			/************************push adsInfo*****************************/
			AdsInfo adsInfo;
			adsInfo.ImageID = ImageID;
			adsInfo.url = cell[1];
			adsInfo.label = cell[2];
			adsInfo.score = float(atof(cell[3].c_str()));

			/************************push adsInfo*****************************/
			if ( 	( adsInfo.ImageID < 0 ) || ( adsInfo.score < 0 ) || 
					( strlen(adsInfo.url.c_str()) == 0 ) || ( strlen(adsInfo.label.c_str()) == 0 ) )
				continue;
			gAdsInfo.push_back( adsInfo );
		}
		else
			continue;		
	}
	/*********************************close file*************************************/
	stream_reader.close();
	printf("load %d imgData...", gAdsInfo.size() );

	/*********************************MutiThread*************************************/	
	{
		pthread_t *pThread = new pthread_t[THREADCOUNT];
		ThreadParam *pParam = new ThreadParam[THREADCOUNT];
		
		for(i=0; i<THREADCOUNT; ++i)
		{
			pParam[i].para = int(i);
			pParam[i].nThreads = THREADCOUNT;
			pParam[i].svPath = savePath;
			pParam[i].BinReSizeImg = BinReSizeImg;

			int rc = pthread_create(pThread+i, NULL, workThread_Ads,(void*)(pParam+i));
		}

		for(i=0; i<THREADCOUNT; ++i)
		{
			pthread_join(pThread[i], NULL);
		}
		sleep (1);
		delete [] pThread;
		delete [] pParam;
	}

	return 0;
} 


int DL_DownloadImg_IdUrlScore( char *szQueryList, char *label, char *savePath, long BinReSizeImg )
{
	/***********************************Init*************************************/
	long i;
	unsigned long long ImageID = 0;
	gAdsInfo.clear();
	
	API_COMMEN api_commen;
	vector<string> cell ;
	
	/********************************Open Query List*****************************/
	std::ifstream stream_reader( szQueryList );
	std::string line;
	
	/*****************************Process one by one*****************************/
	while( getline(stream_reader, line) )
	{
		if( strlen(line.c_str()) == 0 )
		{
		  	continue;
		}
	  	//cout<<line<<endl;
		
	    cell.clear() ;
		api_commen.split( line, ",", cell );
		//for(i=0;i<cell.size();i++)
		//	printf("cell.size:%d,cell[%d]:%s\n", cell.size(), i, cell[i].c_str() );
		
		if ( cell.size() == 3 )
		{
			/************************push adsInfo*****************************/
			AdsInfo adsInfo;
			adsInfo.ImageID = atol(cell[0].c_str());
			adsInfo.url = cell[1];
			adsInfo.label = label;
			adsInfo.score = float(atof(cell[2].c_str()));

			/************************push adsInfo*****************************/
			if ( 	( adsInfo.ImageID < 0 ) || ( adsInfo.score < 0 ) || 
					( strlen(adsInfo.url.c_str()) == 0 ) || ( strlen(adsInfo.label.c_str()) == 0 ) )
				continue;
			gAdsInfo.push_back( adsInfo );
		}
		else if ( cell.size() == 4 )
		{
			/************************push adsInfo*****************************/
			AdsInfo adsInfo;
			adsInfo.ImageID = atol(cell[0].c_str());
			adsInfo.url = cell[2];
			adsInfo.label = cell[3];
			adsInfo.score = float(atof(cell[1].c_str()));

			/************************push adsInfo*****************************/
			if ( 	( adsInfo.ImageID < 0 ) || ( adsInfo.score < 0 ) || 
					( strlen(adsInfo.url.c_str()) == 0 ) || ( strlen(adsInfo.label.c_str()) == 0 ) )
				continue;
			gAdsInfo.push_back( adsInfo );
		}
		else
			continue;		
	}
	/*********************************close file*************************************/
	stream_reader.close();
	printf("load %d imgData...", gAdsInfo.size() );

	/*********************************MutiThread*************************************/	
	{
		pthread_t *pThread = new pthread_t[THREADCOUNT];
		ThreadParam *pParam = new ThreadParam[THREADCOUNT];
		
		for(i=0; i<THREADCOUNT; ++i)
		{
			pParam[i].para = int(i);
			pParam[i].nThreads = THREADCOUNT;
			pParam[i].svPath = savePath;
			pParam[i].BinReSizeImg = BinReSizeImg;

			int rc = pthread_create(pThread+i, NULL, workThread_Ads,(void*)(pParam+i));
		}

		for(i=0; i<THREADCOUNT; ++i)
		{
			pthread_join(pThread[i], NULL);
		}
		sleep (1);
		delete [] pThread;
		delete [] pParam;
	}

	return 0;
} 

/***********************************GetUrlFromFilePath***********************************/
//input:406388023,"/in/2015/11/02/188A5B83-A211-E6B4-6179-6C9F1BE6A295-1JxRGyg.jpg"
string GetUrlFromFilePath(const char *filepath)
{
	long ID = 0;
	int  atom =0;
	string headUrl = "http://inimg01.jiuyan.info/";
	string tmpPath = filepath;
	string Url = headUrl;
	long start = tmpPath.find_first_of('/');
	long end = tmpPath.find_last_of('g');

	if ( (start>0) && (end>0) && ( end>start ) )
		Url += tmpPath.substr(start+1,end-start);
	//printf("input:%s\n",tmpPath.c_str());
	//printf("output:%s\n",Url.c_str());
	
	return Url;
}

int DL_DownloadImg_Ads_Shuying( char *szQueryList,char *savePath, long BinReSizeImg )
{
	/***********************************Init*************************************/
	long i;
	gImgInfo.clear();
	
	API_COMMEN api_commen;
	vector<string> cell ;
	
	/********************************Open Query List*****************************/
	std::ifstream stream_reader( szQueryList );
	std::string line;
	
	/*****************************Process one by one*****************************/
	while( getline(stream_reader, line) )
	{
		if( strlen(line.c_str()) == 0 )
		{
		  	continue;
		}
	  	//cout<<line<<endl;
		
	    cell.clear() ;
		api_commen.split( line, ",", cell );
		//for(i=0;i<cell.size();i++)
		//	printf("cell.size:%d,cell[%d]:%s\n", cell.size(), i, cell[i].c_str() );

		if ( cell.size() == 2 )
		{
			string url = GetUrlFromFilePath( cell[1].c_str() );
			gImgInfo.push_back( std::make_pair( atol(cell[0].c_str()), url ) );
		}
		else if ( cell.size() == 3 )
		{
			string url = GetUrlFromFilePath( cell[2].c_str() );
			gImgInfo.push_back( std::make_pair( atol(cell[0].c_str()), url ) );
		}
		else
			continue;		
	}
	/*********************************close file*************************************/
	stream_reader.close();
	printf("load %d imgData...", gImgInfo.size() );

	/*********************************MutiThread*************************************/	
	{
		pthread_t *pThread = new pthread_t[THREADCOUNT];
		ThreadParam *pParam = new ThreadParam[THREADCOUNT];
		
		for(i=0; i<THREADCOUNT; ++i)
		{
			pParam[i].para = int(i);
			pParam[i].nThreads = THREADCOUNT;
			pParam[i].svPath = savePath;
			pParam[i].BinReSizeImg = BinReSizeImg;

			int rc = pthread_create(pThread+i, NULL, workThread_download,(void*)(pParam+i));
		}

		for(i=0; i<THREADCOUNT; ++i)
		{
			pthread_join(pThread[i], NULL);
		}
		sleep (1);
		delete [] pThread;
		delete [] pParam;
	}

	return 0;
}

int main(int argc, char* argv[])
{
	int  ret = 0;
	char szKeyFiles[256],szSavePath[256];
	API_COMMEN api_commen;

	if (argc == 6 && strcmp(argv[1],"-downloadpath") == 0) {
		ret = DL_DownloadImg_Path( argv[2], argv[3], atol(argv[4]), atol(argv[5]) );
	}
	if (argc == 6 && strcmp(argv[1],"-downloadpath_stringid") == 0) {
		ret = DL_DownloadImg_Path_stringid( argv[2], argv[3], atol(argv[4]), atol(argv[5]) );
	}
	else if (argc == 5 && strcmp(argv[1],"-downloadnopath") == 0) {
		ret = DL_DownloadImg_NoPath( argv[2], argv[3], atol(argv[4]) );
	}
	else if (argc == 6 && strcmp(argv[1],"-downloadnum") == 0) {
		ret = DL_DownloadImg_Num( atol(argv[2]), atol(argv[3]), argv[4], atol(argv[5]) );
	}
	else if (argc == 5 && strcmp(argv[1],"-downloadurl") == 0) {
		ret = DL_DownloadImg_OnlyUrl( argv[2], argv[3], atol(argv[4]) );
	} 
	else if (argc == 6 && strcmp(argv[1],"-downloadidurlscore") == 0) {
		ret = DL_DownloadImg_IdUrlScore( argv[2], argv[3], argv[4], atol(argv[5]) );
	} 
	else if (argc == 5 && strcmp(argv[1],"-ads_shuying") == 0) {
		ret = DL_DownloadImg_Ads_Shuying( argv[2], argv[3], atol(argv[4]) );
	}
	else
	{
		cout << "usage:\n" << endl;
		cout << "\tDemo_downloadimg -downloadpath queryList.csv savePath BinReSizeImg BinOnlyUrl\n" << endl;	
		cout << "\tDemo_downloadimg -downloadpath_stringid queryList.csv savePath BinReSizeImg BinOnlyUrl\n" << endl;	
		cout << "\tDemo_downloadimg -downloadnopath queryList.csv savePath BinReSizeImg\n" << endl;
		cout << "\tDemo_downloadimg -downloadnum startID ImgNum savePath BinReSizeImg\n" << endl;	
		cout << "\tDemo_downloadimg -downloadurl queryList.csv savePath BinReSizeImg\n" << endl;
		cout << "\tDemo_downloadimg -downloadidurlscore queryList.csv label savePath BinReSizeImg\n" << endl;
		cout << "\tDemo_downloadimg -ads_shuying queryList.csv savePath BinReSizeImg\n" << endl;		
		return ret;
	}
	return ret;
}
