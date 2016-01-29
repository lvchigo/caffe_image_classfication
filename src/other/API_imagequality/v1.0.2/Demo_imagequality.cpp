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

#include "API_commen.h"
#include "API_imagequality.h"
#include "TErrorCode.h"

using namespace cv;
using namespace std;

int ImageQuality_ExtractFeat( char *szQueryList, char *szFeatResult, char* KeyFilePath )
{
	/*****************************Init*****************************/
	char loadImgPath[256];
	char szImgPath[256];
	int i, j, label, svImg, nRet = 0;
	long inputLabel, nCount;
	unsigned long long ImageID = 0;
	double allGetLabelTime,tGetLabelTime;
	FILE *fpListFile = 0 , *fpFeatOut = 0;
	
	vector< vector< float > > 		imgFeat;

	API_COMMEN api_commen;
	API_IMAGEQUALITY api_imagequality;

	/***********************************Init*************************************/
	nRet = api_imagequality.Init( KeyFilePath ); 
	if (nRet != 0)
	{
	   cout<<"Fail to initialization "<<endl;
	   return TEC_INVALID_PARAM;
	}
	
	/********************************Open Query List*****************************/
	fpListFile = fopen(szQueryList,"r");
	if (!fpListFile) 
	{
		cout << "0.can't open " << szQueryList << endl;
		return TEC_INVALID_PARAM;
	}

	fpFeatOut = fopen(szFeatResult, "wt");
	if (!fpFeatOut)
	{
		cout << "Can't open result file " << szFeatResult << endl;
		return TEC_INVALID_PARAM;
	}

	nCount = 0;
	tGetLabelTime = 0.0;
	allGetLabelTime = 0.0;
	/*****************************Process one by one*****************************/
	while(EOF != fscanf(fpListFile, "%s %d", loadImgPath, &label))
	{
		IplImage *img = cvLoadImage(loadImgPath);
		if(!img || (img->width<16) || (img->height<16) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
		{	
			cout<<"Can't open " << loadImgPath << endl;
			continue;
		}	
		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld img...\n",nCount);

		/************************getRandomID*****************************/
		api_commen.getRandomID( ImageID );
		
		/*****************************GetLabelFeat*****************************/	
		imgFeat.clear();	
		tGetLabelTime = (double)getTickCount();
		nRet = api_imagequality.ExtractFeat( img, ImageID, imgFeat );	
		tGetLabelTime = (double)getTickCount() - tGetLabelTime;
		tGetLabelTime = tGetLabelTime*1000./cv::getTickFrequency();
		allGetLabelTime += tGetLabelTime;
		if ( (nRet != 0) || (imgFeat.size()<1) )
		{
		   cout<<"Fail to GetFeat!! "<<endl;
		   continue;
		}

		/************************Save GetFeat*****************************/
		for ( i=0;i<imgFeat.size();i++ )
		{
			fprintf(fpFeatOut, "%d ", label );
			for ( j=0;j<imgFeat[i].size();j++ )
			{
				fprintf(fpFeatOut, "%d:%.4f ", j+1, imgFeat[i][j] );
			}
			fprintf(fpFeatOut, "\n");
		}	

		/*********************************Release*************************************/
		cvReleaseImage(&img);img = 0;
	}

	/*********************************Release*************************************/
	api_imagequality.Release();

	/*********************************close file*************************************/
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	
	if (fpFeatOut) {fclose(fpFeatOut);fpFeatOut = 0;}

	/*********************************Print Info*********************************/
	if ( nCount != 0 ) 
	{
		printf( "nCount:%ld,GetLabelTime:%.4fms\n", nCount,allGetLabelTime*1.0/nCount );
	}
	
	cout<<"Done!! "<<endl;
	
	return nRet;
}

int SVM_Predict( char *szQueryList, char* KeyFilePath )
{
	char loadImgPath[256];
	char szImgPath[256];
	int i, j, label, svImg, nRet = 0;
	long inputLabel, nCount;
	unsigned long long ImageID = 0;
	double allPredictTime,tPredictTime;
	FILE *fpListFile = 0;

	API_COMMEN api_commen;
	API_IMAGEQUALITY api_imagequality;
	vector< pair< string, float > > Res;
	
	/***********************************Init*************************************/
	nRet = api_imagequality.Init( KeyFilePath ); 
	if (nRet != 0)
	{
	   cout<<"Fail to initialization "<<endl;
	   return TEC_INVALID_PARAM;
	}
	
	/********************************Open Query List*****************************/
	fpListFile = fopen(szQueryList,"r");
	if (!fpListFile) 
	{
		cout << "0.can't open " << szQueryList << endl;
		return TEC_INVALID_PARAM;
	}

	nCount = 0;
	tPredictTime = 0.0;
	allPredictTime = 0.0;
	/*****************************Process one by one*****************************/
	while(EOF != fscanf(fpListFile, "%s", loadImgPath))
	{
		IplImage *img = cvLoadImage(loadImgPath);
		if(!img || (img->width<16) || (img->height<16) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
		{	
			cout<<"Can't open " << loadImgPath << endl;
			continue;
		}	
		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld img...\n",nCount);

		/************************getRandomID*****************************/
		api_commen.getRandomID( ImageID );
		
		/************************SVM Predict*****************************/	
		Res.clear();
		tPredictTime = (double)getTickCount();
		nRet = api_imagequality.Predict( img, ImageID, Res );
		tPredictTime = (double)getTickCount() - tPredictTime;
		tPredictTime = tPredictTime*1000./cv::getTickFrequency();
		allPredictTime += tPredictTime;
		if (nRet != 0)
		{	
		   	cout<<"Fail to GetSVMPredict!! "<<endl;
		   	continue;
		}

		/*****************************save img data*****************************/
		for( i=0;i<Res.size();i++ )
		{
			if( nCount%50 == 0 )
				printf("imgLabel[%d]:predict_label-%s,score-%.4f\n",i,Res[i].first.c_str(),Res[i].second );
			
			sprintf(szImgPath, "res/%s/%s_%.2f_%ld.jpg",
					Res[i].first.c_str(),Res[i].first.c_str(),Res[i].second,ImageID);
			cvSaveImage( szImgPath,img );
		}
		
		cvReleaseImage(&img);img = 0;
	}

	/*********************************Release*************************************/
	api_imagequality.Release();

	/*********************************close file*************************************/
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	

	/*********************************Print Info*********************************/
	if ( nCount != 0 ) 
	{
		printf( "nCount:%ld,PredictTime:%.4fms\n", nCount, allPredictTime*1.0/nCount );
	}
	
	cout<<"Done!! "<<endl;
	
	return nRet;
}	


int main(int argc, char* argv[])
{
	int  ret = 0;
	char szKeyFiles[256],szSavePath[256];
	API_COMMEN api_commen;

	if (argc == 5 && strcmp(argv[1],"-extract") == 0) {
		strcpy(szKeyFiles, argv[4]);
		api_commen.PadEnd(szKeyFiles);
		ret = ImageQuality_ExtractFeat( argv[2], argv[3], szKeyFiles );
	}
	else if (argc == 4 && strcmp(argv[1],"-predict") == 0) {
		strcpy(szKeyFiles, argv[3]);
		api_commen.PadEnd(szKeyFiles);
		ret = SVM_Predict( argv[2], szKeyFiles );
	}
	else
	{
		cout << "usage:\n" << endl;
		cout << "\tDemo_imagequality -extract queryList.txt szFeat keyFilePath\n" << endl;
		cout << "\tDemo_imagequality -predict queryList.txt keyFilePath\n" << endl;
		return ret;
	}
	return ret;
}
