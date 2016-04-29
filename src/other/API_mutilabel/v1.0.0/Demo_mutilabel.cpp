#include <string>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h> 
#include <sys/types.h> 
#include <iostream>
#include <unistd.h>

#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>

#include "API_commen.h"
#include "API_caffe.h"
#include "similardetect/SD_global.h"
#include "API_mutilabel.h"
#include "TErrorCode.h"

using namespace cv;
using namespace std;

int DL_ExtractFeat( char *szQueryList, char *szFeatResult, char *szKeyFiles, char *layerName, int binGPU, int deviceID, int svMode )
{
	/*****************************Init*****************************/
	char loadImgPath[256];
	char szImgPath[256];
	int i, j, label, svImg, nRet = 0;
	long inputLabel, nCount;
	unsigned long long ImageID = 0;
	double allGetLabelTime,tGetLabelTime;
	FILE *fpListFile = 0 , *fpFeatOut = 0;
	
	vector< vector<float> > imgFeat;
	vector< vector<float> > vecNormDLFeat;
	vector< pair< int, float > > imgLabel;
	vector<float> normDLFeat;
	vector<float> normImageQualityBlurFeat;

	API_COMMEN api_commen;
	API_CAFFE api_caffe;
	
	/***********************************Init*************************************/
	char DL_DeployFile[1024] = {0};
	char DL_ModelFile[1024] = {0};
	char DL_Meanfile[1024] = {0};

	sprintf(DL_DeployFile, "%s/vgg_16/deploy_vgg_16.prototxt",szKeyFiles);
	sprintf(DL_ModelFile, "%s/vgg_16/VGG_ILSVRC_16_layers.caffemodel",szKeyFiles);
	sprintf(DL_Meanfile, "%s/vgg_16/imagenet_mean.binaryproto",szKeyFiles);	//vgg:add 2dcode
	nRet = api_caffe.Init( DL_DeployFile, DL_ModelFile, DL_Meanfile, layerName, binGPU, deviceID ); 
	if (nRet != 0)
	{
	   cout<<"Fail to initialization "<<endl;
	   return TEC_INVALID_PARAM;
	}

	//check mode
	if ( svMode == 1 )  	//1-in73class
		printf( "svMode:%d,Extract in73class Feat!!\n", svMode );
	else if ( svMode == 2 )	//2-in6class;
		printf( "svMode:%d,Extract in6class Feat!!\n", svMode );
	else if ( svMode == 3 )	//3-ads6class;
		printf( "svMode:%d,Extract ads6class Feat!!\n", svMode );
	else if ( svMode == 4 )	//4-imagequality;
		printf( "svMode:%d,Extract imagequality Feat!!\n", svMode );
	else if ( svMode == 5 )	//5-imagequality blur;
		printf( "svMode:%d,Extract imagequality blur Feat!!\n", svMode );
	else
	{
		printf( "svMode:%d,err!!\n", svMode );
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
	while(EOF != fscanf(fpListFile, "%s %d", &loadImgPath, &label))
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

		/*****************************GetMutiImg*****************************/
		vector < Mat_ < Vec3f > > img_dl;
		nRet = api_commen.Img_GetMutiRoi( img, ImageID, img_dl );
		if ( ( nRet != 0) || ( img_dl.size() < 1 ) )
		{
			cout<<"Fail to Img_GetMutiRoi!! "<<endl; 
			continue;
		}	

/*		//save
		for(i=0;i<img_dl.size();i++)
		{
			sprintf(szImgPath, "res/Img_GetMutiRoi_%ld_%d.jpg",ImageID,i);
			imwrite( szImgPath, img_dl[i] );
		}*/
		
		/*****************************GetLabelFeat*****************************/	
		imgLabel.clear();
		imgFeat.clear();	
		int bExtractFeat = 2;		//[In]:Get Label(1),Extract Feat(2),both(3)
		tGetLabelTime = (double)getTickCount();
		nRet = api_caffe.GetLabelFeat( img_dl, ImageID, layerName, bExtractFeat, imgLabel, imgFeat);	
		tGetLabelTime = (double)getTickCount() - tGetLabelTime;
		tGetLabelTime = tGetLabelTime*1000./cv::getTickFrequency();
		allGetLabelTime += tGetLabelTime;
		if ( (nRet != 0) || (imgFeat.size()<1) )
		{
		   cout<<"Fail to GetFeat!! "<<endl;
		   continue;
		}

		/************************Normal Feat*****************************/
		vecNormDLFeat.clear();
		normImageQualityBlurFeat.clear();
		for(i=0;i<imgFeat.size();i++)	//imgFeat.size()=2:1-ads/in36class,2-imagequality;
		{
			normDLFeat.clear();

			/************************DL Feat Normal*****************************/
			if ( ( svMode == 1 ) && ( i == 0 ) )  	//1-in73class
				nRet = api_commen.Normal_L2(imgFeat[i],normDLFeat);
			else if ( ( svMode == 2 ) && ( i == 0 ) ) 	//2-in6class;
				nRet = api_commen.Normal_MinMax(imgFeat[i],normDLFeat);
			else if ( ( svMode == 3 ) && ( i == 0 ) ) 	//3-ads6class;
				nRet = api_commen.Normal_L2(imgFeat[i],normDLFeat);
			else if ( ( svMode == 4 ) && ( i == 1 ) ) 	//4-imagequality;
				nRet = api_commen.Normal_MinMax(imgFeat[i],normDLFeat);
			else if ( ( svMode == 5 ) && ( i > 1 ) ) 	//5-imagequality blur;
			{
				//Merge to one vector
				for( j=0;j<imgFeat[i].size();j++ )
				{
					normImageQualityBlurFeat.push_back(imgFeat[i][j]);	// FOR ImageQualityBlur
				}
			}
			if (nRet != 0)
			{
			   cout<<"Fail to Normal!! "<<endl;
			   continue;
			}
			
			if ( ( normDLFeat.size()>0 ) && (  svMode != 5 ) )
				vecNormDLFeat.push_back(normDLFeat);	//NormDLFeat
		}

		//printf("\n\n");
		
		if ( ( normImageQualityBlurFeat.size()>0 ) && (  svMode == 5 ) )
		{
			/************************DL Feat Normal*****************************/
			normDLFeat.clear();
			nRet = api_commen.Normal_L2(normImageQualityBlurFeat,normDLFeat);		
			if (nRet != 0)
			{
			   cout<<"Fail to Normal_L2!! "<<endl;
			   return nRet;
			}

			vecNormDLFeat.clear();
			vecNormDLFeat.push_back( normDLFeat );
		}

		/************************Save GetFeat*****************************/
		for ( i=0;i<vecNormDLFeat.size();i++ )
		{
			fprintf(fpFeatOut, "%d ", label );
			for ( j=0;j<vecNormDLFeat[i].size();j++ )
			{
				fprintf(fpFeatOut, "%d:%.6f ", j+1, (vecNormDLFeat[i][j]+0.00000001) );
			}
			fprintf(fpFeatOut, "\n");
		}	

		/*********************************Release*************************************/
		cvReleaseImage(&img);img = 0;
	}

	/*********************************Release*************************************/
	api_caffe.Release();

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

int SVM_Predict( char *szQueryList, char* KeyFilePath, char *layerName, int binGPU, int deviceID )
{
	char loadImgPath[256];
	char szImgPath[256];
	int i, j, label, svImg, nRet = 0;
	long inputLabel, nCount;
	unsigned long long ImageID = 0;
	double allPredictTime,tPredictTime;
	FILE *fpListFile = 0;

	API_COMMEN api_commen;
	API_MUTILABEL api_mutilabel;
	vector< pair< string, float > > Res;
	
	/***********************************Init*************************************/
	nRet = api_mutilabel.Init( KeyFilePath, layerName, binGPU, deviceID ); 
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
	while(EOF != fscanf(fpListFile, "%s", &loadImgPath))
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
		//api_commen.getRandomID( ImageID );
		ImageID = api_commen.GetIDFromFilePath( loadImgPath );

		/************************SVM Predict*****************************/	
		Res.clear();
		tPredictTime = (double)getTickCount();
		//[Out]:Res:In37Class/ads6Class/imgquality3class
		nRet = api_mutilabel.Predict(img, ImageID, layerName, Res);
		tPredictTime = (double)getTickCount() - tPredictTime;
		tPredictTime = tPredictTime*1000./cv::getTickFrequency();
		allPredictTime += tPredictTime;
		if ( (nRet != 0) || ( Res.size() < 1 ) )
		{	
		   	cout<<"Fail to GetSVMPredict!! "<<endl;

			IplImage *ImgResize = cvCreateImage(cvSize(255, 255), img->depth, img->nChannels);
			cvResize( img, ImgResize );
			sprintf(szImgPath, "res/err/%ld.jpg",ImageID);
			cvSaveImage( szImgPath,ImgResize );
			cvReleaseImage(&ImgResize);ImgResize = 0;
			cvReleaseImage(&img);img = 0;

		   	continue;
		}

		/*****************************save img data*****************************/
		if( nCount%50 == 0 )
			printf("imgLabel[%d]:predict_label-%s,score-%.4f\n",i,Res[i].first.c_str(),Res[i].second );
		
		IplImage *ImgResize = cvCreateImage(cvSize(255, 255), img->depth, img->nChannels);
		cvResize( img, ImgResize );
		if ( Res.size() == 1 )
		{
			sprintf(szImgPath, "res/%s/%s_%.2f_%ld.jpg",
				Res[0].first.c_str(),Res[0].first.c_str(),Res[0].second,ImageID);
			cvSaveImage( szImgPath,ImgResize );
		}
		else if ( Res.size() == 2 )
		{
			sprintf(szImgPath, "res/%s/%s_%.2f_%s_%.2f_%ld.jpg",
				Res[0].first.c_str(),Res[0].first.c_str(),Res[0].second,Res[1].first.c_str(),Res[1].second,ImageID);
			cvSaveImage( szImgPath,ImgResize );
		}
		else if ( Res.size() == 3 )
		{
			sprintf(szImgPath, "res/%s/%s_%.2f_%s_%.2f_%s_%.2f_%ld.jpg",
				Res[0].first.c_str(),Res[0].first.c_str(),Res[0].second,Res[1].first.c_str(),Res[1].second,
				Res[2].first.c_str(),Res[2].second,ImageID);
			cvSaveImage( szImgPath,ImgResize );
		}
		cvReleaseImage(&ImgResize);ImgResize = 0;
		cvReleaseImage(&img);img = 0;
	}

	/*********************************Release*************************************/
	api_mutilabel.Release();

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

	if (argc == 9 && strcmp(argv[1],"-extract") == 0) {
		strcpy(szKeyFiles, argv[4]);
		api_commen.PadEnd(szKeyFiles);
		ret = DL_ExtractFeat( argv[2], argv[3], szKeyFiles, argv[5], atol(argv[6]), atol(argv[7]), atol(argv[8]) );
	}
	else if (argc == 7 && strcmp(argv[1],"-predict") == 0) {
		ret = SVM_Predict( argv[2], argv[3], argv[4], atol(argv[5]), atol(argv[6]) );
	}
	else
	{
		cout << "usage:\n" << endl;
		cout << "\tDemo_online_classification -extract queryList.txt szFeat keyFilePath layerName binGPU deviceID svMode\n" << endl;
		cout << "\tDemo_online_classification -predict queryList.txt keyFilePath layerName binGPU deviceID\n" << endl;
		return ret;
	}
	return ret;
}
