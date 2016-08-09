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
#include "API_pca.h"
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

int ImageQuality_PCA_LearnModel( char *szQueryList,char *outPCAModel, char *outPCAFeat, char *szKeyFiles )
{
	/*****************************Init*****************************/
	char loadImgPath[256];
	char szImgPath[256];
	int i, j, label, svImg, nRet = 0;
	long inputLabel, nCount;
	unsigned long long ImageID = 0;
	double allGetLabelTime,tGetLabelTime;
	FILE *fpListFile = 0 , *fpOut = 0;

	vector< vector<float> > imgFeat;
	vector< pair< int, vector<float> > > PCA_InFeat;

	API_COMMEN api_commen;
	API_IMAGEQUALITY api_imagequality;
	API_PCA api_pca;
	
	/***********************************Init*************************************/
	char DL_DeployFile[1024] = {0};
	char DL_ModelFile[1024] = {0};
	char DL_Meanfile[1024] = {0};
	sprintf(DL_DeployFile, "%s/vgg_16/deploy_vgg_16.prototxt",szKeyFiles);
	sprintf(DL_ModelFile, "%s/vgg_16/VGG_ILSVRC_16_layers.caffemodel",szKeyFiles);
	sprintf(DL_Meanfile, "%s/vgg_16/imagenet_mean.binaryproto",szKeyFiles); //vgg:add 2dcode

	nRet = api_imagequality.Init( szKeyFiles ); 
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
	
	fpOut = fopen(outPCAFeat, "wt");
	if (!fpOut)
	{
		cout << "Can't open result file " << outPCAFeat << endl;
		return TEC_INVALID_PARAM;
	}

	ImageID = 0;
	nCount = 0;
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
		   cout<<"Fail to GetLabelFeat!! "<<endl;
		   cvReleaseImage(&img);img = 0;
		   continue;
		}

		/************************PCA_InFeat*****************************/
		PCA_InFeat.push_back( std::make_pair( label, imgFeat[0] ) );
		
		/*********************************Release*************************************/
		cvReleaseImage(&img);img = 0;
	}

	/************************PCA*****************************/
	nRet = api_pca.PCA_Feat_Learn(PCA_InFeat, outPCAModel);
	if (nRet != 0)
	{
	   cout<<"Fail to PCA_Feat_Learn!! "<<endl;
	   return nRet;
	}

	/************************PCA*****************************/
	api_pca.PCA_Feat_Init( outPCAModel );
	
	for ( i=0;i<PCA_InFeat.size();i++ )
	{
		/************************PCA_InFeat*****************************/
		vector<float> EncodeFeat;
		nRet = api_pca.PCA_Feat_Encode( PCA_InFeat[i].second, EncodeFeat );
		if (nRet != 0)
		{
		   cout<<"Fail to PCA_Feat_Encode!! "<<endl;
		   continue;
		}

		vector<float> NormEncodeFeat;
		nRet = api_commen.Normal_L2( EncodeFeat, NormEncodeFeat );
		if (nRet != 0)
		{
		   cout<<"Fail to PCA_Feat_Normal!! "<<endl;
		   continue;
		}

		/************************Save GetFeat*****************************/
		fprintf(fpOut, "%d ", PCA_InFeat[i].first );
		for ( j=0;j<NormEncodeFeat.size();j++ )
		{
			fprintf(fpOut, "%d:%.4f ", j+1, NormEncodeFeat[j] );
		}
		fprintf(fpOut, "\n");

		/************************Check PCA Model*****************************/
		if (i<3)
		{
			vector<float> DecodeFeat;
			nRet = api_pca.PCA_Feat_Decode( EncodeFeat, PCA_InFeat[i].second.size(), DecodeFeat );
			if (nRet != 0)
			{
			   cout<<"Fail to PCA_Feat_Decode!! "<<endl;
			   return nRet;
			}
			printf("Check PCA Model:");
			for ( j=0;j<PCA_InFeat[i].second.size();j++ )
			{
				printf("%.6f-%.6f ", PCA_InFeat[i].second[j], DecodeFeat[j] );
			}
			printf("\n");
		}	
	}
	
	/*********************************Release*************************************/
	api_pca.PCA_Feat_Release();
	api_imagequality.Release();

	/*********************************close file*************************************/
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	
	if (fpOut) {fclose(fpOut);fpOut = 0;}

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

			//if ( ( Res[i].first == "quality.bad.bad" ) && ( Res[i].second >0.7 ) )
			{
				sprintf(szImgPath, "res/%s/%s_%.2f_%ld.jpg",
						Res[i].first.c_str(),Res[i].first.c_str(),Res[i].second,ImageID);
				cvSaveImage( szImgPath,img );
			}
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


int TestModel( char *szQueryList, char* KeyFilePath )
{
	char loadImgPath[256];
	char szImgPath[256];
	int i, j, label, svImg, nRet = 0;
	long inputLabel, nCount;
	unsigned long long ImageID = 0;
	double allPredictTime,tPredictTime;
	FILE *fpListFile = 0;
	int labelNum[8] = {0};	//0/1-sample pos/neg;2/3-predict pos/neg;4/5-predict>0.8 pos/neg;6/7-predict>0.8 pos/neg err;
	float score_T = 0.8;

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
	while(EOF != fscanf(fpListFile, "%s %d", loadImgPath, &label))
	{
		IplImage *img = cvLoadImage(loadImgPath);
		if(!img || (img->width<16) || (img->height<16) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
		{	
			cout<<"Can't open " << loadImgPath << endl;
			continue;
		}	

		/************************count label*****************************/
		//0/1-sample pos/neg;2/3-predict pos/neg;4/5-predict>0.8 pos/neg;6/7-predict>0.8 pos/neg err;
		if ( label == 1 )
			labelNum[0]++;
		else if ( label == 0 )
			labelNum[1]++;
		else
		{	
			printf( "label:%d err!!\n", label );
			continue;
		}	

		/************************nCount*****************************/
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
				printf("imgLabel[%d]:predict_label-%s,score-%.4f,label-%d\n",i,Res[i].first.c_str(),Res[i].second,label );
			
			sprintf(szImgPath, "res/%s/%s_%.2f_%d_%ld.jpg",
					Res[i].first.c_str(),Res[i].first.c_str(),Res[i].second,label,ImageID);
			cvSaveImage( szImgPath,img );
		}

		/************************count label*****************************/
		//0/1-sample pos/neg;2/3-predict pos/neg;4/5-predict>0.8 pos/neg;6/7-predict>0.8 pos/neg err;
		if ( ( Res[0].first == "quality.good.good" ) && ( label == 1 ) )
			labelNum[2]++;
		if ( ( Res[0].first == "quality.bad.bad" ) && ( label == 0 )  )
			labelNum[3]++;	
		if ( ( Res[0].first == "quality.good.good" ) && ( label == 1 )  && ( Res[0].second>=score_T ) )
			labelNum[4]++;
		if ( ( Res[0].first == "quality.bad.bad" ) && ( label == 0 )  && ( Res[0].second>=score_T ) )
			labelNum[5]++;	
		if ( ( Res[0].first == "quality.good.good" ) && ( label == 0 )  && ( Res[0].second>=score_T ) )
			labelNum[6]++;	//predict>score_T && "bad" predict to "good"
		if ( ( Res[0].first == "quality.bad.bad" ) && ( label == 1 )  && ( Res[0].second>=score_T ) )
			labelNum[7]++;	//predict>score_T && "good" predict to "bad"
		
		cvReleaseImage(&img);img = 0;
	}

	/*********************************Release*************************************/
	api_imagequality.Release();

	/*********************************close file*************************************/
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	

	/************************count label*****************************/
	//0/1-sample pos/neg;2/3-predict pos/neg;4/5-predict>0.8 pos/neg;6/7-predict>0.8 pos/neg err;
	if ( ( labelNum[0] != 0 ) && ( labelNum[1] != 0 ) )
	{
		printf( "[sample]:pos-%d,neg-%d\n", labelNum[0], labelNum[1] );
		printf( "[predict]:pos-%d,neg-%d;[err]:pos-%d,neg-%d;[recall]:pos-%.4f,neg-%.4f;[accuracy]:pos-%.4f,neg-%.4f;\n", 
			labelNum[2], labelNum[3], (labelNum[1]-labelNum[3]), (labelNum[0]-labelNum[2]),
			labelNum[2]*1.0/labelNum[0], labelNum[3]*1.0/labelNum[1],
			labelNum[2]*1.0/(labelNum[2]+labelNum[1]-labelNum[3]), 
			labelNum[3]*1.0/(labelNum[3]+labelNum[0]-labelNum[2]) );
		printf( "[predict>=%.4f]:pos-%d,neg-%d;[err]:pos-%d,neg-%d;[recall]:pos-%.4f,neg-%.4f;[accuracy]:pos-%.4f,neg-%.4f;\n", 
			score_T, labelNum[4], labelNum[5], labelNum[6], labelNum[7], 
			labelNum[4]*1.0/labelNum[0], labelNum[5]*1.0/labelNum[1],
			labelNum[4]*1.0/(labelNum[4]+labelNum[6]), 
			labelNum[5]*1.0/(labelNum[5]+labelNum[7]));
	}
	
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
	else if (argc == 4 && strcmp(argv[1],"-testmodel") == 0) {
		strcpy(szKeyFiles, argv[3]);
		api_commen.PadEnd(szKeyFiles);
		ret = TestModel( argv[2], szKeyFiles );
	}
	else
	{
		cout << "usage:\n" << endl;
		cout << "\tDemo_imagequality -extract queryList.txt szFeat keyFilePath\n" << endl;
		cout << "\tDemo_imagequality -predict queryList.txt keyFilePath\n" << endl;
		cout << "\tDemo_imagequality -testmodel queryList.txt keyFilePath\n" << endl;
		return ret;
	}
	return ret;
}
