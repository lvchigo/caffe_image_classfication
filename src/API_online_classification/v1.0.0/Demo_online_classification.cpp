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
#include "SD_global.h"
#include "API_online_classification.h"
#include "TErrorCode.h"

using namespace cv;
using namespace std;

int SimilarDetect(char *szFileList, char *szKeyFiles, char *szSavePath, char *szSaveName, int ClassID, int subClassID, int BinSaveFeat)
{
	/*****************************Init*****************************/
	IplImage *img = 0;
	int  ret = 0;
	unsigned long long ImageID = 0;
	char szImgPath[512];
	char szSaveImgPath[512];
	FILE *fpListFile = 0 ;
	std::vector<int> resultVect;
	SD_RES result;
	SD_GLOBAL_1_1_0::ClassInfo ci;
	vector < SD_GLOBAL_1_1_0::ClassInfo > classList;

	API_COMMEN api_commen;

	/********************************Open Query List*****************************/
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
	ci.ClassID = ClassID;
	ci.SubClassID = subClassID;
	classList.push_back(ci);
	ret = SD_GLOBAL_1_1_0::LoadClassData(classList);
	if (ret != 0)
	{
	   cout<<"Fail to LoadClassData. Error code:"<< ret << endl;
	   goto ERR;
	}
	while( EOF != fscanf(fpListFile, "%s", szImgPath))
	{
		img = cvLoadImage(szImgPath, 1);					//待提取特征图像文件
		if(!img) 
		{	
			cout<<"can't open " << szImgPath << ",or unsupported image format!! "<< endl;
			continue;
		}	
		
		/************************getRandomID*****************************/
		api_commen.getRandomID( ImageID );

		/************************SimilarDetect*****************************/
		ret = SD_GLOBAL_1_1_0::SimilarDetect(img, ImageID, ClassID, subClassID,&result, BinSaveFeat);
		if (TOK == ret)
		{
			printf("%lld\t%lld\t%d\n",ImageID,result.id,result.sMode);
			if (result.sMode == 0)//no detect similar image
			{
				sprintf (szSaveImgPath, "%s/%s_%lld.jpg",szSavePath,szSaveName,ImageID);
				IplImage *ImageMedia = cvCreateImage(cvSize(256, 256), img->depth, img->nChannels);
				cvResize(img, ImageMedia);
				cvSaveImage(szSaveImgPath,ImageMedia);
				cvReleaseImage(&ImageMedia);
			}
		}
		else if (ret == TEC_UNSUPPORTED)
		{
			cout<< szImgPath << " is unsupported image format!! "<< endl;
			continue;
		}
		cvReleaseImage(&img);
		img = 0;
	}

ERR:
	//释放资源
	SD_GLOBAL_1_1_0::Uninit();

	if (img)	
		cvReleaseImage(&img);

	if (fpListFile) {
		fclose(fpListFile);
		fpListFile = 0;
	}

	printf("SimilarDetect Done!\n");
	return ret;
}

int ImageQuality_RemoveWhitePart( char *szQueryList )
{
	/*****************************Init*****************************/
	char loadImgPath[256];
	char szImgPath[256];
	int i, j, label, svImg, tank, nRet = 0;
	long inputLabel, nCount;
	unsigned long long ImageID = 0;
	FILE *fpListFile = 0 ;

	API_COMMEN api_commen;
	
	/********************************Open Query List*****************************/
	fpListFile = fopen(szQueryList,"r");
	if (!fpListFile) 
	{
		cout << "0.can't open " << szQueryList << endl;
		return TEC_INVALID_PARAM;
	}

	nCount = 0;
	/*****************************Process one by one*****************************/
	while(EOF != fscanf(fpListFile, "%s %d", loadImgPath,&label ))
	{
		IplImage *img = cvLoadImage(loadImgPath);
		if(!img || (img->width<16) || (img->height<16) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
		{	
			cout<<"Can't open " << loadImgPath << endl;

			cvReleaseImage(&img);img = 0;
			continue;
		}	
		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld img...\n",nCount);

		/************************getRandomID*****************************/
		api_commen.getRandomID( ImageID );

		/*****************************Remove White Part*****************************/
		IplImage* ImgRemoveWhite = api_commen.RemoveWhitePart( img, ImageID );
		if( !ImgRemoveWhite || (ImgRemoveWhite->width<16) || (ImgRemoveWhite->height<16) ) 
		{	
			cout<<"ImgRemoveWhite err: " << loadImgPath << endl;

			cvReleaseImage(&img);img = 0;
			cvReleaseImage(&ImgRemoveWhite);ImgRemoveWhite = 0;
			continue;
		}	

		cvSaveImage( loadImgPath,ImgRemoveWhite );
		/*********************************Release*************************************/
		cvReleaseImage(&img);img = 0;
		cvReleaseImage(&ImgRemoveWhite);ImgRemoveWhite = 0;
	}

	/*********************************close file*************************************/
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	
	
	cout<<"Done!! "<<endl;
	
	return nRet;
}

int ImageQuality_RemoveSmallImg( char *szQueryList, char *svPath, int size )
{
	/*****************************Init*****************************/
	char loadImgPath[256];
	char szImgPath[256];
	int i, j, svImg, tank, nRet = 0;
	long inputLabel, nCount;
	unsigned long long ImageID = 0;
	FILE *fpListFile = 0 ;

	API_COMMEN api_commen;
	
	/********************************Open Query List*****************************/
	fpListFile = fopen(szQueryList,"r");
	if (!fpListFile) 
	{
		cout << "0.can't open " << szQueryList << endl;
		return TEC_INVALID_PARAM;
	}

	nCount = 0;
	/*****************************Process one by one*****************************/
	while(EOF != fscanf(fpListFile, "%s", loadImgPath ))
	{
		IplImage *img = cvLoadImage(loadImgPath);
		if(!img || (img->width<16) || (img->height<16) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
		{	
			cout<<"Can't open " << loadImgPath << endl;

			cvReleaseImage(&img);img = 0;
			continue;
		}	
		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld img...\n",nCount);

		/************************getRandomID*****************************/
		api_commen.getRandomID( ImageID );

		if( (img->width>size) || (img->height>size) ) 
		{	
			sprintf(szImgPath, "%s/%ld.jpg", svPath, ImageID);
			cvSaveImage( szImgPath,img );
		}	
		
		/*********************************Release*************************************/
		cvReleaseImage(&img);img = 0;
	}

	/*********************************close file*************************************/
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	
	
	cout<<"Done!! "<<endl;
	
	return nRet;
}

int ImageQuality_CreatSample( char *szQueryList )
{
	/*****************************Init*****************************/
	char loadImgPath[256];
	char szImgPath[256];
	int i, j, label, svImg, tank, nRet = 0;
	long inputLabel, nCount;
	unsigned long long ImageID = 0;
	FILE *fpListFile = 0 ;

	API_COMMEN api_commen;
	
	/********************************Open Query List*****************************/
	fpListFile = fopen(szQueryList,"r");
	if (!fpListFile) 
	{
		cout << "0.can't open " << szQueryList << endl;
		return TEC_INVALID_PARAM;
	}

	nCount = 0;
	/*****************************Process one by one*****************************/
	while(EOF != fscanf(fpListFile, "%s", loadImgPath ))
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
		tank = ( ImageID % 19997 ) % 5;		//balance

		/*****************************Remove White Part*****************************/
		IplImage* ImgRemoveWhite = api_commen.RemoveWhitePart( img, ImageID );

		/*****************************Creat Sample*****************************/
		switch(tank)
		{
			case 1: 
				/*****************************CreatSample_LowContrast*****************************/	
			    nRet = api_commen.CreatSample_LowContrast( ImgRemoveWhite, ImageID, 3.0 );	
				if (nRet != 0)
				{
				   cout<<"Fail to CreatSample_LowContrast!! "<<endl;
				   continue;
				}
			    break; 

		    case 2: 
		    	/*****************************CreatSample_LowContrast*****************************/	
				nRet = api_commen.CreatSample_LowContrast( ImgRemoveWhite, ImageID, 1.0/3.0 );	
				if (nRet != 0)
				{
				   cout<<"Fail to CreatSample_LowContrast!! "<<endl;
				   continue;
				}
				break;

		    case 3: 
		    	/*****************************CreatSample_LowResolution*****************************/	
				nRet = api_commen.CreatSample_LowResolution( ImgRemoveWhite, ImageID );	
				if (nRet != 0)
				{
				   cout<<"Fail to CreatSample_LowResolution!! "<<endl;
				   continue;
				}
			    break; 

			case 4:
				/*****************************CreatSample_smooth*****************************/	
				nRet = api_commen.CreatSample_smooth( ImgRemoveWhite, ImageID );	
				if (nRet != 0)
				{
				   cout<<"Fail to CreatSample_LowResolution!! "<<endl;
				   continue;
				}
				break; 

		    default: 
			    /*****************************CreatSample_addGaussNoise*****************************/	
				nRet = api_commen.CreatSample_addGaussNoise( ImgRemoveWhite, ImageID );	
				if (nRet != 0)
				{
				   cout<<"Fail to CreatSample_LowResolution!! "<<endl;
				   continue;
				}
			    break; 
		}

		/*********************************Release*************************************/
		cvReleaseImage(&img);img = 0;
		cvReleaseImage(&ImgRemoveWhite);ImgRemoveWhite = 0;
	}

	/*********************************close file*************************************/
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	
	
	cout<<"Done!! "<<endl;
	
	return nRet;
}

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
	API_ONLINE_CLASSIFICATION api_online_classification;
	vector< pair< string, float > > Res;

	//check
	string text;
	map<string, long> map_Num_Res_Label;
	map<string, long>::iterator it_Num_Res_Label;
	
	/***********************************Init*************************************/
	nRet = api_online_classification.Init( KeyFilePath, layerName, binGPU, deviceID ); 
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
		api_commen.getRandomID( ImageID );
		//ImageID = api_commen.GetIDFromFilePath( loadImgPath );

		/************************SVM Predict*****************************/	
		Res.clear();
		tPredictTime = (double)getTickCount();
		//[Out]:Res:In37Class/ads6Class/imgquality3class
		nRet = api_online_classification.Predict(img, ImageID, layerName, Res);
		tPredictTime = (double)getTickCount() - tPredictTime;
		tPredictTime = tPredictTime*1000./cv::getTickFrequency();
		allPredictTime += tPredictTime;
		if (nRet != 0)
		{	
		   	cout<<"Fail to GetSVMPredict!! "<<endl;
			cvReleaseImage(&img);img = 0;
		   	continue;
		}

		/*****************************save img data*****************************/
		for( i=0;i<Res.size();i++ )
		{
			if( nCount%50 == 0 )
				printf("imgLabel[%d]:predict_label-%s,score-%.4f\n",i,Res[i].first.c_str(),Res[i].second );
			
			IplImage *ImgResize = cvCreateImage(cvSize(255, 255), img->depth, img->nChannels);
			cvResize( img, ImgResize );
			if ( ( i == 0 ) || ( i == 1 ) )
			{
				sprintf(szImgPath, "res/%s/%s_%.2f_%ld.jpg",
					Res[i].first.c_str(),Res[i].first.c_str(),Res[i].second,ImageID);
				cvSaveImage( szImgPath,ImgResize );
			}
			else if ( i == 2 )	//imagequality
			{
				sprintf(szImgPath, "res/%s/%s_%s_%.2f_%ld.jpg",
					Res[i].first.c_str(),Res[i].first.c_str(),Res[0].first.c_str(),Res[i].second,ImageID);
				cvSaveImage( szImgPath,img );
			}
			cvReleaseImage(&ImgResize);ImgResize = 0;
		}

		//check data 
		for(i=0;i<Res.size();i++)  
		{	
			text = Res[i].first;
			it_Num_Res_Label = map_Num_Res_Label.find(text);
			if(it_Num_Res_Label == map_Num_Res_Label.end())
			{
			    map_Num_Res_Label[text] = 1;
			}
			else
			{	
				map_Num_Res_Label[text] = it_Num_Res_Label->second+1;
			}
		}
		
		cvReleaseImage(&img);img = 0;
	}

	/*********************************Release*************************************/
	api_online_classification.Release();

	/*********************************close file*************************************/
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	

	/*********************************Print Info*********************************/
	if ( nCount != 0 ) 
	{
		printf( "nCount:%ld,PredictTime:%.4fms\n", nCount, allPredictTime*1.0/nCount );

		//print check data
		long Num_Check[3] = {0};
		printf("Label_Num:%d\n",map_Num_Res_Label.size());
		for(it_Num_Res_Label = map_Num_Res_Label.begin(); it_Num_Res_Label != map_Num_Res_Label.end(); it_Num_Res_Label++)
		{
			text = it_Num_Res_Label->first.c_str();
			printf("%s_%ld\n", text.c_str(), it_Num_Res_Label->second );

			if (text == "other.other.other")
				Num_Check[0] += it_Num_Res_Label->second;
			else if ( (text == "food.food.food") || (text == "goods.goods.goods") ||
				 (text == "people.people.people") ||(text == "pet.pet.pet") || 
				 (text == "scene.scene.scene") )
				Num_Check[1] += it_Num_Res_Label->second;
			else
				Num_Check[2] += it_Num_Res_Label->second;
		}
		printf("\n");
		
		long allLabel = Num_Check[0]+Num_Check[1]+Num_Check[2];
		printf("AllLabel:%ld,2nd-class-label:%ld_%.2f,1rd-class-label:%ld_%.2f,other:%ld_%.2f\n",
			allLabel, 	Num_Check[2], Num_Check[2]*100.0/(Num_Check[1]+Num_Check[2]), 
						Num_Check[1], Num_Check[1]*100.0/(Num_Check[1]+Num_Check[2]), 
						Num_Check[0], Num_Check[0]*100.0/allLabel );
	}
	
	cout<<"Done!! "<<endl;
	
	return nRet;
}	

int main(int argc, char* argv[])
{
	int  ret = 0;
	char szKeyFiles[256],szSavePath[256];
	API_COMMEN api_commen;

	if (argc == 9 && strcmp(argv[1],"-simdetect") == 0)
	{
		strcpy(szKeyFiles, argv[3]);
		api_commen.PadEnd(szKeyFiles);
		strcpy(szSavePath, argv[4]);
		api_commen.PadEnd(szSavePath);
		SimilarDetect(argv[2], szKeyFiles, szSavePath, argv[5], atol(argv[6]), atol(argv[7]), atol(argv[8]) );
	}
	else if (argc == 3 && strcmp(argv[1],"-removewhitepart") == 0) {
		ret = ImageQuality_RemoveWhitePart( argv[2] );
	}
	else if (argc == 5 && strcmp(argv[1],"-removesmallimg") == 0) {
		ret = ImageQuality_RemoveSmallImg( argv[2], argv[3], atol(argv[4]) );
	}
	else if (argc == 3 && strcmp(argv[1],"-creatsample") == 0) {
		ret = ImageQuality_CreatSample( argv[2] );
	}
	else if (argc == 9 && strcmp(argv[1],"-extract") == 0) {
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
		cout << "\tDemo_online_classification -simdetect ImageList.txt keyFilePath savePath saveName ClassID subCLassID BinSaveFeat\n" << endl;
		cout << "\tDemo_online_classification -removewhitepart queryList.txt\n" << endl;
		cout << "\tDemo_online_classification -removesmallimg queryList.txt svPath size\n" << endl;
		cout << "\tDemo_online_classification -creatsample queryList.txt\n" << endl;
		cout << "\tDemo_online_classification -extract queryList.txt szFeat keyFilePath layerName binGPU deviceID svMode\n" << endl;
		cout << "\tDemo_online_classification -predict queryList.txt keyFilePath layerName binGPU deviceID\n" << endl;
		return ret;
	}
	return ret;
}
