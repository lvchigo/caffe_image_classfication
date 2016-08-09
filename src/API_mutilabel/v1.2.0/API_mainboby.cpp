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
#include "API_caffe.h"
#include "API_linearsvm.h"
#include "SD_global.h"
#include "API_mainboby.h"
#include "TErrorCode.h"
#include "plog/Log.h"

#include "ColorLayout.h"
#include "EdgeHist.h"

using namespace cv;
using namespace std;

//#define 	USE_CLD_FEAT 	1
#define 	USE_EHD_FEAT 	1
#define     CLD_DIM         72 
#define     EHD_DIM         80 

static bool vecEntropySortComp(const pair <int, float> elem1, const pair <int, float> elem2)
{
	return (elem1.second > elem2.second);
}

static bool PredictResSortComp(
	const pair< pair< string, Vec4i >, float > elem1, 
	const pair< pair< string, Vec4i >, float > elem2)
{
	return (elem1.second > elem2.second);
}

static bool IOUSortComp(
	const pair< int, float > elem1, 
	const pair< int, float > elem2)
{
	return (elem1.second > elem2.second);
}


/***********************************Init*************************************/
/// construct function 
API_MAINBOBY::API_MAINBOBY()
{
}

/// destruct function 
API_MAINBOBY::~API_MAINBOBY(void)
{
}

/***********************************Init*************************************/
int API_MAINBOBY::Init( 
	const char* 	KeyFilePath,						//[In]:KeyFilePath
	const char* 	layerName,							//[In]:layerName:"fc7"
	const int		binGPU, 							//[In]:USE GPU(1) or not(0)
	const int		deviceID )	 						//[In]:GPU ID
{
	int i,nRet = 0;
	char tPath[1024] = {0};
	char tPath2[1024] = {0};
	char tPath3[1024] = {0};

	/***********************************Load similardetect File**********************************/
	SD_GLOBAL_1_1_0::ClassInfo ci;
	vector < SD_GLOBAL_1_1_0::ClassInfo > classList;
	
	sprintf(tPath, "%s/mainboby/similarDetect/keyfile",KeyFilePath);
	printf("load similardetect File:%s\n",tPath);
	nRet  = SD_GLOBAL_1_1_0::Init(tPath);
	if (nRet != 0)
	{
	   LOOGE<<"similardetect init err :can't open"<<tPath;
	   return TEC_INVALID_PARAM;
	}
	
	ci.ClassID = 70020;
	ci.SubClassID = 70020;
	classList.push_back(ci);
	nRet = SD_GLOBAL_1_1_0::LoadClassData(classList);
	if (nRet != 0)
	{
	   LOOGE<<"similardetect init err : Fail to LoadClassData. Error code:"<<nRet;
	   return TEC_INVALID_PARAM;
	}

	/***********************************Init Bing*************************************/
	objNess = new Objectness(2, 8, 2);
	sprintf(tPath, "%s/mainboby/voc2007/ObjNessB2W8MAXBGR",KeyFilePath);
	objNess->loadTrainedModelOnly(tPath);
	
	/***********************************Init api_caffe*************************************/
	sprintf(tPath, "%s/mainboby/googlenet/googlenet_deploy.prototxt",KeyFilePath);
	sprintf(tPath2, "%s/mainboby/googlenet/googlenet_models_iter_26000.caffemodel",KeyFilePath);
	sprintf(tPath3, "%s/mainboby/googlenet/googlenet_imagenet_mean.binaryproto",KeyFilePath);
	nRet = api_caffe.Init( tPath, tPath2, tPath3, layerName, binGPU, deviceID ); 
	if (nRet != 0)
	{
	   LOOGE<<"Fail to initialization ";
	   return TEC_INVALID_PARAM;
	}

	/***********************************Load dic_90Class File**********************************/
	dic_90Class.clear();
	sprintf(tPath, "%s/mainboby/googlenet/dict_90class",KeyFilePath);
	printf("load dic_90Class:%s\n",tPath);
	api_commen.loadWordDict(tPath,dic_90Class);
	printf( "dict:size-%d,tag:", int(dic_90Class.size()) );
	for ( i=0;i<dic_90Class.size();i++ )
	{
		printf( "%d-%s ",i,dic_90Class[i].c_str() );
	}
	printf( "\n" );

	return nRet;
}

int API_MAINBOBY::GetCldEhdFeat_Hypothese( IplImage *MainBody, vector< float > &feat )
{
	if(!MainBody || (MainBody->width<16) || (MainBody->height<16) || MainBody->nChannels != 3 || MainBody->depth != IPL_DEPTH_8U) 
	{	
		LOOGE<<"[input err]";
		return TEC_INVALID_PARAM;
	}	
	
	int i,j,rWidth,rHeight,nRet=TOK;
	int MaxLen = 256;	//maxlen:720-same with online
	char imgPath[1024] = {0};
	API_COMMEN api_commen;

	/***********************************Resize Image width && height*****************************/
	IplImage *imgResize;
	if( ( MainBody->width>MaxLen ) || ( MainBody->height>MaxLen ) )
	{
		nRet = api_commen.GetReWH( MainBody->width, MainBody->height, MaxLen, rWidth, rHeight );	
		if (nRet != 0)
		{
			LOOGE<<"[GetReWH err]";
			cvReleaseImage(&imgResize);imgResize = 0;
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

	feat.clear();
	/*****************************Extract Feat:CLD*****************************/
#ifdef USE_CLD_FEAT
	int CLD_Feat[CLD_DIM] = {0};
	vector< float > featCLD;
	vector< float > featCLDNorm;
	GF_INTERNAL::MultiBlock_LayoutExtractor( imgResize, CLD_Feat );	//72d
	for( j=0;j<CLD_DIM;j++ )
		featCLD.push_back( CLD_Feat[j] );
	
	nRet = api_commen.Normal_L2( featCLD,featCLDNorm );
	nRet += api_commen.Normal_MinMax( featCLDNorm,featCLD );
	if (nRet != 0)
	{
	   LOOGE<<"[Fail to Normal_L2]";
	   return nRet;
	}

	for( j=0;j<featCLD.size();j++ )
		feat.push_back( featCLD[j] );
#endif

	/*****************************Extract Feat:EHD*****************************/
#ifdef USE_EHD_FEAT
	int EHD_Feat[EHD_DIM] = {0};
	vector< float > featEHD;
	vector< float > featEHDNorm;
	GF_INTERNAL::EdgeHistExtractor( imgResize, EHD_Feat );				//80d
	for( j=0;j<EHD_DIM;j++ )
		featEHD.push_back( EHD_Feat[j] );

	nRet = api_commen.Normal_L2( featEHD,featEHDNorm );
	nRet += api_commen.Normal_MinMax( featEHDNorm,featEHD );
	if (nRet != 0)
	{
	   LOOGE<<"[Fail to Normal_L2]";
	   cvReleaseImage(&imgResize);imgResize = 0;
	   return nRet;
	}

	for( j=0;j<featEHD.size();j++ )
		feat.push_back( featEHD[j] );
#endif

	cvReleaseImage(&imgResize);imgResize = 0;

	return nRet;
}

int API_MAINBOBY::GetTwoRect_Intersection_Union( ValStructVec<float, Vec4i> inBox, double &Intersection, double &Union  )
{
	if (inBox.size()!=2)
	{	
		LOOGE<<"[GetTwoRect_Union_Intersection init err]";
		return TEC_INVALID_PARAM;
	}
	
	double x[5]={0};
	double y[5]={0};
	int xy[4][4] = {0};
	int m,i,j,i1,i2,j1,j2,k,kk,n=inBox.size();

    k=0;
    memset(xy,0,sizeof(xy));
    for(i=0;i<n;i++)
    {
        x[k]=inBox[i][0];
        y[k]=inBox[i][1];
        k++;
        x[k]=inBox[i][2];
        y[k]=inBox[i][3];
        k++;
    }
    sort(x,x+2*n);
    sort(y,y+2*n);
	kk = 0;
    for(k=0;k<n;k++)
    {
        for(i1=0;i1<2*n;i1++)
        {
            if(x[i1]==inBox[k][0])
                break;
        }
        for(i2=0;i2<2*n;i2++)
        {
            if(x[i2]==inBox[k][2])
                break;
        }
        for(j1=0;j1<2*n;j1++)
        {
            if(y[j1]==inBox[k][1])
                break;
        }
        for(j2=0;j2<2*n;j2++)
        {
            if(y[j2]==inBox[k][3])
                break;
        }
        for(i=i1;i<i2;i++)
        {
            for(j=j1;j<j2;j++)
            {
                xy[i][j] |= 1<<k;
            }
        }
		kk |= 1<<k;
    }

	Union = 0.0;
	Intersection = 0.0;
    for(i=0;i<2*n;i++)
    {
        for(j=0;j<2*n;j++)
        {
			Union += ((xy[i][j] != 0 ? 1:0)*(x[i+1]-x[i])*(y[j+1]-y[j]));
            Intersection += ((xy[i][j] == kk ? 1:0)*(x[i+1]-x[i])*(y[j+1]-y[j])); 
        }
    }
    //printf("Rect-1:%d-%d-%d-%d\n",inBox[0][0],inBox[0][1],inBox[0][2],inBox[0][3]);
	//printf("Rect-2:%d-%d-%d-%d\n",inBox[1][0],inBox[1][1],inBox[1][2],inBox[1][3]);
    //printf("Union: %.2f\n",Union);
	//printf("Intersection: %.2f\n",Intersection);
    //printf("\n");

    return 0;
}

int API_MAINBOBY::GetTwoRect_Intersection_Union_vector( vector<Vec4i> inBox, double &Intersection, double &Union )
{
	if ( inBox.size()<1 )
	{	
		LOOGE<<"[GetTwoRect_Union_Intersection init err]";
		return TEC_INVALID_PARAM;
	}
	
	double x[5]={0};
	double y[5]={0};
	int xy[4][4] = {0};
	int m,i,j,i1,i2,j1,j2,k,kk,n=inBox.size();

    k=0;
    memset(xy,0,sizeof(xy));
    for(i=0;i<n;i++)
    {
        x[k]=inBox[i][0];
        y[k]=inBox[i][1];
        k++;
        x[k]=inBox[i][2];
        y[k]=inBox[i][3];
        k++;
    }
    sort(x,x+2*n);
    sort(y,y+2*n);
	kk = 0;
    for(k=0;k<n;k++)
    {
        for(i1=0;i1<2*n;i1++)
        {
            if(x[i1]==inBox[k][0])
                break;
        }
        for(i2=0;i2<2*n;i2++)
        {
            if(x[i2]==inBox[k][2])
                break;
        }
        for(j1=0;j1<2*n;j1++)
        {
            if(y[j1]==inBox[k][1])
                break;
        }
        for(j2=0;j2<2*n;j2++)
        {
            if(y[j2]==inBox[k][3])
                break;
        }
        for(i=i1;i<i2;i++)
        {
            for(j=j1;j<j2;j++)
            {
                xy[i][j] |= 1<<k;
            }
        }
		kk |= 1<<k;
    }

	Union = 0.0;
	Intersection = 0.0;
    for(i=0;i<2*n;i++)
    {
        for(j=0;j<2*n;j++)
        {
			Union += ((xy[i][j] != 0 ? 1:0)*(x[i+1]-x[i])*(y[j+1]-y[j]));
            Intersection += ((xy[i][j] == kk ? 1:0)*(x[i+1]-x[i])*(y[j+1]-y[j])); 
        }
    }
	
    //printf("Rect-1:%d-%d-%d-%d\n",inBox[0][0],inBox[0][1],inBox[0][2],inBox[0][3]);
	//printf("Rect-2:%d-%d-%d-%d\n",inBox[1][0],inBox[1][1],inBox[1][2],inBox[1][3]);
    //printf("Union: %.2f\n",Union);
	//printf("Intersection: %.2f\n",Intersection);
    //printf("\n");

    return 0;
}



int API_MAINBOBY::GetIouFeat_Hypothese( int index, ValStructVec<float, Vec4i> inBox, vector< float > &feat )
{
	if ( (index<0) || (inBox.size()<1) )
	{	
		LOOGE<<"[GetIouFeat_Hypothese init err]";
		return TEC_INVALID_PARAM;
	}
	
	double Intersection, Union, Iou = 0;
	int i, j, nRet = 0;
	ValStructVec<float, Vec4i> twoBox;

	feat.clear();
	for(i=0;i<inBox.size();i++)
	{
		twoBox.clear();
		twoBox.pushBack(inBox(index), inBox[index]); 
		twoBox.pushBack(inBox(i), inBox[i]); 

		Union = 0;
		Intersection = 0;
		nRet = GetTwoRect_Intersection_Union( twoBox, Intersection, Union );
		if (nRet!=0) 
		{
			LOOGE<<"[GetTwoRect_Intersection_Union]";
			return TEC_INVALID_PARAM;
		}

		if ( ( Union == 0 ) || ( Intersection > Union ) )
		{
			Iou = 0;
			LOOGE<<"[Get Err Intersection && Union ]";
			return TEC_INVALID_PARAM;
		}
		
		Iou = Intersection*1.0/Union ;
		feat.push_back(float(Iou));
	}
	
    return 0;
}

// calculate entropy of an image
int API_MAINBOBY::ExtractFeat_Entropy( vector< float > inFeat, double &Entropy )
{
	if (inFeat.size()<1)
	{	
		LOOGE<<"[ExtractFeat_Entropy err]";
		return TEC_INVALID_PARAM;
	}
	
	int i, nRet = 0;
	vector<float> NormFeat;

	nRet = api_commen.Normal_L2( inFeat, NormFeat );
	if (nRet!=0) 
	{
		LOOGE<<"[Normal_L2]";
		return TEC_INVALID_PARAM;
	}

	Entropy = 0;
	for(i =0;i<inFeat.size();i++)
	{
		if(inFeat[i]==0.0)
			Entropy = Entropy;
		else
			Entropy = Entropy-inFeat[i]*(log(inFeat[i])/log(2.0));
	}
	
	return nRet; 
}

int API_MAINBOBY::GetIoU_Hypothese( ValStructVec<float, Vec4i> inBox, vector< pair<int, int> > vecClusterIndex, const int MAX_CLUSTERS, ValStructVec<float, Vec4i> &outIoU )
{
	if ( (inBox.size()<1) || (vecClusterIndex.size()<1) )
	{	
		LOOGE<<"[GetIoU_Hypothese err]";
		return TEC_INVALID_PARAM;
	}

	/***********************************init*************************************/
	int i, j, index, minIndex, clusterLabel, nRet = 0;
	vector< vector< pair<int, int> > > vecClusters(MAX_CLUSTERS);
	for ( i = 0; i < MAX_CLUSTERS; i++)
		vecClusters[i].reserve(vecClusterIndex.size());
	
	for(i=0;i<vecClusterIndex.size();i++)
	{
		index = vecClusterIndex[i].first;
		clusterLabel = vecClusterIndex[i].second;
		if( clusterLabel >= MAX_CLUSTERS )
			continue;
		
		vecClusters[clusterLabel].push_back( make_pair( index, clusterLabel ) );	
	}

	/***********************************init*************************************/
	for(i=0;i<vecClusters.size();i++)
	{
		minIndex = vecClusterIndex.size()-1;
		for(j=0;j<vecClusters[i].size();j++)
		{
			index = vecClusters[i][j].first;
			if (index<minIndex)
				minIndex = index;
		}

		outIoU.pushBack(inBox(minIndex), inBox[minIndex]); 	
	}

	return nRet;
}

int API_MAINBOBY::GetIoU_Hypothese_withEntropy( 
	ValStructVec<float, Vec4i> 		inBox, 
	vector< pair<int, double> > 	vecEntropy, 
	vector< pair<int, int> > 		vecClusterIndex, 
	const int 						MAX_CLUSTERS, 
	ValStructVec<float, Vec4i> 		&outIoU )
{
	if ( (inBox.size()<1) || (vecClusterIndex.size()<1) || 
		(inBox.size()!=vecEntropy.size()) || (inBox.size()!=vecClusterIndex.size()))
	{	
		LOOGE<<"[GetIoU_Hypothese err]";
		return TEC_INVALID_PARAM;
	}

	/***********************************init*************************************/
	int i, j, m, index, indexEntropy, maxIndex, clusterLabel, nRet = 0;

	/***********************************vecClusters*************************************/
	vector< vector< pair<int, int> > > vecClusters(MAX_CLUSTERS);
	for ( i = 0; i < MAX_CLUSTERS; i++)
		vecClusters[i].reserve(vecClusterIndex.size());
	
	for(i=0;i<vecClusterIndex.size();i++)
	{
		index = vecClusterIndex[i].first;
		clusterLabel = vecClusterIndex[i].second;
		if( clusterLabel >= MAX_CLUSTERS )
			continue;
		
		vecClusters[clusterLabel].push_back( make_pair( index, clusterLabel ) );	
	}

	/***********************************init*************************************/
	for(i=0;i<vecClusters.size();i++)
	{
		maxIndex = vecEntropy.size()-1;
		indexEntropy = vecEntropy.size()-1;
		for(j=0;j<vecEntropy.size();j++)
		{
			index = vecEntropy[j].first;
			for(m=0;m<vecClusters[i].size();m++)
			{
				if (index == vecClusters[i][m].first)
				{
					maxIndex = index;
					break;
				}	
			}
			if ( maxIndex != vecEntropy.size()-1 )
			{
				indexEntropy = j;
				break;
			}
		}

		outIoU.pushBack(vecEntropy[indexEntropy].second, inBox[maxIndex]); 	
	}

	return nRet;
}

IplImage* API_MAINBOBY::ResizeImg( IplImage *img, int MaxLen )
{
	int rWidth, rHeight, nRet = 0;

	IplImage *imgResize;
	if( ( img->width>MaxLen ) || ( img->height>MaxLen ) )
	{
		nRet = api_commen.GetReWH( img->width, img->height, MaxLen, rWidth, rHeight );	
		if (nRet != 0)
		{
			LOOGE<<"[GetReWH]";
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

//BinTraining:2-NO Remove Rectfor Training;1-Remove small Rect for Training;0-Remove small Rect for Test
int API_MAINBOBY::Get_Bing_Hypothese( IplImage *img, vector< pair<float, Vec4i> > &outBox, int BinTraining )
{
	if( !img || (img->width<128) || (img->height<128) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U ) 
	{	
		LOOGE<<"[input err]";
		return TEC_INVALID_PARAM;
	}

	/***********************************init*************************************/
	int i,j,width, height, rWidth, rHeight, T_WH, topN = 1000;
	float ratio_wh = 0;

	/************************getObjBndBoxes*****************************/
	ValStructVec<float, Vec4i> inBox;
	Mat matImg(img);
	inBox.clear();
	inBox.reserve(10000);
	
	run.start();
	objNess->getObjBndBoxes(matImg, inBox, 130);
	run.end();
	LOOGI<<"[getObjBndBoxes] time:"<<run.time();

	if (inBox.size()<1)
	{	
		LOOGE<<"[getObjBndBoxes err]";
		return TEC_INVALID_PARAM;
	}

	width = img->width;
	height = img->height;
	if ( BinTraining != 0 )
	{
		T_WH = (width>height)?((int)(height*0.1)):((int)(width*0.1));
		T_WH = (T_WH>24)?T_WH:24;
	}
	else if ( BinTraining == 0 )
	{
		T_WH = (width>height)?((int)(height*0.2)):((int)(width*0.2));
		T_WH = (T_WH>32)?T_WH:32;
	}
		
	/***********************************remove small roi*************************************/
	outBox.clear();
	for(i=0;i<inBox.size();i++)
	{
		inBox[i][0] = (int)(inBox[i][0]>0.0)?inBox[i][0]:0;
		inBox[i][1] = (int)(inBox[i][1]>0.0)?inBox[i][1]:0;
		inBox[i][2] = (int)(inBox[i][2]>0.0)?inBox[i][2]:0;
		inBox[i][3] = (int)(inBox[i][3]>0.0)?inBox[i][3]:0;	

		inBox[i][0] = (int)(inBox[i][0]<width)?inBox[i][0]:(width-1);
		inBox[i][1] = (int)(inBox[i][1]<height)?inBox[i][1]:(height-1);
		inBox[i][2] = (int)(inBox[i][2]<width)?inBox[i][2]:(width-1);
		inBox[i][3] = (int)(inBox[i][3]<height)?inBox[i][3]:(height-1);
		
		rWidth = inBox[i][2]-inBox[i][0];
		rHeight = inBox[i][3]-inBox[i][1];	

		//remove err roi
		if ( (rWidth<1) || (rHeight<1) )
			continue;

		if ( (inBox[i][0]<0) || (inBox[i][1]<0) ||(inBox[i][2]<0) ||(inBox[i][3]<0) )
		{
			printf("out of rect :%d-%d-%d-%d\n",inBox[i][0],inBox[i][1],inBox[i][2],inBox[i][3]);
		}

		if ( ( BinTraining == 0 ) || ( BinTraining == 1 ) )
		{
			ratio_wh = rWidth*1.0/rHeight;
			
			//remove small roi
			if ( (rWidth<T_WH) || (rHeight<T_WH) || (ratio_wh<0.2) || (ratio_wh>5) )
				continue;
		}

		outBox.push_back( std::make_pair(inBox(i), inBox[i]) ); 
		if ( outBox.size() == topN )
			break;
	}

	/***********************************Release**********************************/
	//ValStructVec<float, Vec4i>().swap(inBox);
	if (outBox.size()<1)
	{	
		LOOGE<<"[remove small roi err]";
		return TEC_INVALID_PARAM;
	}
	
	return TOK;
}


//vectorIouCover: vector< pair<Vec4i, pair< iou, cover > > > 
int API_MAINBOBY::Get_iou_cover( 
	vector< pair< float, Vec4i > > 				bingBox, 
	vector< pair< string, Vec4i > > 			gtBox, 
	vector< pair<Vec4i, pair< double, double > > > &vectorIouCover )
{
	if ( ( bingBox.size()<1 ) || ( gtBox.size()<1 ) )
	{	
		LOOGE<<"[GetTwoRect_Union_Intersection init err]";
		return TEC_INVALID_PARAM;
	}

	double Intersection, Union, tmpIOU, maxIOU, coverGT;
	int i,j,label,nRet=0;
	vector<Vec4i> twoBox;
	vector< pair< int, float > > vectorIOU;

	vectorIouCover.clear();
	for(i=0;i<bingBox.size();i++)
	{
		vectorIOU.clear();
		for(j=0;j<gtBox.size();j++)
		{
			twoBox.clear();
			twoBox.push_back(bingBox[i].second); 
			twoBox.push_back(gtBox[j].second); 

			Intersection = 0;
			Union = 0;
			nRet = GetTwoRect_Intersection_Union_vector( twoBox, Intersection, Union );
			if ( ( nRet!= 0 ) || ( Union == 0 ) || ( Intersection > Union ) )
			{
				tmpIOU = 0;
				LOOGE<<"[Get Err Intersection && Union ]";
				continue;
			}
			
			tmpIOU = Intersection*1.0/Union ;
			vectorIOU.push_back( std::make_pair( j, tmpIOU ) );
		}

		//sort
		sort(vectorIOU.begin(), vectorIOU.end(),IOUSortComp);

		//count cover ground truth
		label = vectorIOU[0].first;
		maxIOU = vectorIOU[0].second;
		coverGT = (gtBox[label].second[2]- gtBox[label].second[0])*(gtBox[label].second[3]- gtBox[label].second[1]);
		if ( coverGT > 0 )
		{
			coverGT = Intersection*1.0/coverGT ;
			//printf("maxIOU:%.4f,coverGT:bing-%d-%d-%d-%d,gt-%d-%d-%d-%d\n",maxIOU,
			//	bingBox[i].second[0],bingBox[i].second[1],bingBox[i].second[2],bingBox[i].second[3],
			//	gtBox[label].second[0],gtBox[label].second[1],gtBox[label].second[2],gtBox[label].second[3]);
		}

		if ( (maxIOU>=0.7) && (coverGT==1) || (maxIOU<=0.3) && (coverGT!=1) )
			vectorIouCover.push_back( std::make_pair( bingBox[i].second, std::make_pair( maxIOU, coverGT ) ) );
	}
	
	return 0;
}


int API_MAINBOBY::Get_Hypothese( IplImage *img, ValStructVec<float, Vec4i> &outBox )
{
	if( !img || (img->width<128) || (img->height<128) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U ) 
	{	
		LOOGE<<"[input err]";
		return TEC_INVALID_PARAM;
	}

	/***********************************init*************************************/
	int i, j, rWidth,rHeight, width, height, T_WH, nRet = 0;
	float ratio_wh = 0;
	double Entropy = 0;
	int KMeans_topN = 15;
	const int MAX_CLUSTERS = 5;
	RunTimer<double> run;
	
	Mat KMeans_labels;
	Mat KMeans_centers;

	ValStructVec<float, Vec4i> bigBoxes;
	vector< pair<int, int> > vecClusterIndex;
	map< int, float > mapIndexIoU;
	vector< float > featHypothese;
	vector< pair<int,double> > vecEntropy;

	/************************getObjBndBoxes*****************************/
	ValStructVec<float, Vec4i> inBox;
	Mat matImg(img);
	inBox.clear();
	inBox.reserve(10000);
	
	run.start();
	objNess->getObjBndBoxes(matImg, inBox, 130);
	run.end();
	LOOGI<<"[getObjBndBoxes] time:"<<run.time();

	T_WH = (img->width>img->height)?((int)(img->height*0.2)):((int)(img->width*0.2));
		
	/***********************************remove small roi*************************************/
	bigBoxes.clear();
	for(i=0;i<inBox.size();i++)
	{
		width = inBox[i][2]-inBox[i][0];
		height = inBox[i][3]-inBox[i][1];
		ratio_wh = width*1.0/height;

		//remove small roi
		if ( (width<T_WH) || (height<T_WH) || (ratio_wh<0.2) || (ratio_wh>5) )
			continue;

		Vec4i box = inBox[i];
		bigBoxes.pushBack(inBox(i), box); 
		if ( bigBoxes.size() == KMeans_topN )
			break;
	}

	if (bigBoxes.size()<1)
	{	
		LOOGE<<"[remove small roi err]";
		return TEC_INVALID_PARAM;
	}

	/***********************************getROI && Feat*************************************/
	KMeans_topN = bigBoxes.size();
	LOOGI<<"[KMeans_topN] Num:"<<KMeans_topN;

	/***********************************Feat Extract*************************************/
	Mat KMeans_Feat(KMeans_topN, EHD_DIM, CV_32F);
	//Mat KMeans_Feat(KMeans_topN, CLD_DIM+EHD_DIM, CV_32F);
	//Mat KMeans_Feat(KMeans_topN, KMeans_topN, CV_32F);
	
	run.start();
	vecEntropy.clear();
	for(i=0;i<KMeans_topN;i++)
	{
		/***********************************Image Feat Extract*************************************/
		width = bigBoxes[i][2]-bigBoxes[i][0];
		height = bigBoxes[i][3]-bigBoxes[i][1];

		cvSetImageROI( img,cvRect(bigBoxes[i][0],bigBoxes[i][1], width, height) );	//for in36class && ads6class
		IplImage* MutiROI = cvCreateImage(cvGetSize(img),img->depth,img->nChannels);
		cvCopy( img, MutiROI, NULL );
		cvResetImageROI(img);	

		featHypothese.clear();
		nRet = GetCldEhdFeat_Hypothese( MutiROI, featHypothese );
		if (nRet!=0) 
		{
			LOOGE<<"[GetCldEhdFeat_Hypothese]";
			cvReleaseImage(&MutiROI);MutiROI = 0;
			return TEC_INVALID_PARAM;
		}
	    
		cvReleaseImage(&MutiROI);MutiROI = 0;

		/***********************************IoU Feat Extract*************************************/
/*		featHypothese.clear();
		nRet = GetIouFeat_Hypothese( i, bigBoxes, featHypothese );
		if ( (nRet!=0) || ( KMeans_topN != featHypothese.size() ) )
		{
			LOOGE<<"[GetIouFeat_Hypothese]";
			return TEC_INVALID_PARAM;
		}*/

		/***********************************ExtractFeat_Entropy*************************************/
/*		Entropy = 0;
		nRet = ExtractFeat_Entropy( featHypothese, Entropy );
		if (nRet!=0)
		{
			LOOGE<<"[ExtractFeat_Entropy]";
			return TEC_INVALID_PARAM;
		}

		vecEntropy.push_back( std::make_pair( i, Entropy ) );

		//sort vecEntropy
		sort(vecEntropy.begin(), vecEntropy.end(),vecEntropySortComp);	*/

		/***********************************write feat*************************************/
		for (j = 0; j < featHypothese.size(); j++)
    		KMeans_Feat.at<int>(i,j) = featHypothese[j];
	}
	run.end();
	LOOGI<<"[getROI && Feat] time:"<<run.time();

	/***********************************kmeans*************************************/
	run.start();
	kmeans(KMeans_Feat, MAX_CLUSTERS, KMeans_labels,
               TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0),
               3, KMEANS_PP_CENTERS, KMeans_centers);
	run.end();
	//LOOGI<<"[kmeans] time:"<<run.time();

	for( i = 0; i < KMeans_topN; i++ )
    {
		vecClusterIndex.push_back( make_pair( i, KMeans_labels.at<int>(i) ) );		
    }

	/***********************************GetIoU_Hypothese*************************************/
	outBox.clear();
	run.start();
	nRet = GetIoU_Hypothese( bigBoxes, vecClusterIndex, MAX_CLUSTERS, outBox );
	//nRet = GetIoU_Hypothese_withEntropy( bigBoxes, vecEntropy, vecClusterIndex, MAX_CLUSTERS, outBox );
	run.end();
	LOOGI<<"[GetIoU_Hypothese] time:"<<run.time();
	if ( (nRet!=0) || (outBox.size()<1) )
	{
		LOOGE<<"[GetIoU_Hypothese]";
		return TEC_INVALID_PARAM;
	}
	
	return nRet;
}

// calculate entropy of an image
double API_MAINBOBY::ExtractFeat_Entropy_cell(Mat img)
{
	double temp[256];
	for(int i=0;i<256;i++)
	{
		temp[i] = 0.0;
	}

	for(int m=0;m<img.rows;m++)
	{
		const uchar* t = img.ptr<uchar>(m);
		for(int n=0;n<img.cols;n++)
		{
			int i = t[n];
			temp[i] = temp[i]+1;
		}
	}

	for(int i=0;i<256;i++)
	{
		temp[i] = temp[i]/(img.rows*img.cols);
	}

	double result = 0;
	for(int i =0;i<256;i++)
	{
		if(temp[i]==0.0)
			result = result;
		else
			result = result-temp[i]*(log(temp[i])/log(2.0));
	}
	
	return result; 
}


int API_MAINBOBY::Get_Hypothese_Entropy( IplImage *img, UInt64 ImageID, ValStructVec<float, Vec4i> &outBox )
{
	if( !img || (img->width<128) || (img->height<128) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U ) 
	{	
		LOOGE<<"[input err]";
		return TEC_INVALID_PARAM;
	}

	/***********************************init*************************************/
	int i, j, index, width, height, T_WH, nRet = 0;
	float ratio_wh = 0;
	double Entropy = 0;
	int topN = 20;
	const int topN_Entropy = 5;
	RunTimer<double> run;

	ValStructVec<float, Vec4i> bigBoxes;
	vector< pair<int, int> > vecClusterIndex;
	map< int, float > mapIndexIoU;
	vector< float > featHypothese;
	vector< pair<int,float> > vecEntropy;
	vector< float > vecEntropyOnly;

	/************************getObjBndBoxes*****************************/
	ValStructVec<float, Vec4i> inBox;
	Mat matImg(img);
	inBox.clear();
	inBox.reserve(10000);
	
	run.start();
	objNess->getObjBndBoxes(matImg, inBox, 130);
	run.end();
	LOOGI<<"[getObjBndBoxes] time:"<<run.time();

	/***********************************init*************************************/
	float scoreBingMin = inBox(inBox.size()-1);
	float scoreBingMax = inBox(0);
	float scoreBingMinMax = scoreBingMax-scoreBingMin;
	T_WH = (img->width>img->height)?((int)(img->height*0.2)):((int)(img->width*0.2));
		
	/***********************************remove small roi*************************************/
	bigBoxes.clear();
	for(i=0;i<inBox.size();i++)
	{
		width = inBox[i][2]-inBox[i][0];
		height = inBox[i][3]-inBox[i][1];
		ratio_wh = width*1.0/height;

		//remove small roi
		if ( (width<T_WH) || (height<T_WH) || (ratio_wh<0.2) || (ratio_wh>5) )
			continue;

		Vec4i box = inBox[i];
		bigBoxes.pushBack(inBox(i), box); 
		if ( bigBoxes.size() == topN )
			break;
	}

	if (bigBoxes.size()<1)
	{	
		LOOGE<<"[remove small roi err]";
		return TEC_INVALID_PARAM;
	}
	
	run.start();
	vecEntropyOnly.clear();
	for(i=0;i<bigBoxes.size();i++)
	{
		//Image Feat Extract
		width = bigBoxes[i][2]-bigBoxes[i][0];
		height = bigBoxes[i][3]-bigBoxes[i][1];

		cvSetImageROI( img,cvRect(bigBoxes[i][0],bigBoxes[i][1], width, height) );	//for in36class && ads6class
		IplImage* MutiROI = cvCreateImage(cvGetSize(img),img->depth,img->nChannels);
		cvCopy( img, MutiROI, NULL );
		cvResetImageROI(img);	

		IplImage* imgResize = cvCreateImage(cvSize(64, 64), MutiROI->depth, MutiROI->nChannels);
		cvResize( MutiROI, imgResize );

		IplImage* grayResize = cvCreateImage(cvGetSize(imgResize), MutiROI->depth, 1 );
		cvCvtColor(imgResize, grayResize, CV_BGR2GRAY); 

		Mat matGray(grayResize);
	    
		//ExtractFeat_Entropy
		Entropy = ExtractFeat_Entropy_cell( matGray );
		vecEntropyOnly.push_back(float(Entropy));

		cvReleaseImage(&MutiROI);MutiROI = 0;
		cvReleaseImage(&imgResize);imgResize = 0;
		cvReleaseImage(&grayResize);grayResize = 0;
	}
	run.end();
	LOOGI<<"[getROI && Feat] time:"<<run.time();
	

	//normlize
	vector < float > NormEntropy;
	nRet = api_commen.Normal_L2( vecEntropyOnly, NormEntropy );
	if (nRet!=0)
	{
		LOOGE<<"[Normal_L2]";
		return TEC_INVALID_PARAM;
	}

	//normlize
	printf("ImageID:%lld,Entropy:",ImageID);
	vecEntropy.clear();
	for(i=0;i<bigBoxes.size();i++)
	{
		vecEntropy.push_back( std::make_pair( i, (NormEntropy[i]*(bigBoxes(i)-scoreBingMin)/scoreBingMinMax) ) );
		printf("%d:%.4f:%.4f:%.4f ", i, NormEntropy[i], (bigBoxes(i)-scoreBingMin)/scoreBingMinMax, 
			(NormEntropy[i]*(bigBoxes(i)-scoreBingMin)/scoreBingMinMax) );

		//sort vecEntropy
		sort(vecEntropy.begin(), vecEntropy.end(),vecEntropySortComp);
	}
	printf("\n\n");

	//send data
	printf("ImageID:%lld,Rank:",ImageID);
	outBox.clear();
	for(i=0;i<vecEntropy.size();i++)
	{
		index = vecEntropy[i].first;
		printf("%d:%d ", i, index );
		
		outBox.pushBack(inBox(index), inBox[index]); 	
		if ( outBox.size() == topN_Entropy )
			break;
	}
	printf("\n\n");

	if (outBox.size()<1)
	{	
		LOOGE<<"[remove small roi err]";
		return TEC_INVALID_PARAM;
	}
	
	return nRet;
}


/***********************************Extract Feat**********************************/
int API_MAINBOBY::ExtractFeat( 
	IplImage* 						image, 								//[In]:image
	UInt64							ImageID,							//[In]:ImageID
	const char* 					layerName,							//[In]:Layer Name by Extract
	vector< pair< int, float > > 	&imgLabel,							//[Out]label
	vector< vector< float > > 		&vecIn73ClassFeat,					//for in73class
	vector< vector< float > > 		&vecIn6ClassFeat)					//for in6class
{
	if(!image || (image->width<16) || (image->height<16) || image->nChannels != 3 || image->depth != IPL_DEPTH_8U) 
	{	
		LOOGE<<"image err!!";
		return TEC_INVALID_PARAM;
	}	

	/*****************************Init*****************************/
	int i,j,nRet = 0;
	double tPredictTime = 0;
	RunTimer<double> run;
	
	;
	vector< vector<float> > imgFeat;
	vector<float> EncodeFeat;
	vector<float> normIn73ClassFeat;
	vector<float> normIn6ClassFeat;
	
	/*****************************GetMutiImg*****************************/
	vector < Mat_ < Vec3f > > img_dl;
	nRet = api_commen.Img_GetMutiRoi( image, ImageID, img_dl );	
	if ( ( nRet != 0) || ( img_dl.size() < 1 ) )
	{
		LOOGE<<"Fail to Img_GetMutiRoi!!";
		return TEC_BAD_STATE;
	}
	
	/*****************************GetLabelFeat*****************************/	
	imgLabel.clear();
	imgFeat.clear();	
	int bExtractFeat = 3;		//[In]:Get Label(1),Extract Feat(2),both(3)
	run.start();
	//nRet = api_caffe.GetLabelFeat( img_dl, ImageID, layerName, bExtractFeat, imgLabel, imgFeat);
	nRet = api_caffe.GetLabelFeat_GoogleNet( img_dl, ImageID, GOOGLENET_NUM, layerName, bExtractFeat, imgLabel, imgFeat);
	if ( (nRet != 0) || (imgLabel.size()<1) || (imgFeat.size()<1) )
	{
		LOOGE<<"Fail to GetLabelFeat!!";
		return TEC_BAD_STATE;
	}
	run.end();
	LOOGI<<"[ExtractFeat--GetLabelFeat] time:"<<run.time();

	/************************Normal && PCA Feat*****************************/
	vecIn73ClassFeat.clear();
	vecIn6ClassFeat.clear();
	for(i=0;i<imgFeat.size();i++)
	{
		if ( i == 0 )
		{
			normIn73ClassFeat.clear();
			normIn6ClassFeat.clear();
			EncodeFeat.clear();	

			/************************DL Ads6ClassFeat Normal*****************************/
			nRet = api_commen.Normal_L2(imgFeat[i],normIn73ClassFeat);
			if (nRet != 0)
			{
			   LOOGE<<"Fail to GetFeat!!";
			   return nRet;
			}

			/************************DL In6ClassFeat Normal*****************************/
			nRet = api_commen.Normal_MinMax(imgFeat[i],normIn6ClassFeat);	//NormDLFeat FOR in5class
			if (nRet != 0)
			{
			   LOOGE<<"Fail to Normal_MinMax!!";
			   return nRet;
			}

			vecIn73ClassFeat.push_back(normIn73ClassFeat);	//linearSVM, FOR in73Class
			vecIn6ClassFeat.push_back(normIn6ClassFeat);	// FOR in6class
		}
	}

	return 0;
}

/***********************************Predict**********************************/
int API_MAINBOBY::Predict_in90class(
	IplImage						*image, 			//[In]:image
	UInt64							ImageID,			//[In]:ImageID
	const char* 					layerName,			//[In]:Layer Name by Extract
	vector< pair< string, float > >	&Res)				//[Out]:Res:In37Class/ads6Class/imgquality3class
{
	if(!image || (image->width<16) || (image->height<16) || image->nChannels != 3 || image->depth != IPL_DEPTH_8U) 
	{	
		LOOGE<<"image err!!";
		return TEC_INVALID_PARAM;
	}	

	/*****************************Init*****************************/
	int i,nRet = 0;
	double tPredictTime = 0;
	RunTimer<double> run;
	
	vector< pair< int, float > > imgLabel;
	vector< vector<float> > imgFeat;
	vector< vector<float> > vecIn73ClassFeat;
	vector< vector<float> > vecIn6ClassFeat;
	vector < pair < int,float > > tmpRes;
	vector < pair < int,float > > tmpResIn37Class;
	vector< pair< string, float > >	ResIn37Class;

	/*****************************ExtractFeat*****************************/
	run.start();
	nRet = ExtractFeat( image, ImageID, layerName, imgLabel, vecIn73ClassFeat, vecIn6ClassFeat );
	if ( (nRet != 0) || (imgLabel.size()<1) || 
		(vecIn73ClassFeat.size()<1) || (vecIn6ClassFeat.size()<1) )
	{
		LOOGE<<"Fail to GetFeat!!";
		return TEC_BAD_STATE;
	}
	run.end();
	LOOGI<<"[Predict--ExtractFeat] time:"<<run.time();

	/************************Predict In37Class*****************************/
	{
		/************************SVM 73class Predict*****************************/	
/*		tmpRes.clear();
		nRet = api_libsvm_73class.Predict( vecIn73ClassFeat, tmpRes );	//PCA_FEAT FOR 72Class
		if (nRet != 0)
		{	
			LOOGE<<"Fail to Predict SVM 72class !!";
		   	return nRet;
		}*/		

		/************************Merge37classLabel*****************************/	
		ResIn37Class.clear();
		tmpResIn37Class.clear();
		//nRet = Merge73classLabel( tmpRes, ResIn37Class, tmpResIn37Class );
		nRet = Merge73classLabel( imgLabel, ResIn37Class, tmpResIn37Class );
		if (nRet != 0)
		{	
			LOOGE<<"Fail to Merge37classLabel!!";
		   	return nRet;
		}
	}

	/************************Merge Res*****************************/
	Res.clear();
	if ( ResIn37Class.size() > 0 )
		Res.push_back( make_pair( ResIn37Class[0].first, ResIn37Class[0].second ) );
	else
		Res.push_back( make_pair( "null", 0 ) );

	return nRet;
}



/***********************************Predict**********************************/
int API_MAINBOBY::Predict_in37class(
	IplImage						*image, 			//[In]:image
	UInt64							ImageID,			//[In]:ImageID
	const char* 					layerName,			//[In]:Layer Name by Extract
	vector< pair< string, float > >	&Res)				//[Out]:Res:In37Class/ads6Class/imgquality3class
{
	if(!image || (image->width<16) || (image->height<16) || image->nChannels != 3 || image->depth != IPL_DEPTH_8U) 
	{	
		LOOGE<<"image err!!";
		return TEC_INVALID_PARAM;
	}	

	/*****************************Init*****************************/
	int i,nRet = 0;
	double tPredictTime = 0;
	RunTimer<double> run;
	
	vector< pair< int, float > > imgLabel;
	vector< vector<float> > imgFeat;
	vector< vector<float> > vecIn73ClassFeat;
	vector< vector<float> > vecIn6ClassFeat;
	vector < pair < int,float > > tmpRes;
	vector < pair < int,float > > tmpResIn37Class;
	vector< pair< string, float > >	ResIn37Class;

	/*****************************ExtractFeat*****************************/
	run.start();
	nRet = ExtractFeat( image, ImageID, layerName, imgLabel, vecIn73ClassFeat, vecIn6ClassFeat );
	if ( (nRet != 0) || (vecIn73ClassFeat.size()<1) || (vecIn6ClassFeat.size()<1) )
	{
		LOOGE<<"Fail to GetFeat!!";
		return TEC_BAD_STATE;
	}
	run.end();
	LOOGI<<"[Predict--ExtractFeat] time:"<<run.time();

	/************************Predict In37Class*****************************/
	{
		/************************SVM 73class Predict*****************************/	
/*		tmpRes.clear();
		nRet = api_libsvm_73class.Predict( vecIn73ClassFeat, tmpRes );	//PCA_FEAT FOR 72Class
		if (nRet != 0)
		{	
			LOOGE<<"Fail to Predict SVM 72class !!";
		   	return nRet;
		}*/		

		/************************Merge37classLabel*****************************/	
		ResIn37Class.clear();
		tmpResIn37Class.clear();
		//nRet = Merge73classLabel( tmpRes, ResIn37Class, tmpResIn37Class );
		nRet = Merge37classLabel( imgLabel, ResIn37Class, tmpResIn37Class );
		//nRet = Merge46classLabel( tmpRes, ResIn37Class, tmpResIn37Class );
		if (nRet != 0)
		{	
			LOOGE<<"Fail to Merge37classLabel!!";
		   	return nRet;
		}
		
		//Merge37classLabel
		if (ResIn37Class[0].first == "other.other.other")	//recall
		{ 
			/************************SVM in6Class Predict*****************************/	
			tmpRes.clear();
			//printf("api_libsvm_in6Class.Predict!\n");
/*			nRet = api_libsvm_in6Class.Predict( vecIn6ClassFeat, tmpRes );	// FOR in6class
			if (nRet != 0)
			{	
				LOOGE<<"Fail to Predict SVM in5class!!";
			   	return nRet;
			}*/

			/************************MergeIn6ClassLabel*****************************/	
			ResIn37Class.clear();
			tmpResIn37Class.clear();
			//printf("api_libsvm_in6Class.MergeIn6ClassLabel!\n");
			//nRet = MergeIn37_6ClassLabel( tmpRes, ResIn37Class, tmpResIn37Class );
			nRet = MergeIn46_6ClassLabel( tmpRes, ResIn37Class, tmpResIn37Class );
			if (nRet != 0)
			{	
				LOOGE<<"Fail to MergeIn6ClassLabel!!";
			   	return nRet;
			}
		}
	}

	/************************Merge Res*****************************/
	Res.clear();
	if ( ResIn37Class.size() > 0 )
		Res.push_back( make_pair( ResIn37Class[0].first, ResIn37Class[0].second ) );
	else
		Res.push_back( make_pair( "null", 0 ) );

	return nRet;
}

/***********************************Predict_Hypothese**********************************/
int API_MAINBOBY::Predict_Hypothese(
	IplImage						*image, 			//[In]:image
	UInt64							ImageID,			//[In]:ImageID
	const char* 					layerName)			//[In]:Layer Name by Extract
{
	if(!image || (image->width<16) || (image->height<16) || image->nChannels != 3 || image->depth != IPL_DEPTH_8U ) 
	{	
		LOOGE<<"input err!!";
		return TEC_INVALID_PARAM;
	}

	int i, j, width, height, nRet = 0;
	vector< pair< string, float > >	Res;

	/************************Get_Hypothese*****************************/
	ValStructVec<float, Vec4i> HypotheseBox;
	HypotheseBox.clear();
	nRet = Get_Hypothese( image, HypotheseBox );
	if ( (nRet!=0) || (HypotheseBox.size()<1) )
	{
		LOOGE<<"[Get_Hypothese Err!!ImageID:]"<<ImageID;
		return TEC_INVALID_PARAM;
	}

	for(i=0;i<HypotheseBox.size();i++)
	{
		/***********************************Image Feat Extract*************************************/
		width = HypotheseBox[i][2]-HypotheseBox[i][0];
		height = HypotheseBox[i][3]-HypotheseBox[i][1];

		cvSetImageROI( image,cvRect(HypotheseBox[i][0],HypotheseBox[i][1], width, height) );	//for in36class && ads6class
		IplImage* MutiROI = cvCreateImage(cvGetSize(image),image->depth,image->nChannels);
		cvCopy( image, MutiROI, NULL );
		cvResetImageROI(image);	

		IplImage *ImgResize = cvCreateImage(cvSize(255, 255), MutiROI->depth, MutiROI->nChannels);
		cvResize( MutiROI, ImgResize );

		/***********************************SimilarDetect**********************************/
		//printf( "Start GetAdsLabel[1]:SimilarDetect...\n");
		SD_RES result_SD;
		nRet = SD_GLOBAL_1_1_0::SimilarDetect( ImgResize, ImageID, 70020, 70020, &result_SD, 0 );//search SD
		if ( nRet!=TOK )
		{
			LOOGE<<"SimilarDetect err!!";
			cvReleaseImage(&MutiROI);MutiROI = 0;
			cvReleaseImage(&ImgResize);ImgResize = 0;
			return nRet;
		}

		//detect similar image
		if ( result_SD.sMode == SD_eSame )
		{
			cvReleaseImage(&MutiROI);MutiROI = 0;
			cvReleaseImage(&ImgResize);ImgResize = 0;
			continue;
		}

		/***********************************Predict_in90class**********************************/
		Res.clear();
		//nRet = Predict_in37class( ImgResize, ImageID, layerName, Res );
		nRet = Predict_in90class( ImgResize, ImageID, layerName, Res );
		if (nRet != 0) 
		{
			LOOGE<<"Fail to Predict_in37class!!";
			cvReleaseImage(&MutiROI);MutiROI = 0;
			cvReleaseImage(&ImgResize);ImgResize = 0;
			return TEC_BAD_STATE;
		}

		/************************save img data*****************************/
		//if (Res[0].first != "other.other.other") 
		if ( (Res[0].second >= 0.7 ) && (width>96) && (height>96) )
		{	
			sprintf(szImgPath, "res_roi/%s/%s_%.2f_%lld.jpg",
				Res[0].first.c_str(),Res[0].first.c_str(),Res[0].second,ImageID);
			cvSaveImage( szImgPath,ImgResize );
		}
		    
		cvReleaseImage(&MutiROI);MutiROI = 0;
		cvReleaseImage(&ImgResize);ImgResize = 0;
	}

	return nRet;
}

/***********************************Predict**********************************/
int API_MAINBOBY::Predict(
	IplImage										*image, 			//[In]:image
	UInt64											ImageID,			//[In]:ImageID
	const char* 									layerName,
	vector< pair< pair< string, Vec4i >, float > >	&Res)			//[In]:Layer Name by Extract
{
	if(!image || (image->width<16) || (image->height<16) || image->nChannels != 3 || image->depth != IPL_DEPTH_8U ) 
	{	
		LOOGE<<"input err!!";
		return TEC_INVALID_PARAM;
	}

	int i, j, width, height, nRet = 0;
	int topN = 5;
	vector< pair< string, float > >					singleRes;
	vector< pair< pair< string, Vec4i >, float > >	MergeRes;

	/************************Get_Hypothese*****************************/
	ValStructVec<float, Vec4i> HypotheseBox;
	HypotheseBox.clear();
	nRet = Get_Hypothese( image, HypotheseBox );
	if ( (nRet!=0) || (HypotheseBox.size()<1) )
	{
		LOOGE<<"[Get_Hypothese Err!!ImageID:]"<<ImageID;
		return TEC_INVALID_PARAM;
	}

	MergeRes.clear();
	for(i=0;i<HypotheseBox.size();i++)
	{	
		//Predict singleRes
		width = HypotheseBox[i][2]-HypotheseBox[i][0];
		height = HypotheseBox[i][3]-HypotheseBox[i][1];

		cvSetImageROI( image,cvRect(HypotheseBox[i][0],HypotheseBox[i][1], width, height) );	//for in36class && ads6class
		IplImage* MutiROI = cvCreateImage(cvGetSize(image),image->depth,image->nChannels);
		cvCopy( image, MutiROI, NULL );
		cvResetImageROI(image);	

		IplImage *ImgResize = cvCreateImage(cvSize(255, 255), MutiROI->depth, MutiROI->nChannels);
		cvResize( MutiROI, ImgResize );

		singleRes.clear();
		nRet = Predict_in90class( ImgResize, ImageID, layerName, singleRes );
		//nRet = Predict_in37class( ImgResize, ImageID, layerName, singleRes );
		if (nRet != 0) 
		{
			LOOGE<<"Fail to Predict_in37class!!";
			cvReleaseImage(&MutiROI);MutiROI = 0;
			cvReleaseImage(&ImgResize);ImgResize = 0;
			return TEC_BAD_STATE;
		}

		//push data
		MergeRes.push_back(make_pair(make_pair(singleRes[0].first,HypotheseBox[i]),singleRes[0].second));
		    
		cvReleaseImage(&MutiROI);MutiROI = 0;
		cvReleaseImage(&ImgResize);ImgResize = 0;
	}

	Res.clear();
	if ( MergeRes.size()<1 )
	{
		//null Res
		Res.push_back(make_pair(make_pair("null",HypotheseBox[0]),1));

		LOOGE<<"[Res.size()<1!!ImageID:]"<<ImageID;
		return nRet;
	}
	
	//sort AllRes
	sort(MergeRes.begin(), MergeRes.end(),PredictResSortComp);

	/************************Merge Res*****************************/
	nRet = Merge_Predict( MergeRes, Res);
	if (nRet != 0) 
	{
		LOOGE<<"Fail to Merge_Predict!!";
		return TEC_BAD_STATE;
	}
/*	topN = (MergeRes.size()>topN)?topN:MergeRes.size();
	Res.assign( MergeRes.begin(), MergeRes.begin()+topN );*/

	return nRet;
}

int API_MAINBOBY::Merge_Predict(
	vector< pair< pair< string, Vec4i >, float > > 	inImgLabel,
	vector< pair< pair< string, Vec4i >, float > > 	&outImgLabel)
{
	if ( inImgLabel.size() < 1 ) 
	{ 
		LOOGE<<"MergeLabel[err]:inImgLabel.size()<1!!";
		return TEC_INVALID_PARAM;
	}

	int i, tmpSize, maxSize, maxIndex, nRet = 0;
	int topN = 5;
	vector< pair< pair< string, Vec4i >, float > >	MergeRes;
	
	map< string, pair< float, Vec4i > > filterInfo;
	map< string, pair< float, Vec4i > >::iterator itFilterInfo;

	//find max size Rect
	maxSize = 0;
	maxIndex = 0;
	for ( i=0;i<inImgLabel.size();i++ )
	{
		Vec4i tmpRect = inImgLabel[i].first.second;
		tmpSize = (tmpRect[2]-tmpRect[0])*(tmpRect[3]-tmpRect[1]);
		if ( tmpSize > maxSize )
		{
			maxSize = tmpSize;
			maxIndex = i;
		}
	}

	//filter info without max Size Rect
	for ( i=inImgLabel.size()-1;i>=0;i-- )
	{
		if ( ( i!= maxIndex ) && ( inImgLabel[i].first.first != "other.other.other" ) )
		{
			filterInfo[inImgLabel[i].first.first] = make_pair(inImgLabel[i].second,inImgLabel[i].first.second);
		}
	}

	//send data without max Size Rect
	MergeRes.clear();
	for(itFilterInfo = filterInfo.begin(); itFilterInfo != filterInfo.end(); itFilterInfo++)
	{
		MergeRes.push_back(make_pair(make_pair(itFilterInfo->first,itFilterInfo->second.second),itFilterInfo->second.first));
	}
	//push max Rect Info
	MergeRes.push_back(inImgLabel[maxIndex]);

	//sort AllRes
	sort(MergeRes.begin(), MergeRes.end(),PredictResSortComp);

	/************************Merge Res*****************************/
	outImgLabel.clear();
	topN = (MergeRes.size()>topN)?topN:MergeRes.size();
	outImgLabel.assign( MergeRes.begin(), MergeRes.begin()+topN );

	return nRet;
}


int API_MAINBOBY::Merge73classLabel(
	vector< pair< int, float > >		inImgLabel, 		//[In]:ImgDetail from GetLabel
	vector< pair< string, float > > 	&LabelInfo,			//[Out]:LabelInfo
	vector< pair< int,float > > 		&intLabelInfo )		//[Out]:intLabelInfo
{
	if ( inImgLabel.size() < 1 ) 
	{ 
		LOOGE<<"MergeLabel[err]:inImgLabel.size()<1!!";
		return TEC_INVALID_PARAM;
	}
	
	int i,label,nRet = 0;
	float score = 0.0;
	LabelInfo.clear();
	intLabelInfo.clear();

	for ( i=0;i<inImgLabel.size();i++ )
	{
		label = inImgLabel[i].first;
		score = inImgLabel[i].second;		

		if (label < dic_90Class.size() )
		{
			LabelInfo.push_back( std::make_pair( dic_90Class[label], score ) );
			intLabelInfo.push_back( std::make_pair( label, score ) );
		}
		else
		{ 
			LOOGE<<"Merge74classLabel[err]!!";
			return TEC_INVALID_PARAM;
		}
	}
	
	return nRet;
}

int API_MAINBOBY::Merge37classLabel(
	vector< pair< int, float > >		inImgLabel, 		//[In]:ImgDetail from GetLabel
	vector< pair< string, float > > 	&LabelInfo,			//[Out]:LabelInfo
	vector< pair< int,float > > 		&intLabelInfo )		//[Out]:intLabelInfo
{
	if ( inImgLabel.size() < 1 ) 
	{
		LOOGE<<"MergeLabel[err]:inImgLabel.size()<1!!";
		return 1;
	}
	
	int i,index,tmpLabel,nRet = 0;
	float score = 0.0;
	const int onlineLabel[74] = {	1, 8, 2, 0, 8, 3, 8, 8, 8, 4,
			       					0, 5, 6, 8, 8, 8, 8, 7, 19,9,
			       					15,19,15,0, 19,10,11,0, 12,13,
			       					0, 0, 0, 11,16,15,0, 0, 15,19,
			       					18,0, 17,25,20,14,21,25,22,23,
			        				24,28,26,27,28,0, 33,33,33,29,
			        				0, 0, 33,33,33,30,31,34,32,33,
			        				33,35,36,0 };

	const float T_Label[74] = {		0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.6, 0.7,	//fruit
			       					0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7,
			       					0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.5, 0.7, 0.7,	//flower
			       					0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.8,	//shoe
			       					0.7, 0.7, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6,	//people.*
			        				0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7,
			        				0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7,
			        				0.6, 0.7, 0.7, 0.7 };								//text
	LabelInfo.clear();
	intLabelInfo.clear();
	
	for ( i=0;i<inImgLabel.size();i++ )
	{
		tmpLabel = 0;
		index = inImgLabel[i].first;
		score = inImgLabel[i].second;
		//printf( "MergeLabel:i-%d,index-%d,score-%.4f\n",i,index, score );
		if ( i == 0 )
		{
			if (score>=T_Label[index])
			{
				tmpLabel = onlineLabel[index];
			}
			else
			{
				tmpLabel = 0;
			}
			LabelInfo.push_back( std::make_pair( dic_in37class[tmpLabel], score ) );
			intLabelInfo.push_back( std::make_pair( tmpLabel, score ) );
			//printf("index:%d,label:%s,score:%.4f\n",tmpLabel,dic_in37class[tmpLabel].c_str(),score);
		}
	}
	
	return nRet;
}

int API_MAINBOBY::MergeIn37_6ClassLabel(
	vector< pair< int, float > > 		inImgLabel, 		//[In]:inImgLabel
	vector< pair< string, float > > 	&LabelInfo,			//[Out]:outImgLabel
	vector< pair< int,float > > 		&intLabelInfo )		//[Out]:intLabelInfo
{
	if ( inImgLabel.size() < 1 ) 
	{
		LOOGE<<"MergeLabel[err]:inImgLabel.size()<1!!";
		return TEC_INVALID_PARAM;
	}

	int i,index,tmpLabel,nRet = 0;
	float score = 0.0;
	const int onlineLabel[6] = { 8, 0, 25, 28, 19, 33 };	//[food,other,people,pet,puppet,scene]
	LabelInfo.clear();
	intLabelInfo.clear();
	
	for ( i=0;i<inImgLabel.size();i++ )
	{
		tmpLabel = 0;
		index = inImgLabel[i].first;
		score = inImgLabel[i].second;
		//printf( "MergeLabel:i-%d,index-%d,score-%.4f\n",i,index, score );
		if ( 0 == i )
		{
			if (
				( ( index>=0  ) && ( index<2  ) && (score>=0.8) ) || 	//food other
				( ( index==2  ) 				&& (score>=0.6) ) || 	//people
				( ( index>=3  ) && ( index<6  ) && (score>=0.8) ) 		//pet puppet scene
				 )
			{
				tmpLabel = onlineLabel[index];
			}
			else
			{
				tmpLabel = 0;
			}
			LabelInfo.push_back( std::make_pair( dic_in37class[tmpLabel], score ) );
			intLabelInfo.push_back( std::make_pair( tmpLabel, score ) );
			//printf("index:%d,label:%s,score:%.4f\n",tmpLabel,dic_in37class[tmpLabel].c_str(),score);
		}
	}
	
	return nRet;
}


int API_MAINBOBY::Merge46classLabel(
	vector< pair< int, float > >		inImgLabel, 		//[In]:ImgDetail from GetLabel
	vector< pair< string, float > > 	&LabelInfo,			//[Out]:LabelInfo
	vector< pair< int,float > > 		&intLabelInfo )		//[Out]:intLabelInfo
{
	if ( inImgLabel.size() < 1 ) 
	{
		LOOGE<<"MergeLabel[err]:inImgLabel.size()<1!!";
		return 1;
	}
	
	int i,index,tmpLabel,nRet = 0;
	float score = 0.0;
	const int onlineLabel[74] = {	1, 7, 2, 0, 7, 3, 7, 7, 7, 4,
			       					7, 5, 6, 7, 7, 7, 7, 7, 8, 9,
			       					17,24,17,10,11,12,13,24,14,15,
			       					16,24,24,18,19,17,24,24,17,20,
			       					23,21,22,25,26,27,28,29,30,31,
			        				32,36,34,35,36,43,43,43,43,37,
			        				0, 0, 43,38,43,39,40,44,41,42,
			        				43,45,0, 0 };
	
	const float T_Label[74] = {		0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.6, 0.7,	//fruit
			       					0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7,
			       					0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.5, 0.7, 0.7,	//flower
			       					0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.8,	//shoe
			       					0.7, 0.7, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6,	//people.*
			        				0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7,
			        				0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7,
			        				0.6, 0.7, 0.7, 0.7 };								//text
	LabelInfo.clear();
	intLabelInfo.clear();
	
	for ( i=0;i<inImgLabel.size();i++ )
	{
		tmpLabel = 0;
		index = inImgLabel[i].first;
		score = inImgLabel[i].second;
		//printf( "MergeLabel:i-%d,index-%d,score-%.4f\n",i,index, score );
		if ( i == 0 )
		{
			if (score>=T_Label[index])
			{
				tmpLabel = onlineLabel[index];
			}
			else
			{
				tmpLabel = 0;
			}
			LabelInfo.push_back( std::make_pair( dic_mainbody46class[tmpLabel], score ) );
			intLabelInfo.push_back( std::make_pair( tmpLabel, score ) );
			//printf("index:%d,label:%s,score:%.4f\n",tmpLabel,dic_in37class[tmpLabel].c_str(),score);
		}
	}
	
	return nRet;
}

int API_MAINBOBY::MergeIn46_6ClassLabel(
	vector< pair< int, float > > 		inImgLabel, 		//[In]:inImgLabel
	vector< pair< string, float > > 	&LabelInfo,			//[Out]:outImgLabel
	vector< pair< int,float > > 		&intLabelInfo )		//[Out]:intLabelInfo
{
	if ( inImgLabel.size() < 1 ) 
	{
		LOOGE<<"MergeLabel[err]:inImgLabel.size()<1!!";
		return TEC_INVALID_PARAM;
	}

	int i,index,tmpLabel,nRet = 0;
	float score = 0.0;
	const int onlineLabel[6] = { 7, 0, 33, 36, 24, 43 };	//[food,other,people,pet,puppet,scene]
	LabelInfo.clear();
	intLabelInfo.clear();
	
	for ( i=0;i<inImgLabel.size();i++ )
	{
		tmpLabel = 0;
		index = inImgLabel[i].first;
		score = inImgLabel[i].second;
		//printf( "MergeLabel:i-%d,index-%d,score-%.4f\n",i,index, score );
		if ( 0 == i )
		{
			if (
				( ( index>=0  ) && ( index<2  ) && (score>=0.8) ) || 	//food other
				( ( index==2  ) 				&& (score>=0.6) ) || 	//people
				( ( index>=3  ) && ( index<6  ) && (score>=0.8) ) 		//pet puppet scene
				 )
			{
				tmpLabel = onlineLabel[index];
			}
			else
			{
				tmpLabel = 0;
			}
			LabelInfo.push_back( std::make_pair( dic_mainbody46class[tmpLabel], score ) );
			intLabelInfo.push_back( std::make_pair( tmpLabel, score ) );
			//printf("index:%d,label:%s,score:%.4f\n",tmpLabel,dic_in37class[tmpLabel].c_str(),score);
		}
	}
	
	return nRet;
}

/***********************************Release**********************************/
void API_MAINBOBY::Release()
{
	/***********************************Similar Detect******************************/
	SD_GLOBAL_1_1_0::Uninit();
	
	/***********************************net Model**********************************/
	api_caffe.Release();

	/***********************************dict Model**********************************/
	dic_90Class.clear();

	/***********************************SVM Model******libsvm3.2.0********************/
	//api_libsvm_73class.Release();
	//api_libsvm_in6Class.Release();

}






