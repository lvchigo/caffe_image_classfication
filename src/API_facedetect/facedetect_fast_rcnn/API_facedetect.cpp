#pragma once
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

#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>
#include <opencv/ml.h>
#include <opencv/cvaux.h>

#include "API_commen.h"
#include "plog/Log.h"
#include "API_facedetect.h"
#include "TErrorCode.h"

#include "caffe/caffe.hpp"
#include <Python.h>

using namespace caffe;
using namespace cv;
using namespace std;

static bool PredictResSortComp(
	const pair< pair< int, Vec4i >, float > elem1, 
	const pair< pair< int, Vec4i >, float > elem2)
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
API_FACE_DETECT::API_FACE_DETECT()
{
}

/// destruct function 
API_FACE_DETECT::~API_FACE_DETECT(void)
{
}

/***********************************Init*************************************/
int API_FACE_DETECT::Init( 
	const char* 	KeyFilePath,						//[In]:KeyFilePath
	const char* 	layerName,							//[In]:layerName:"fc7"
	const int		binGPU, 							//[In]:USE GPU(1) or not(0)
	const int		deviceID )	 						//[In]:GPU ID
{
	char tPath[1024] = {0};
	char tPath2[1024] = {0};
	char tPath3[1024] = {0};

	/***********************************Init Bing*************************************/
	objNess = new Objectness(2, 8, 2);
	sprintf(tPath, "%s/mainbody_frcnn/BING_voc2007/ObjNessB2W8MAXBGR",KeyFilePath);
	objNess->loadTrainedModelOnly(tPath);
	
#if USING_BINGPP
	//MAX_THREAD_NUM = omp_get_max_threads();
	//initGPU(MAX_THREAD_NUM);	//add by bingpp

	initGPU(1); //add by bingpp
#endif

	/***********************************Python**********************************/
	Py_Initialize();
	PyRun_SimpleString("print 'Hello Python!'\n");
	Py_Finalize();

	/***********************************Init**********************************/
/*	sprintf(tPath, "%s/face_detect/faster_rcnn/faster_rcnn_test.pt",KeyFilePath);
	sprintf(tPath2, "%s/face_detect/faster_rcnn/VGG16_faster_rcnn_final_face1000sample_1000iter.caffemodel",KeyFilePath);
    Caffe::SetDevice(0);
    Caffe::set_mode(Caffe::GPU);

	caffe::shared_ptr<caffe::Net<float> > net_dl;
	net_dl.reset(new caffe::Net<float>(string(tPath), caffe::TEST));
	net_dl->CopyTrainedLayersFrom( string(tPath2) );*/

	/***********************************Init FastRCNNCls**********************************/
	fastRCNN = new FastRCNNCls;
	sprintf(tPath, "%s/mainbody_frcnn/frcnn_model/CaffeNet_in_90class_20151226/test_caffenet.prototxt",KeyFilePath);	//voc2007+2012+in20151211+svd
	sprintf(tPath2, "%s/mainbody_frcnn/frcnn_model/CaffeNet_in_90class_20151226/caffenet_fast_rcnn_iter_40000.caffemodel",KeyFilePath);	//voc2007+2012+in20151211+svd
#ifdef USE_MODEL_VGG16
	sprintf(tPath, "%s/face_detect/fast_rcnn/test_vgg16_svd.prototxt",KeyFilePath);
	sprintf(tPath2, "%s/face_detect/fast_rcnn/vgg16_fast_rcnn_iter_80000_facedetect10class_20160322_svd.caffemodel",KeyFilePath);
	//sprintf(tPath, "%s/face_detect/faster_rcnn/faster_rcnn_test.pt",KeyFilePath);
	//sprintf(tPath2, "%s/face_detect/faster_rcnn/VGG16_faster_rcnn_final_face1000sample_1000iter.caffemodel",KeyFilePath);
#endif
	fastRCNN->set_model( tPath, tPath2, binGPU, deviceID );

	tgt_cls.clear();
	for (i=0; i<numObjClass; i++) {
		tgt_cls.push_back(i+1);
	}
	
	/***********************************Load dic_voc20Class File**********************************/
	dic_voc20Class.clear();
	sprintf(tPath, "%s/face_detect/Dict_FAST_RCNN_FACEDETECT10label.txt",KeyFilePath);
	printf("load dic:%s\n",tPath);
	api_commen.loadWordDict(tPath,dic_voc20Class);
	printf( "dict:size-%d,tag:", int(dic_voc20Class.size()) );
	for ( i=0;i<dic_voc20Class.size();i++ )
	{
		printf( "%d-%s ",i,dic_voc20Class[i].c_str() );
	}
	printf( "\n" );

	return nRet;
}

int API_FACE_DETECT::GetTwoRect_Intersection_Union( vector<Vec4i> inBox, double &Intersection, double &Union )
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

//vectorIouCover: vector< pair<Vec4i, pair< iou, cover > > > 
int API_FACE_DETECT::Get_iou_cover( 
	vector< pair< float, Vec4i > > 				bingBox, 
	vector< pair< string, Vec4i > > 			gtBox, 
	vector< pair<Vec4i, pair< double, int > > > &vectorIouCover )
{
	if ( ( bingBox.size()<1 ) || ( gtBox.size()<1 ) )
	{	
		LOOGE<<"[GetTwoRect_Union_Intersection init err]";
		return TEC_INVALID_PARAM;
	}

	double Intersection, Union, tmpIOU, maxIOU;
	int label, coverGT;
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
			nRet = GetTwoRect_Intersection_Union( twoBox, Intersection, Union );
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
		coverGT = 0;
		label = vectorIOU[0].first;
		maxIOU = vectorIOU[0].second;
		if ( (bingBox[i].second[0]<=gtBox[label].second[0]) && (bingBox[i].second[1]<=gtBox[label].second[1]) &&
			 (bingBox[i].second[2]>=gtBox[label].second[2]) && (bingBox[i].second[3]>=gtBox[label].second[3]) )
		{
			coverGT = 1;
			//printf("maxIOU:%.4f,coverGT:bing-%d-%d-%d-%d,gt-%d-%d-%d-%d\n",maxIOU,
			//	bingBox[i].second[0],bingBox[i].second[1],bingBox[i].second[2],bingBox[i].second[3],
			//	gtBox[label].second[0],gtBox[label].second[1],gtBox[label].second[2],gtBox[label].second[3]);
		}

		//Filter Proposal From IOU
		//if ( (maxIOU>=0.7) && (coverGT==1) || (maxIOU<=0.3) && (coverGT!=1) )
		if ( (maxIOU>=0.5) || (maxIOU<=0.3) )
			vectorIouCover.push_back( std::make_pair( bingBox[i].second, std::make_pair( maxIOU, coverGT ) ) );
	}
	
	return 0;
}

int API_FACE_DETECT::Get_Xml_Hypothese( 
	string ImageID, 
	int width, 
	int height, 
	vector< pair< string, Vec4i > > vecLabelRect, 
	vector< pair< string, Vec4i > > &vecOutRect)
{
	if( vecLabelRect.size() < 1 ) 
	{	
		LOOGE<<"[input err]";
		return TEC_INVALID_PARAM;
	}

	/***********************************init*************************************/
	int rWidth, rHeight, biasWidth, biasHeight, roiSizeNum;
	float ratio_wh = 0;
	string label;

	float roiSizeMultiple, roiSize = 0;
	roiSizeMultiple = 0.05;
	roiSizeNum = 2;	//2,4,6

	/************************get_xml_Hypothese*****************************/
	vecOutRect.clear();
	for (i=0;i<vecLabelRect.size();i++)
	{
		label = vecLabelRect[i].first;
		rWidth = vecLabelRect[i].second[2]-vecLabelRect[i].second[0];
		rHeight = vecLabelRect[i].second[3]-vecLabelRect[i].second[1];

		//printf("ImageID:%s,i:%d,label:%s,x1:%d,y1:%d,x2:%d,y2:%d,rWidth:%d,rHeight:%d\n",
		//	ImageID.c_str(),i,label.c_str(),vecLabelRect[i].second[0],vecLabelRect[i].second[1],
		//	vecLabelRect[i].second[2],vecLabelRect[i].second[3],rWidth,rHeight);

		//remove err roi
		if ( (vecLabelRect[i].second[0]<0) || (vecLabelRect[i].second[1]<0) || 
			 (vecLabelRect[i].second[2]>width) || (vecLabelRect[i].second[3]>height) )
			continue;

		//remove err roi
		if ( (rWidth<5) || (rHeight<5) )
			continue;

		roiSize = 0;
		for (j=0;j<roiSizeNum;j++)
		{
			roiSize += roiSizeMultiple;
			biasWidth = int(rWidth*roiSize+0.5);
			biasHeight = int(rHeight*roiSize+0.5);
			
			//ADD-SIZE
			if ( (vecLabelRect[i].second[0]-biasWidth>1) && (vecLabelRect[i].second[1]-biasHeight>1) && 
				 (vecLabelRect[i].second[2]+biasWidth<(width-1)) && vecLabelRect[i].second[3]+biasHeight<(height-1) )
			{
				Vec4i Rect;
				Rect[0] = vecLabelRect[i].second[0]-biasWidth;
				Rect[1] = vecLabelRect[i].second[1]-biasHeight;
				Rect[2] = vecLabelRect[i].second[2]+biasWidth;
				Rect[3] = vecLabelRect[i].second[3]+biasHeight;

				Rect[0] = (int)(Rect[0]>1.0)?Rect[0]:1;
				Rect[1] = (int)(Rect[1]>1.0)?Rect[1]:1;
				Rect[2] = (int)(Rect[2]>1.0)?Rect[2]:1;
				Rect[3] = (int)(Rect[3]>1.0)?Rect[3]:1;	

				Rect[0] = (int)(Rect[0]<width)?Rect[0]:(width-1);
				Rect[1] = (int)(Rect[1]<height)?Rect[1]:(height-1);
				Rect[2] = (int)(Rect[2]<width)?Rect[2]:(width-1);
				Rect[3] = (int)(Rect[3]<height)?Rect[3]:(height-1);
				
				vecOutRect.push_back( std::make_pair( label, Rect ) );
			}
		}

		//add self
		vecOutRect.push_back( std::make_pair( label, vecLabelRect[i].second ) );
	}

	return TOK;
}

//BinTraining:2-NO Remove Rectfor Training;1-Remove small Rect for Training;0-Remove small Rect for Test
int API_FACE_DETECT::Get_Bing_Hypothese( IplImage *img, vector< pair<float, Vec4i> > &outBox, int BinTraining )
{
	if( !img || (img->width<32) || (img->height<32) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U ) 
	{	
		LOOGE<<"[input err]";
		return TEC_INVALID_PARAM;
	}

	/***********************************init*************************************/
	int width, height, rWidth, rHeight, T_WH, topN = 1000;//100,500,1000
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
	if ( BinTraining == 0 )
	{
		T_WH = (width>height)?((int)(height*0.2)):((int)(width*0.2));
		T_WH = (T_WH>TImgSize)?T_WH:TImgSize;
	}
	else if ( BinTraining == 1 )
	{
		T_WH = (width>height)?((int)(height*0.1)):((int)(width*0.1));
		T_WH = (T_WH>TRectSize)?T_WH:TRectSize;
	}
		
	/***********************************remove small roi*************************************/
	outBox.clear();
	for(i=0;i<inBox.size();i++)
	{
		inBox[i][0] = (int)(inBox[i][0]>1.0)?inBox[i][0]:1;
		inBox[i][1] = (int)(inBox[i][1]>1.0)?inBox[i][1]:1;
		inBox[i][2] = (int)(inBox[i][2]>1.0)?inBox[i][2]:1;
		inBox[i][3] = (int)(inBox[i][3]>1.0)?inBox[i][3]:1;	

		inBox[i][0] = (int)(inBox[i][0]<width)?inBox[i][0]:(width-1);
		inBox[i][1] = (int)(inBox[i][1]<height)?inBox[i][1]:(height-1);
		inBox[i][2] = (int)(inBox[i][2]<width)?inBox[i][2]:(width-1);
		inBox[i][3] = (int)(inBox[i][3]<height)?inBox[i][3]:(height-1);
		
		rWidth = inBox[i][2]-inBox[i][0];
		rHeight = inBox[i][3]-inBox[i][1];	

		//remove err roi
		if ( (rWidth<5) || (rHeight<5) )
			continue;

		if ( (inBox[i][0]<0) || (inBox[i][1]<0) ||(inBox[i][2]<0) ||(inBox[i][3]<0) )
		{
			printf("out of rect :%d-%d-%d-%d\n",inBox[i][0],inBox[i][1],inBox[i][2],inBox[i][3]);
		}

		//remove small roi
		if ( ( BinTraining == 0 ) || ( BinTraining == 1 ) )
		{
			ratio_wh = rWidth*1.0/rHeight;
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

int API_FACE_DETECT::Predict(
	IplImage										*image, 			//[In]:image
	string											ImageID,			//[In]:ImageID
	vector< pair< pair< string, Vec4i >, float > >	&Res)			//[In]:Layer Name by Extract
{	
	if(!image || (image->width<16) || (image->height<16) || image->nChannels != 3 || image->depth != IPL_DEPTH_8U ) 
	{	
		LOOGE<<"input err!!";
		return TEC_INVALID_PARAM;
	}

	int label, T_WH, width, height, rWidth, rHeight, x1, x2, y1, y2, topN = 300;
	nRet = 0;
	float ratio_wh = 0;
	vector< pair< pair< int, Vec4i >, float > > 	vecLabel;
	vector< pair< pair< string, Vec4i >, float > >	MergeRes;
	
	/************************Get_Bing_Hypothese*****************************/
	vector< pair<float, Vec4i> > HypotheseBox;
	HypotheseBox.clear();
	nRet = Get_Bing_Hypothese( image, HypotheseBox, 2 );
	if ( (nRet!=0) || (HypotheseBox.size()<1) )
	{
		LOOGE<<"[Get_Bing_Hypothese Err!!ImageID:]"<<ImageID;
		return TEC_INVALID_PARAM;
	}
	//printf("HypotheseBox.size():%d\n",HypotheseBox.size());

	fastRCNN->set_boxes( HypotheseBox );
	
	/************************fastRCNN->detect*****************************/
	Mat imgMat(image);
	vector<PropBox> randPboxes = fastRCNN->detect(imgMat, tgt_cls);
	if ( randPboxes.size()<1 )
	{
		LOOGE<<"[randPboxes.size()<1!!ImageID:]"<<ImageID<<"HypotheseBox.size():"<<HypotheseBox.size();	
		return TEC_INVALID_PARAM;
	}
	//printf("randPboxes.size():%d\n",randPboxes.size());

	/************************data ch*****************************/
	width = image->width;
	height = image->height;
	T_WH = (width>height)?((int)(height*0.1)):((int)(width*0.1));
	T_WH = (T_WH>TRectSize)?T_WH:TRectSize;
		
	vecLabel.clear();
	//printf("ImageID_%lld:",ImageID);
	for ( i=0;i<randPboxes.size();i++ )
	{
		label = randPboxes[i].cls_id-1;
		if ( randPboxes[i].confidence < 0.5 )	//T=0.5
		{
			//printf("out of size label:%d\n",label);
			continue;
		}

		Vec4i rect;
		rect[0] = (int)(randPboxes[i].x1<0)?0:randPboxes[i].x1;
		rect[1] = (int)(randPboxes[i].y1<0)?0:randPboxes[i].x1;
		rect[2] = (int)(randPboxes[i].x2<0)?0:randPboxes[i].x1;
		rect[3] = (int)(randPboxes[i].y2<0)?0:randPboxes[i].x1;

		rect[0] = (int)(randPboxes[i].x1<width-1)?randPboxes[i].x1:(width-1);
		rect[1] = (int)(randPboxes[i].y1<height-1)?randPboxes[i].y1:(height-1);
		rect[2] = (int)(randPboxes[i].x2<width-1)?randPboxes[i].x2:(width-1);
		rect[3] = (int)(randPboxes[i].y2<height-1)?randPboxes[i].y2:(height-1);

		//remove small roi
		rWidth = rect[2]-rect[0];
		rHeight = rect[3]-rect[1];
		ratio_wh = rWidth*1.0/rHeight;
		if ( (ratio_wh<0.125) || (ratio_wh>8) )
			continue;

		vecLabel.push_back( std::make_pair( std::make_pair( label, rect ), randPboxes[i].confidence ) );
		//printf("%d_%d_%d_%d_%d_%.4f ",label, rect[0], rect[1], rect[2], rect[3], randPboxes[i].confidence );
	}
	//printf("\n");
	if ( vecLabel.size()<1 )
	{
		LOOGE<<"[vecLabel.size()<1!!ImageID:]"<<ImageID;
		return TEC_INVALID_PARAM;
	}
	//printf("vecLabel.size():%d\n",vecLabel.size());

	//sort AllRes
	sort(vecLabel.begin(), vecLabel.end(),PredictResSortComp);

	/************************MergeVOC20classLabel*****************************/
	MergeRes.clear();
 	nRet = MergeVOC20classLabel( vecLabel, MergeRes );
	if ( (nRet!=0) || (MergeRes.size()<1) )
	{
		LOOGE<<"[MergeVOC20classLabel Err!!ImageID:]"<<ImageID;
		return TEC_INVALID_PARAM;
	}

	/************************Merge Res*****************************/
	Res.clear();
/*	nRet = Merge_Predict( MergeRes, Res);
	if (nRet != 0) 
	{
		LOOGE<<"Fail to Merge_Predict!!";
		return TEC_BAD_STATE;
	}*/
	topN = (MergeRes.size()>topN)?topN:MergeRes.size();
	Res.assign( MergeRes.begin(), MergeRes.begin()+topN );
	//printf("ImageID:%lld,HypotheseBox:%d,randPboxes:%d,vecLabel:%d,Res.size():%d\n",
	//	ImageID, HypotheseBox.size(), randPboxes.size(), vecLabel.size(), Res.size());

	return nRet;
}

int API_FACE_DETECT::MergeVOC20classLabel(
	vector< pair< pair< int, Vec4i >, float > >			inImgLabel, 		//[In]:ImgDetail from GetLabel
	vector< pair< pair< string, Vec4i >, float > > 		&LabelInfo)			//[Out]:LabelInfo
{
	if ( inImgLabel.size() < 1 ) 
	{ 
		LOOGE<<"MergeLabel[err]:inImgLabel.size()<1!!";
		return TEC_INVALID_PARAM;
	}
	
	int label;
	float score = 0.0;
	LabelInfo.clear();

	for ( i=0;i<inImgLabel.size();i++ )
	{
		label = inImgLabel[i].first.first;
		Vec4i rect = inImgLabel[i].first.second;
		score = inImgLabel[i].second;		

		//if (label<dic_voc20Class.size())
		if (label<6)
		{
			LabelInfo.push_back( std::make_pair( std::make_pair( dic_voc20Class[label], rect ), score ) );
		}
		else if (label>=dic_voc20Class.size())
		{ 
			LOOGE<<"MergeVOC20classLabel[err]!!";
			return TEC_INVALID_PARAM;
		}
	}
	
	return TOK;
}

/***********************************Release**********************************/
void API_FACE_DETECT::Release()
{
	/***********************************net Model**********************************/
	delete fastRCNN;

	/***********************************Release Bing*************************************/
	delete objNess;

#if USING_BINGPP
	//releaseGPU(MAX_THREAD_NUM);//add by bingpp
	releaseGPU(1);//add by bingpp
#endif
}


