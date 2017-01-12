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

#include "API_data_augmentation/API_data_augmentation.h"

using namespace cv;
using namespace std;

static bool SortComp(const pair<int,int>& input1, const pair<int,int>& input2)
{
    return (input1.second > input2.second);
}

/***********************************Init*************************************/
/// construct function 
API_DATA_AUGMENTATION::API_DATA_AUGMENTATION()
{
}

/// destruct function 
API_DATA_AUGMENTATION::~API_DATA_AUGMENTATION(void)
{
}

/***********************************Get_Neg_Patch**********************************/
int API_DATA_AUGMENTATION::Get_Neg_Patch(
	vector < pair < string,Vec4i > > 	vecInRect,			//[In]:Input Rect
	string								patchName,			//[In]:Patch Name
	string								negName,			//[In]:Get Neg Patch Name
	const int 							width,				//[In]:image.width
	const int							height,				//[In]:image.height
	const float							T_IoU,				//[In]:T_IoU
	vector < pair < string,Vec4i > > 	&vecOutRect)		//[Out]:Res
{
	if( (vecInRect.size()<1) || (width<16) || (height<16) ) 
	{	
		printf("Get_Neg_Patch Err!!\n");
		return -1;
	}

	//init
    int i,j,PatchSize,PatchWidth,PatchHeight,tmp,nRet=0;
	float iou,max_iou,t_iou,t_cover=0.2;
	Vec4i maxRect;

	vector < pair < string,Vec4i > > 	vecInSelectedRect;
	vector < pair < string,Vec4i > > 	vecOutSelectedRect;
	vector < pair < int,int > > 		vecPatch;	//<index,PatchSize>

	/*****************************init*****************************/
	vecOutRect.clear();
	t_iou = std::max( std::min( T_IoU, float(0.3) ), float(0.0) );
	vecOutRect.assign( vecInRect.begin(), vecInRect.end() );

	/*****************************BBox NMS*****************************/
	for(i=0;i<vecInRect.size();i++)
	{
		if ( vecInRect[i].first == patchName )
		{
			vecInSelectedRect.push_back( std::make_pair( vecInRect[i].first, vecInRect[i].second ) );

			PatchSize = (vecInRect[i].second[2]-vecInRect[i].second[0])*(vecInRect[i].second[3]-vecInRect[i].second[1]);
			vecPatch.push_back( std::make_pair( vecInSelectedRect.size()-1, PatchSize ) );
		}
	}

	//unnormal
	if ( (vecInSelectedRect.size()<1) || (vecPatch.size()<1) )
		return 0;
	
	/*****************************get patch info*****************************/
	std::sort(vecPatch.begin(),vecPatch.end(),SortComp);
	tmp = vecPatch[0].first; //index
	maxRect = vecInSelectedRect[tmp].second;
	PatchWidth = maxRect[2]-maxRect[0];
	PatchHeight = maxRect[3]-maxRect[1];

	//up
	vecOutSelectedRect.clear();
	tmp = int(maxRect[1]-(1-t_cover)*PatchHeight);
	if(tmp>0)
	{
		Vec4i patch(maxRect[0],tmp,maxRect[2],tmp+PatchHeight);
		vecOutSelectedRect.push_back( std::make_pair( negName, patch ) );
	}

	//down
	tmp = int(maxRect[3]+(1-t_cover)*PatchHeight);
	if(tmp<height)
	{
		Vec4i patch(maxRect[0],tmp-PatchHeight,maxRect[2],tmp);
		vecOutSelectedRect.push_back( std::make_pair( negName, patch ) );
	}

	//left
	tmp = int(maxRect[0]-(1-t_cover)*PatchWidth);
	if(tmp>0)
	{
		Vec4i patch(tmp,maxRect[1],tmp+PatchWidth,maxRect[3]);
		vecOutSelectedRect.push_back( std::make_pair( negName, patch ) );
	}

	//right
	tmp = int(maxRect[2]+(1-t_cover)*PatchWidth);
	if(tmp<width)
	{
		Vec4i patch(tmp-PatchWidth,maxRect[1],tmp,maxRect[3]);
		vecOutSelectedRect.push_back( std::make_pair( negName, patch ) );
	}

	//unnormal
	if (vecOutSelectedRect.size()<1)
		return 0;

	/*****************************Rect_IOU*****************************/
	for(i=0;i<vecOutSelectedRect.size();i++)
	{
		max_iou = 0;
		for(j=0;j<vecInSelectedRect.size();j++)
		{
			iou = 0;
			nRet = Rect_IOU(vecOutSelectedRect[i].second, vecInSelectedRect[j].second, iou);
			if (nRet!=0)
			{
				printf("Rect_IOU Err!!\n");
				continue;
			}

			max_iou = (iou>max_iou)?iou:max_iou;
		}

		//selected patch
		if (max_iou<t_iou)
			vecOutRect.push_back( std::make_pair( vecOutSelectedRect[i].first, vecOutSelectedRect[i].second ) );
	}

	return 0;
}

int API_DATA_AUGMENTATION::Rect_IOU(Vec4i rect1, Vec4i rect2, float &iou) 
{ 
	if ( (rect1[0]<0)||(rect1[1]<0)||(rect1[0]>rect1[2])||(rect1[1]>rect1[3])||
		 (rect2[0]<0)||(rect2[1]<0)||(rect2[0]>rect2[2])||(rect2[1]>rect2[3]))
		return -1;
	
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



