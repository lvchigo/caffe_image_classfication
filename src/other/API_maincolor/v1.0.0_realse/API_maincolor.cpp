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

#include <vector>
#include <list>
#include <map>
#include <algorithm>

#include "API_maincolor.h"

using namespace std;

/***********************************Init*************************************/
/// construct function 
API_MAINCOLOR_1_0_0::API_MAINCOLOR_1_0_0()
{
}

/// destruct function 
API_MAINCOLOR_1_0_0::~API_MAINCOLOR_1_0_0(void)
{
}

/***********************************Init*************************************/
static bool ImgSortComp(
	const pair< vector< int >, float > elem1, 
	const pair< vector< int >, float > elem2)
{
	return (elem1.second > elem2.second);
}


/***********************************Predict**********************************/
int API_MAINCOLOR_1_0_0::Predict(
	uchar*										bgr,				//[In]:image->bgr
	int 										width,				//[In]:image->width
	int 										height, 			//[In]:image->height
	int 										channel,			//[In]:image->channel
	int 										numColorBlock,		//[In]:numColorBlock
	vector< pair< vector< int >,float > >		&Res)				//[Out]:Res

{
	if( !bgr || (width<16) || (height<16) || (channel != 3) || (numColorBlock<1) ) 
	{	
		cout<<"image err!!" << endl;
		return -1;
	}	

	/*****************************Init*****************************/
	int nRet = 0;

	/*****************************gaussianFilter Img*****************************/
	gaussianFilter( bgr, width, height, channel );
	gaussianFilter( bgr, width, height, channel );
	gaussianFilter( bgr, width, height, channel );

	/*****************************predict Img*****************************/
	Res.clear();
	nRet = ColorHistogram( bgr, width, height, channel, numColorBlock, Res );	//img_resize
	if (nRet != 0)
	{	
		cout<<"Fail to ColorHistogramExtract!! "<<endl;
		return nRet;
	}

	return nRet;
}

int API_MAINCOLOR_1_0_0::ColorHistogram(
	uchar*										bgr,				//[In]:image->bgr
	int 										width,				//[In]:image->width
	int 										height, 			//[In]:image->height
	int 										channel,			//[In]:image->channel
	int 										numColorBlock,		//[In]:numColorBlock)
	vector< pair< vector< int >,float > >		&Res)				//[Out]:Res
{
	int nRet = 0;
	int i,j,k,count,tmpPixel;
	
	int blockPixel = int( 255.0/numColorBlock + 0.5 );
	int halfBlockPixel = int( blockPixel*0.5 + 0.5 );
	double size = 1.0/ (height * width) ;

	vector< int > vecColor;
	map< vector< int >,long > mapMainColor;
	map< vector< int >,long >::iterator itMainColor;
	vector< pair< vector< int >,float > > ResMainColor;
	
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) 
		{
			vecColor.clear();
			for(k=0;k<channel;k++)
			{
				tmpPixel = bgr[i*width*channel+j*channel+k];
				tmpPixel = int(tmpPixel*1.0/blockPixel)*blockPixel + halfBlockPixel;	//norm	
				vecColor.push_back( tmpPixel );	//[B-G-R],[gray]
			}
			
			itMainColor = mapMainColor.find(vecColor);
			if (itMainColor == mapMainColor.end()) // not find
			{
				mapMainColor[vecColor] = 1;
			}
			else
			{
				count = itMainColor->second; 
	            itMainColor->second = count+1; 
			}
		}
	}

	for(itMainColor = mapMainColor.begin(); itMainColor != mapMainColor.end(); itMainColor++)
	{
		ResMainColor.push_back( std::make_pair( itMainColor->first, itMainColor->second*size ) );
	}

	/*****************************Sort*****************************/
	//sort label result
	sort( ResMainColor.begin(), ResMainColor.end(), ImgSortComp );

	//get max score
	int resNum = (ResMainColor.size()>2)?2:ResMainColor.size();
	for( i=0;i<resNum;i++ )
	{
		Res.push_back( std::make_pair( ResMainColor[i].first, ResMainColor[i].second ) );
		//printf( "ImageID:%lld,i-%d,R-%d,G-%d,B-%d,score-%.8f\n",
		//	ImageID,i,Res[i].first[2],Res[i].first[1],Res[i].first[0],Res[i].second );
	}
	
	return nRet;
}

void API_MAINCOLOR_1_0_0::gaussianFilter(uchar* data, int width, int height, int channel)
{
    int i, j, k, m, n, index, sum;
    int templates[9] = { 1, 2, 1,
                         2, 4, 2,
                         1, 2, 1 };
    sum = height * width * channel * sizeof(uchar);
    uchar *tmpdata = (uchar*)malloc(sum);
    memcpy((int*)tmpdata,(int*)data, sum);

	for(k = 0;k < channel;k++)
    {
	    for(i = 1;i < height - 1;i++)
	    {
	        for(j = 1;j < width - 1;j++)
	        {
	        	sum = 0;
	            index = 0;		
	            for(m = i - 1;m < i + 2;m++)
	            {
	                for(n = j - 1; n < j + 2;n++)
	                {
	                    sum += tmpdata[m*width*channel+n*channel+k] * templates[index];
						index++;
	                }
	            }
	            data[i*width*channel+j*channel+k] = int(sum*1.0/16+0.5);
        	}
        }
    }
	
    free(tmpdata);
}




