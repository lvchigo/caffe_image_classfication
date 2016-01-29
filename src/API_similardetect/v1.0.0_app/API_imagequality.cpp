#include <vector>
#include <math.h>
#include <iostream>
#include <string.h>

#include "API_imagequality.h"
#include "TErrorCode.h"

using namespace std;

/***********************************Init*************************************/
/// construct function 
API_IMAGEQUALITY::API_IMAGEQUALITY()
{
}

/// destruct function 
API_IMAGEQUALITY::~API_IMAGEQUALITY(void)
{
}

/***********************************ExtractFeat*************************************/
//blur 
//reference No-reference Image Quality Assessment using blur and noisy
//write by Min Goo Choi, Jung Hoon Jung  and so on 

int API_IMAGEQUALITY::ExtractFeat_Blur(
	unsigned char 				*pSrcImg, 
	int 						width, 
	int 						height,
	int 						nChannel,
	vector< float > 			&fBlur )		//45D=5*9D
{
	if( !pSrcImg || nChannel != 3 )  {
		printf("ExtractFeat_Blur err!!\n");
		return TEC_INVALID_PARAM;
	}
	
	int i,j,k,nRet=0;
	int Imgwidth = width;
	int Imgheight = height;
	int Bl_width = Imgwidth, Bl_height = Imgheight;

	unsigned char *pGrayImg = api_imageprocess.ImageRGB2Gray(pSrcImg, width, height, nChannel);

	vector< float > blockBlur;
	fBlur.clear();
	blockBlur.clear();
	//nRet = ExtractFeat_Blur_Block( pGrayImg, width, height, 1, blockBlur );
	nRet = ExtractFeat_Blur_Block_App( pGrayImg, width, height, 1, blockBlur );
	if (nRet != 0)
	{
	   	cout<<"ExtractFeat_Blur_Block err!!" << endl;
		if (pGrayImg) {delete [] pGrayImg;pGrayImg = NULL;}
		
		return TEC_INVALID_PARAM;
	}

	/**********************push data*************************************/
	for( k=0;k<blockBlur.size();k++ )
		fBlur.push_back( blockBlur[k] );
	
	/**********************cvReleaseImage*************************************/
	if (pGrayImg) {delete [] pGrayImg;pGrayImg = NULL;}

	return nRet;
}

//only for gray image
int API_IMAGEQUALITY::ExtractFeat_Blur_Block(
	unsigned char 				*pSrcImg, 
	int 						width, 
	int 						height,
	int 						nChannel,
	vector< float > 			&fBlur)		//5D
{
	if( (!pSrcImg) || (nChannel!=1) )
		return TEC_INVALID_PARAM;

	//for mean filter
	unsigned char *pNoisyData = api_imageprocess.ImageMedianFilter( pSrcImg, width, height, nChannel, 3 );
	//printf("ImageMedianFilter:0-%d,1-%d,2-%d\n",pNoisyData[0],pNoisyData[1],pNoisyData[2]);

	int nWid = width;
	int nHei = height; 
	int total = (nWid)*(nHei);
	int iLineBytes = nWid*nChannel;
	int iNoisyBytes = nWid*nChannel;

	int steps = 0;
	//result
	//blur
	double blur_mean = 0;
	double blur_ratio = 0;
	//noisy
	double nosiy_mean = 0;
	double nosiy_ratio = 0;

	//means DhMean and DvMean in paper
	//for edge
	// it is global mean in paper i will try local later
	double ghMean = 0; 
	double gvMean = 0;
	//for noisy
	double gNoisyhMean = 0;
	double gNoisyvMean = 0;
	//Nccand-mean
	double gNoisyMean = 0;

	//tmp color value for h v
	double ch = 0;	
	double cv = 0;
	//The Thresh blur value best detected
	const double blur_th = 0.1;
	//blur value sum
	double blurvalue = 0;
	//blur count
	int blur_cnt = 0;
	//edge count
	int h_edge_cnt = 0;
	int v_edge_cnt = 0;
	//noisy count
	int noisy_cnt = 0;
	// noisy value
	double noisy_value = 0;
	
	//mean Dh(x,y) in the paper 
	// in code it means Dh(x,y) and Ax(x,y)
	double* phEdgeMatric = new double[total];
	double* pvEdgeMatric = new double[total];
	// for noisy
	//Dh Dv in the paper
	double* phNoisyMatric = new double[total];
	double* pvNoisyMatric = new double[total];
	//Ncond in the paper
	double * NoisyM = new double[total];

	//means Ch(x,y) Cv(x,y) in the paper
	double* tmpH = new double[total];
	double* tmpV = new double[total];
	

	//for blur and noisy
	//loop 1
	for(int i = 0; i < nHei; i ++)
	{
		unsigned char* pOffset = pSrcImg;
		unsigned char* pNoisyOff = pNoisyData;
		steps = i*nWid;	

		for(int j = 0; j < nWid; j ++)
		{	
			int nSteps = steps + j;
			if(i == 0 || i == nHei -1)
			{
				//for edge
				phEdgeMatric[nSteps] = 0;
				pvEdgeMatric[nSteps] = 0;
				//for noisy
				phNoisyMatric[nSteps] = 0;
				pvNoisyMatric[nSteps] = 0;
			}
			else if(j == 0 || j == nWid -1)
			{
				//for edge
				phEdgeMatric[nSteps] = 0;
				pvEdgeMatric[nSteps] = 0;
				//for noisy
				phNoisyMatric[nSteps] = 0;
				pvNoisyMatric[nSteps] = 0;
			}
			else
			{
				//for edge
				ch = fabs(*(pOffset-1) - *(pOffset+1)) * 1.0 / 255.0;
				phEdgeMatric[nSteps] = ch;
				ghMean += ch;

				cv = fabs(*(pOffset-nWid) - *(pOffset+nWid)) * 1.0 / 255.0;
				pvEdgeMatric[nSteps] = cv;
				gvMean += cv;

				//for noisy
				ch = fabs(*(pNoisyOff-1) - *(pNoisyOff+1)) * 1.0 / 255.0;
				phNoisyMatric[nSteps] = ch;
				gNoisyhMean += ch;
				cv = fabs(*(pNoisyOff-nWid) - *(pNoisyOff+nWid)) * 1.0 / 255.0;
				pvNoisyMatric[nSteps] = cv;
				gNoisyvMean += cv;
			}
			
			double tmp_blur_value = 0;
			double tmp_ch = 0;
			double tmp_cv = 0;
			ch = (phEdgeMatric[nSteps] / 2);
			if(ch != 0)
				tmp_ch = fabs((*pOffset) * 1.0 / 255 - ch) * 1.0 / ch;	
			cv = (pvEdgeMatric[nSteps] / 2);
			if(cv != 0)
				tmp_cv = fabs((*pOffset) * 1.0 / 255 - cv) * 1.0 / cv;

			tmp_blur_value = max(tmp_ch,tmp_cv);
		//	blurvalue += tmp_blur_value;
			if(tmp_blur_value > blur_th) 
			{
				blur_cnt ++;
				blurvalue += tmp_blur_value;
			}

			pOffset ++;
			pNoisyOff ++;
		}
		pSrcImg += iLineBytes;
		pNoisyData += iNoisyBytes;
	}

	//for edge and noisy
	//for edge
	ghMean /= (total);
	gvMean /= (total);	
	//noisy
	gNoisyhMean /= total;
	gNoisyvMean /= total;

	//loop 2
	for(int i = 0; i < nHei; i ++)
	{
		steps = i*nWid;
		for(int j = 0; j < nWid; j ++)
		{
			int nSteps = steps + j;
			ch = phEdgeMatric[nSteps];
			tmpH[nSteps] = ch > ghMean ?  ch : 0;
			cv = pvEdgeMatric[nSteps];
			tmpV[nSteps] = cv > gvMean ?  cv : 0;

			ch = phNoisyMatric[nSteps];
			cv = pvNoisyMatric[nSteps];
			if(ch <= gNoisyhMean && cv <= gNoisyvMean)
			{
				NoisyM[nSteps] = max(ch,cv);
			}
			else
				NoisyM[nSteps] = 0;

			gNoisyMean += NoisyM[nSteps];
		}
	}
	gNoisyMean /= total;

	//loop 3
	for(int i = 0; i < nHei; i ++)
	{
		steps = i*(nWid);
		for(int j = 0; j < nWid; j ++)
		{
			int nSteps = steps + j;
			//for edge
			if(i == 0 || i == nHei -1)
			{
			//	phEdge[steps+j] = 0;
			//	pvEdge[steps+j] = 0;
			}
			else if(j == 0 || j == nWid -1)
			{
			//	phEdge[steps+j] = 0;
			//	pvEdge[steps+j] = 0;
			}
			else
			{
				//for edge
				if(tmpH[nSteps] > tmpH[nSteps-1] && tmpH[nSteps] > tmpH[nSteps+1])
				{
				//	phEdge[steps+j] = 1;
					h_edge_cnt ++;
				}
				//else phEdge[steps+j] = 0;

				if(tmpV[nSteps] > tmpV[steps-nWid] && tmpV[nSteps] > tmpV[steps+nWid])
				{
				//	pvEdge[steps+j] = 1;
					v_edge_cnt ++;
				}
			//	else pvEdge[steps+j] = 0;
				
				if(NoisyM[nSteps] > gNoisyMean)
				{
					noisy_cnt++;
					noisy_value += NoisyM[nSteps];
				}

			}
		
		}
	}

	if(phEdgeMatric){delete []phEdgeMatric; phEdgeMatric = NULL;}
	if(pvEdgeMatric){delete []pvEdgeMatric; pvEdgeMatric = NULL;}
	if(phNoisyMatric){delete []phNoisyMatric; phNoisyMatric = NULL;}
	if(pvNoisyMatric){delete []pvNoisyMatric; pvNoisyMatric = NULL;}
	if(NoisyM){delete []NoisyM; NoisyM = NULL;}
	if(tmpH){delete []tmpH; tmpH = NULL;}
	if(tmpV){delete []tmpV; tmpV = NULL;}

	if ( blur_cnt == 0 ) 
		blur_mean = 0;
	else
		blur_mean = blurvalue * 1.0 / blur_cnt;
	
	if ( (h_edge_cnt+v_edge_cnt) == 0 ) 
		blur_ratio = 0;
	else
		blur_ratio = blur_cnt * 1.0 / (h_edge_cnt+v_edge_cnt);
	
	if ( noisy_cnt == 0 ) 
		nosiy_mean = 0;
	else
		nosiy_mean = noisy_value * 1.0 / noisy_cnt;
	
	if ( total == 0 ) 
		nosiy_ratio = 0;
	else
		nosiy_ratio = noisy_cnt * 1.0 / total;

	//the para is provided by paper
	//another para 1.55 0.86 0.24 0.66
	double gReulst = 1 -( blur_mean + blur_ratio*0.95 + nosiy_mean*0.3 + nosiy_ratio*0.75 );

	fBlur.push_back( blur_mean );
	fBlur.push_back( blur_ratio );
	fBlur.push_back( nosiy_mean );
	fBlur.push_back( nosiy_ratio );
	fBlur.push_back( gReulst );

	return TOK;
}

//only for gray image
int API_IMAGEQUALITY::ExtractFeat_Blur_Block_App(
	unsigned char 				*pSrcImg, 
	int 						width, 
	int 						height,
	int 						nChannel,
	vector< float > 			&fBlur)		//5D
{
	if( (!pSrcImg) || (nChannel!=1) )
		return TEC_INVALID_PARAM;

	//for mean filter
	unsigned char *pNoisyData = api_imageprocess.ImageMedianFilter( pSrcImg, width, height, nChannel, 3 );
	//printf("ImageMedianFilter:0-%d,1-%d,2-%d\n",pNoisyData[0],pNoisyData[1],pNoisyData[2]);

	int nWid = width;
	int nHei = height; 
	int total = (nWid)*(nHei);
	int iLineBytes = nWid*nChannel;
	int iNoisyBytes = nWid*nChannel;

	int steps = 0;
	//result
	//blur
	double blur_mean = 0;
	double blur_ratio = 0;

	//means DhMean and DvMean in paper
	//for edge
	// it is global mean in paper i will try local later
	double ghMean = 0; 
	double gvMean = 0;

	//tmp color value for h v
	double ch = 0;	
	double cv = 0;
	//The Thresh blur value best detected
	const double blur_th = 0.1;
	//blur value sum
	double blurvalue = 0;
	//blur count
	int blur_cnt = 0;
	//edge count
	int h_edge_cnt = 0;
	int v_edge_cnt = 0;
	//noisy count
	int noisy_cnt = 0;
	// noisy value
	double noisy_value = 0;
	
	//mean Dh(x,y) in the paper 
	// in code it means Dh(x,y) and Ax(x,y)
	double* phEdgeMatric = new double[total];
	double* pvEdgeMatric = new double[total];

	//means Ch(x,y) Cv(x,y) in the paper
	double* tmpH = new double[total];
	double* tmpV = new double[total];
	

	//for blur and noisy
	//loop 1
	for(int i = 0; i < nHei; i ++)
	{
		unsigned char* pOffset = pSrcImg;
		steps = i*nWid;	

		for(int j = 0; j < nWid; j ++)
		{	
			int nSteps = steps + j;
			if(i == 0 || i == nHei -1)
			{
				//for edge
				phEdgeMatric[nSteps] = 0;
				pvEdgeMatric[nSteps] = 0;
			}
			else if(j == 0 || j == nWid -1)
			{
				//for edge
				phEdgeMatric[nSteps] = 0;
				pvEdgeMatric[nSteps] = 0;
			}
			else
			{
				//for edge
				ch = fabs(*(pOffset-1) - *(pOffset+1)) * 1.0 / 255.0;
				phEdgeMatric[nSteps] = ch;
				ghMean += ch;

				cv = fabs(*(pOffset-nWid) - *(pOffset+nWid)) * 1.0 / 255.0;
				pvEdgeMatric[nSteps] = cv;
				gvMean += cv;
			}
			
			double tmp_blur_value = 0;
			double tmp_ch = 0;
			double tmp_cv = 0;
			ch = (phEdgeMatric[nSteps] / 2);
			if(ch != 0)
				tmp_ch = fabs((*pOffset) * 1.0 / 255 - ch) * 1.0 / ch;	
			cv = (pvEdgeMatric[nSteps] / 2);
			if(cv != 0)
				tmp_cv = fabs((*pOffset) * 1.0 / 255 - cv) * 1.0 / cv;

			tmp_blur_value = max(tmp_ch,tmp_cv);
		//	blurvalue += tmp_blur_value;
			if(tmp_blur_value > blur_th) 
			{
				blur_cnt ++;
				blurvalue += tmp_blur_value;
			}

			pOffset ++;
		}
		pSrcImg += iLineBytes;
		pNoisyData += iNoisyBytes;
	}

	//for edge and noisy
	//for edge
	ghMean /= (total);
	gvMean /= (total);	

	//loop 2
	for(int i = 0; i < nHei; i ++)
	{
		steps = i*nWid;
		for(int j = 0; j < nWid; j ++)
		{
			int nSteps = steps + j;
			ch = phEdgeMatric[nSteps];
			tmpH[nSteps] = ch > ghMean ?  ch : 0;
			cv = pvEdgeMatric[nSteps];
			tmpV[nSteps] = cv > gvMean ?  cv : 0;
		}
	}

	//loop 3
	for(int i = 0; i < nHei; i ++)
	{
		steps = i*(nWid);
		for(int j = 0; j < nWid; j ++)
		{
			int nSteps = steps + j;
			//for edge
			if(i == 0 || i == nHei -1)
			{
			//	phEdge[steps+j] = 0;
			//	pvEdge[steps+j] = 0;
			}
			else if(j == 0 || j == nWid -1)
			{
			//	phEdge[steps+j] = 0;
			//	pvEdge[steps+j] = 0;
			}
			else
			{
				//for edge
				if(tmpH[nSteps] > tmpH[nSteps-1] && tmpH[nSteps] > tmpH[nSteps+1])
				{
				//	phEdge[steps+j] = 1;
					h_edge_cnt ++;
				}
				//else phEdge[steps+j] = 0;

				if(tmpV[nSteps] > tmpV[steps-nWid] && tmpV[nSteps] > tmpV[steps+nWid])
				{
				//	pvEdge[steps+j] = 1;
					v_edge_cnt ++;
				}
			//	else pvEdge[steps+j] = 0;
			}
		
		}
	}

	if(phEdgeMatric){delete []phEdgeMatric; phEdgeMatric = NULL;}
	if(pvEdgeMatric){delete []pvEdgeMatric; pvEdgeMatric = NULL;}
	if(tmpH){delete []tmpH; tmpH = NULL;}
	if(tmpV){delete []tmpV; tmpV = NULL;}

	if ( blur_cnt == 0 ) 
		blur_mean = 0;
	else
		blur_mean = blurvalue * 1.0 / blur_cnt;
	
	if ( (h_edge_cnt+v_edge_cnt) == 0 ) 
		blur_ratio = 0;
	else
		blur_ratio = blur_cnt * 1.0 / (h_edge_cnt+v_edge_cnt);

	//the para is provided by paper
	//another para 1.55 0.86 0.24 0.66
	double gReulst = 1 -( blur_mean + blur_ratio*0.95 );

	fBlur.push_back( blur_mean );
	fBlur.push_back( blur_ratio );
	fBlur.push_back( gReulst );

	return TOK;
}

int API_IMAGEQUALITY::ExtractFeat_Blur_test(
	unsigned char 				*pSrcImg, 
	int 						width, 
	int 						height,
	int 						nChannel,
	float			 			&fBlur )
{
	if( !pSrcImg || nChannel != 3 )  {
		printf("ExtractFeat_Blur err!!\n");
		return TEC_INVALID_PARAM;
	}
	
	int i,j,k,nRet=0;
	float Entropy,tmp = 0;

	unsigned char *pGrayImg = api_imageprocess.ImageRGB2Gray(pSrcImg, width, height, nChannel);
	//unsigned char *pFilterImg = api_imageprocess.ImageMedianFilter( pGrayImg, width, height, 1, 3 );

	Entropy = 0;
	for (i=1; i<(height-1); ++i)
    {
        for (j=1; j<(width-1); ++j)
        {
        	tmp = 0;
			//tmp = 2*fabs(pGrayImg[i*(width-2)+j]-pFilterImg[i*(width-2)+j]);
			tmp += fabs(pGrayImg[i*(width-2)+(j+1)]-pGrayImg[i*(width-2)+(j-1)]);
			tmp += fabs(pGrayImg[(i+1)*(width-2)+j]-pGrayImg[(i-1)*(width-2)+j]);
			//tmp += fabs(pFilterImg[i*(width-2)+(j+1)]-pFilterImg[i*(width-2)+(j-1)]);
			//tmp += fabs(pFilterImg[(i+1)*(width-2)+j]-pFilterImg[(i-1)*(width-2)+j]);
			tmp = (tmp<0)?0:tmp;
			tmp = (tmp>255)?255:tmp;

			if(tmp>0)
            	Entropy = Entropy-(tmp*1.0/255)*(log(tmp*1.0/255)/log(2.0));
		}
	}

	fBlur = Entropy*127.0/20000;
	fBlur = (fBlur<0)?0:fBlur;
	fBlur = (fBlur>127)?127:fBlur;
	
	/**********************cvReleaseImage*************************************/
	if (pGrayImg) {delete [] pGrayImg;pGrayImg = NULL;}
	//if (pFilterImg) {delete [] pFilterImg;pFilterImg = NULL;}

	return nRet;
}




