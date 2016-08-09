#pragma once
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

#include "API_pca.h"
#include "TErrorCode.h"

using namespace cv;
using namespace std;


/***********************************Init*************************************/
/// construct function 
API_PCA::API_PCA()
{
}

/// destruct function 
API_PCA::~API_PCA(void)
{
}

void API_PCA::PCA_Feat_Init( const char *PCAModel )
{
	pca_encoding = new PCA();
	FileStorage fs_r( PCAModel, FileStorage::READ );
	fs_r[PCA_MEAN] >> pca_encoding->mean;
	fs_r[PCA_EIGEN_VECTOR] >> pca_encoding->eigenvectors;
	fs_r.release();
}

void API_PCA::PCA_Feat_Release(  )
{
	delete pca_encoding;
}


int API_PCA::PCA_Feat_Learn( vector< pair< int, vector<float> > > inFeat, const char *outPCAModel )
{
	/********************************Check inFeat*****************************/
	int i,j,featDim,index,nRet = 0;
	float sum=0, sum0=0, ratio, T_ratio = PCA_RATIO;
	for( i=0;i<inFeat.size();i++ )
	{
		if (i == 0)
			featDim = inFeat[i].second.size();
		else
		{
			nRet = (featDim == inFeat[i].second.size())?TOK:TEC_INVALID_PARAM;
		}

		if ( (featDim==0) || (nRet != TOK) )
		{
			printf("PCA_Feat_Learn input data err!!\n");
			return TEC_INVALID_PARAM;
		}
	}

	/********************************load samples*****************************/
	Mat SampleSet(inFeat.size(), featDim, CV_32FC1);
	for (i=0; i<inFeat.size(); ++i)
	{
		for (j=0; j<featDim; ++j)
		{
			SampleSet.at<float>(i, j) = inFeat[i].second[j];
		}
	}
	
	/********************************Training*****************************/
	PCA *pca = new PCA(SampleSet, Mat(), CV_PCA_DATA_AS_ROW);///////////////
	//cout << "eigenvalues:" <<endl << pca->eigenvalues <<endl<<endl;

	/**********************calculate the decreased dimensions****************/
	for (i=0; i<pca->eigenvalues.rows; ++i)
	{
		sum += pca->eigenvalues.at<float>(i,0);
	}
	int bPrintf = 1;
	for (i=0; i<pca->eigenvalues.rows; ++i)
	{
		sum0 += pca->eigenvalues.at<float>(i,0);
		ratio = sum0/sum;
		if( (ratio > T_ratio) || ( (i+1)==PCA_DIMS ) ){
			index = i;
			printf("learning:index:%d,ratio:%.4f\n",(i+1),ratio);
			
			break;
		}
		
		if ( (i+1)%100 == 0)
			printf("learning:index:%d,ratio:%.4f\n",(i+1),ratio);
		
	}

	/**********************Save PCA Model****************/
	Mat eigenvetors_d( (index+1), featDim, CV_32FC1 );//eigen values of decreased dimension
	for (int i=0; i<(index+1); ++i)
	{
		pca->eigenvectors.row(i).copyTo(eigenvetors_d.row(i));
	}
	//cout << "eigenvectors" <<endl << eigenvetors_d << endl;
	FileStorage fs_w( outPCAModel, FileStorage::WRITE);//write mean and eigenvalues into xml file
	fs_w << PCA_MEAN << pca->mean;
	fs_w << PCA_EIGEN_VECTOR << eigenvetors_d;
	fs_w.release();
	//Encoding

	delete pca;
	//printf("inFeat Dim:%d,Output Dim:%d\n",featDim,(index+1));

	return nRet;
}

int API_PCA::PCA_Feat_Encode( vector<float> inFeat, vector<float> &EncodeFeat )
{
	int i,j,featDim,index,nRet = 0;
	EncodeFeat.clear();
	
	Mat mInFeat(1,inFeat.size(), CV_32FC1);	//Test input
	for (j=0; j<inFeat.size(); ++j)
	{
		mInFeat.at<float>(0, j) = inFeat[j];
	}
	
	Mat mEncodeFeat(1, pca_encoding->eigenvectors.rows, CV_32FC1);
	pca_encoding->project(mInFeat, mEncodeFeat);
	//cout << endl << "pca_encode:" << endl << mEncodeFeat;

	for (j=0; j<pca_encoding->eigenvectors.rows; ++j)
	{
		EncodeFeat.push_back( mEncodeFeat.at<float>(0, j) );
	}

	//printf("inFeat Dim:%d,EncodeFeat Dim:%d\n",inFeat.size(),pca_encoding->eigenvectors.rows);

	return nRet;
}

int API_PCA::PCA_Feat_Decode( vector<float> EncodeFeat, const int DecodeDim, vector<float> &DecodeFeat )
{
	int j,nRet = 0;
	DecodeFeat.clear();

	Mat mEncodeFeat(1,EncodeFeat.size(), CV_32FC1);	//Test input
	for (j=0; j<EncodeFeat.size(); ++j)
	{
		mEncodeFeat.at<float>(0, j) = EncodeFeat[j];
	}

	//Decoding
	Mat mDecodeFeat(1, DecodeDim, CV_32FC1);
	pca_encoding->backProject(mEncodeFeat,mDecodeFeat);
	//cout <<endl<< "pca_Decode:" << endl << mDecodeFeat;

	for (j=0; j<DecodeDim; ++j)
	{
		DecodeFeat.push_back( mDecodeFeat.at<float>(0, j) );
	}

	//printf("EncodeFeat Dim:%d,DecodeFeat Dim:%d\n",EncodeFeat.size(),DecodeFeat.size());
	
	return nRet;
}

