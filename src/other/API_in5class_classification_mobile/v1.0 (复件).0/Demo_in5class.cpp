#include <string>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h> 
#include <sys/types.h> 
#include <iostream>
#include <unistd.h>
#include <dirent.h>
#include <fstream>

#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/legacy/legacy.hpp"

#include "API_commen.h"
#include "API_pca.h"
#include "API_in5class_classification.h"
#include "TErrorCode.h"

using namespace cv;
using namespace std;

int ExtractLocalFeat( char *szQueryList, char *svFeat )
{
	/*****************************Init*****************************/
	char loadImgPath[256];
	char szImgPath[256];
	int i, j, rWidth, rHeight, label, svImg, nRet = 0;
	long inputLabel, nCount;
	unsigned long long ImageID = 0;
	double allGetLabelTime,tGetLabelTime;
	FILE *fpListFile = 0;
	
	API_COMMEN api_commen;
	
	/***********************************Init*************************************/
	// detecting keypoints
	SurfFeatureDetector detector(400);
	//FastFeatureDetector detector(1,true);
	vector<KeyPoint> keypoints;
	
	// computing descriptors
	//Ptr<DescriptorExtractor > extractor(new SurfDescriptorExtractor());//  extractor;
	Ptr<DescriptorExtractor > extractor(
		new OpponentColorDescriptorExtractor(
			Ptr<DescriptorExtractor>(new SurfDescriptorExtractor())
			)
		);
	Mat training_descriptors(1,extractor->descriptorSize(),extractor->descriptorType());
	printf("detector->descriptorSize():%d,extractor->descriptorSize():%d\n",detector.descriptorSize(),extractor->descriptorSize());
	
	/********************************Open Query List*****************************/
	fpListFile = fopen(szQueryList,"r");
	if (!fpListFile) 
	{
		cout << "0.can't open " << szQueryList << endl;
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

		/***********************************Resize Image width && height*****************************/
		nRet = api_commen.GetReWH( ImgRemoveWhite->width, ImgRemoveWhite->height, 256, rWidth, rHeight );
		if (nRet != 0)
		{
		   	cout<<"Fail to GetReWH!! "<<endl;
			cvReleaseImage(&img);img = 0;
			cvReleaseImage(&ImgRemoveWhite);ImgRemoveWhite = 0;
			
		   	continue;
		}

		/*****************************Resize Img*****************************/
		IplImage *ImgResize = cvCreateImage(cvSize(rWidth, rHeight), img->depth, img->nChannels);
		cvResize( ImgRemoveWhite, ImgResize );
		
		/*****************************GetLabelFeat*****************************/	
		Mat matResizeImg( ImgResize );	
		Mat descriptors;
		tGetLabelTime = (double)getTickCount();
		detector.detect( matResizeImg, keypoints );
		if ( keypoints.size() == 0 )
		{	
			cout<<"keypoints:0!!" << loadImgPath << endl;

			/*********************************Release*************************************/
			cvReleaseImage(&img);img = 0;
			cvReleaseImage(&ImgRemoveWhite);ImgRemoveWhite = 0;
			cvReleaseImage(&ImgResize);ImgResize = 0;
			
			continue;
		}	
		
//		{
//			Mat out; //img_fg.copyTo(out);
//			drawKeypoints(img, keypoints, out, Scalar(255));
//			imshow("fg",img_fg);
//			imshow("keypoints", out);
//			waitKey(0);
//		}
		extractor->compute( matResizeImg, keypoints, descriptors );
		tGetLabelTime = (double)getTickCount() - tGetLabelTime;
		tGetLabelTime = tGetLabelTime*1000./cv::getTickFrequency();
		allGetLabelTime += tGetLabelTime;

		training_descriptors.push_back(descriptors);

		/*********************************Release*************************************/
		cvReleaseImage(&img);img = 0;
		cvReleaseImage(&ImgRemoveWhite);ImgRemoveWhite = 0;
		cvReleaseImage(&ImgResize);ImgResize = 0;
	}
	
	/*********************************close file*************************************/
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	

	cout << "Total descriptors: " << training_descriptors.rows << endl;
	/*********************************save training_descriptors*************************************/
	FileStorage fs( svFeat, FileStorage::WRITE);
	fs << "training_descriptors" << training_descriptors;
	fs.release();

	/*********************************Print Info*********************************/
	if ( nCount != 0 ) 
	{
		printf( "nCount:%ld,GetLabelTime:%.4fms\n", nCount,allGetLabelTime*1.0/nCount );
	}
	
	cout<<"Done!! "<<endl;
	
	return nRet;
}

int Build_vocabolary( char *svFeat, char *svVocabulary )
{	
	/*********************************load feat*************************************/
	FileStorage fs(svFeat, FileStorage::READ);
	Mat training_descriptors;
	fs["training_descriptors"] >> training_descriptors;
	fs.release();

	/********************************* BOW training*************************************/
	BOWKMeansTrainer bowtrainer(1000); //num clusters
	bowtrainer.add(training_descriptors);
	cout << "cluster BOW features" << endl;
	Mat vocabulary = bowtrainer.cluster();

	/*********************************save BOW vocabulary*************************************/
	FileStorage fs1(svVocabulary, FileStorage::WRITE);
	fs1 << "vocabulary" << vocabulary;
	fs1.release();

	cout<<"Done!! "<<endl;
	
	return 0;
}

int ExtractBOWFeat( char *szQueryList, char *szKeyFiles, char *svVocabulary, char *svBOWFeat ) 
{
	/*****************************Init*****************************/
	char loadImgPath[256];
	char szImgPath[256];
	int i, j, rWidth, rHeight, label, svImg, nRet = 0;
	long inputLabel, nCount;
	unsigned long long ImageID = 0;
	double allGetLabelTime,tGetLabelTime;
	FILE *fpListFile = 0 , *fpFeatOut = 0;
	
	API_COMMEN api_commen;
	API_IN5CLASS_CLASSIFICATION api_in5class;
	vector< vector<float> > imgFeat;

	/********************************Open Query List*****************************/
	fpListFile = fopen(szQueryList,"r");
	if (!fpListFile) 
	{
		cout << "0.can't open " << szQueryList << endl;
		return TEC_INVALID_PARAM;
	}

	fpFeatOut = fopen(svBOWFeat, "wt");
	if (!fpFeatOut)
	{
		cout << "Can't open result file " << svBOWFeat << endl;
		return TEC_INVALID_PARAM;
	}

	/********************************Init BOVW*****************************/
	nRet = api_in5class.Init( szKeyFiles, svVocabulary );
	if (nRet != 0)
	{
	   cout<<"Fail to Init!! "<<endl;
	   return TEC_INVALID_PARAM;
	}

	/*****************************Process one by one*****************************/
	while(EOF != fscanf(fpListFile, "%s %d", loadImgPath, &label))
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

		/************************Feat Extract*****************************/	
		imgFeat.clear();
		nRet = api_in5class.ExtractFeat( img, ImageID, imgFeat );
		if ( (nRet != 0) || (imgFeat.size()<1) )
		{
		   cout<<"Fail to ExtractFeat!! "<<endl;
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
	
	/*********************************close file*************************************/
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}
	if (fpFeatOut) {fclose(fpFeatOut);fpFeatOut = 0;}

	/*********************************Release*************************************/
	api_in5class.Release();

	cout<<"Done!! "<<endl;

	return 0;
}

int main(int argc, char* argv[])
{
	int  ret = 0;
	char szKeyFiles[256],szSavePath[256];
	API_COMMEN api_commen;
	
	if (argc == 4 && strcmp(argv[1],"-extractlocalfeat") == 0) {
		ret = ExtractLocalFeat( argv[2], argv[3] );
	}
	else if (argc == 4 && strcmp(argv[1],"-build_vocabolary") == 0) {
		ret = Build_vocabolary( argv[2], argv[3] );
	}
	else if (argc == 6 && strcmp(argv[1],"-extractbowfeat") == 0) {
		ret = ExtractBOWFeat( argv[2], argv[3], argv[4], argv[5] );
	}
	else
	{
		cout << "usage:\n" << endl;
		cout << "\tDemo_in5class -extractlocalfeat queryList.txt LocalFeat.yml\n" << endl;		
		cout << "\tDemo_in5class -build_vocabolary Feat.yml vocabolary.yml\n" << endl;	
		cout << "\tDemo_in5class -extractbowfeat queryList.txt KeyFilePath vocabolary.yml BOWFeat.Model\n" << endl;	
		return ret;
	}
	return ret;
}
