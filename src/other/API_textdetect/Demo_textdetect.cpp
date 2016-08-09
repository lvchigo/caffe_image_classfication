//
//  main.cpp
//  RobustTextDetection
//
//  Created by Saburo Okita on 05/06/14.
//  Copyright (c) 2014 Saburo Okita. All rights reserved.
//

#include <string>
#include <vector>
#include <stdio.h>
#include <stdlib.h>		//do shell
#include <sys/stat.h> 
#include <sys/types.h> 
#include <iostream>
#include <unistd.h>
#include <map>

#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>

#include "API_commen.h"
#include "TErrorCode.h"
#include "plog/Log.h"


#include <opencv2/opencv.hpp>
//#include <tesseract/baseapi.h>

#include "RobustTextDetection.h"
#include "ConnectedComponent.h"

using namespace std;
using namespace cv;


int test()
{

//    namedWindow( "" );
//    moveWindow("", 0, 0);
    
    Mat image = imread( "/home/chigo/working/caffe_img_classification/test/bin/textdetect_sh/data/TestText.png" );
    
    /* Quite a handful or params */
    RobustTextParam param;
    param.minMSERArea        = 10;
    param.maxMSERArea        = 2000;
    param.cannyThresh1       = 20;
    param.cannyThresh2       = 100;
    
    param.maxConnCompCount   = 3000;
    param.minConnCompArea    = 75;
    param.maxConnCompArea    = 600;
    
    param.minEccentricity    = 0.1;
    param.maxEccentricity    = 0.995;
    param.minSolidity        = 0.4;
    param.maxStdDevMeanRatio = 0.5;
    
    /* Apply Robust Text Detection */
    /* ... remove this temp output path if you don't want it to write temp image files */
    string temp_output_path = "/home/chigo/working/caffe_img_classification/test/bin/textdetect_sh/data/output/";
    RobustTextDetection detector(param, temp_output_path );
    pair<Mat, Rect> result = detector.apply( image );
    
    /* Get the region where the candidate text is */
    Mat stroke_width( result.second.height, result.second.width, CV_8UC1, Scalar(0) );
    Mat(result.first, result.second).copyTo( stroke_width);
    
    
/*  //Use Tesseract to try to decipher our image
    tesseract::TessBaseAPI tesseract_api;
    tesseract_api.Init(NULL, "eng"  );
    tesseract_api.SetImage((uchar*) stroke_width.data, stroke_width.cols, stroke_width.rows, 1, stroke_width.cols);
    
    string out = string(tesseract_api.GetUTF8Text());

    //Split the string by whitespace
    vector<string> splitted;
    istringstream iss( out );
    copy( istream_iterator<string>(iss), istream_iterator<string>(), back_inserter( splitted ) );
    
    //And draw them on screen
    CvFont font = cvFontQt("Helvetica", 24.0, CV_RGB(0, 0, 0) );
    Point coord = Point( result.second.br().x + 10, result.second.tl().y );
    for( string& line: splitted ) {
        addText( image, line, coord, font );
        coord.y += 25;
    }
    
    rectangle( image, result.second, Scalar(0, 0, 255), 2);
    
    //Append the original and stroke width images together
    cvtColor( stroke_width, stroke_width, CV_GRAY2BGR );
    Mat appended( image.rows, image.cols + stroke_width.cols, CV_8UC3 );
    image.copyTo( Mat(appended, Rect(0, 0, image.cols, image.rows)) );
    stroke_width.copyTo( Mat(appended, Rect(image.cols, 0, stroke_width.cols, stroke_width.rows)) );
    
    imshow("", appended );
    waitKey();
*/
    
    return 0;
}

int frcnn_test( char *szQueryList, char* KeyFilePath )
{
	char tPath[256];
	char loadImgPath[256];
	char szImgPath[256];
	char savePath[256];
	vector<string> vecText;
	int i, j, label, svImg, nRet = 0;
	long inputLabel, nCount, nCountFace;
	string strImageID,text,name;
	double allPredictTime;
	FILE *fpListFile = 0;

	RunTimer<double> run;
	API_COMMEN api_commen;
	API_FACE_ANNOATION api_face_annoation;

	vector< FaceAnnoationInfo > Res;

	/***********************************Init**********************************/
	RobustTextParam param;
    param.minMSERArea        = 10;
    param.maxMSERArea        = 2000;
    param.cannyThresh1       = 20;
    param.cannyThresh2       = 100;
    
    param.maxConnCompCount   = 3000;
    param.minConnCompArea    = 75;
    param.maxConnCompArea    = 600;
    
    param.minEccentricity    = 0.1;
    param.maxEccentricity    = 0.995;
    param.minSolidity        = 0.4;
    param.maxStdDevMeanRatio = 0.5;

	/***********************************Init**********************************/
	plog::init(plog::info, "plog.txt"); 

	const static Scalar colors[] =  { 	CV_RGB(0,0,255),	CV_RGB(0,255,255),	CV_RGB(0,255,0),	
										CV_RGB(255,255,0),	CV_RGB(255,0,0),	CV_RGB(255,0,255),
										CV_RGB(0,128,255), 	CV_RGB(255,128,0)} ;

	CvFont font;
 	cvInitFont(&font,CV_FONT_HERSHEY_TRIPLEX,0.35f,0.7f,0,1,CV_AA);
	
	/********************************Open Query List*****************************/
	fpListFile = fopen(szQueryList,"r");
	if (!fpListFile) 
	{
		cout << "0.can't open " << szQueryList << endl;
		return TEC_INVALID_PARAM;
	}

	nCount = 0;
	nCountFace = 0;
	allPredictTime = 0.0;
	/*****************************Process one by one*****************************/
	while(EOF != fscanf(fpListFile, "%s", loadImgPath))
	{
		IplImage *img = cvLoadImage(loadImgPath);
		if(!img || (img->width<32) || (img->height<32) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
		{	
			cout<<"Can't open " << loadImgPath << endl;
			cvReleaseImage(&img);img = 0;
			continue;
		}	
		//printf("loadImgPath:%s\n",loadImgPath);

		/************************getRandomID*****************************/
		strImageID = api_commen.GetStringIDFromFilePath( loadImgPath );
		
		/************************Predict*****************************/	
		Res.clear();
		run.start();

		Mat image(img);
		/* Apply Robust Text Detection */
	    /* ... remove this temp output path if you don't want it to write temp image files */
	    string temp_output_path = "/home/chigo/working/caffe_img_classification/test/bin/textdetect_sh/data/output/";
	    RobustTextDetection detector(param, temp_output_path );
	    pair<Mat, Rect> result = detector.apply( image );
		run.end();
		//LOOGI<<"[Predict] time:"<<run.time();
		allPredictTime += run.time();
		
		/************************save img data*****************************/
		{
			name = Res[0].label;
			if (name=="face")
				sprintf(tPath, "res_predict/face/", tPath );
			else
				sprintf(tPath, "res_predict/noface/", tPath );
		}
		for(i=0;i<Res.size();i++)  
		{						
			Scalar color = colors[i%8];
			
			cvRectangle( img, cvPoint(Res[i].rect[0], Res[i].rect[1]),
	                   cvPoint(Res[i].rect[2], Res[i].rect[3]), color, 2, 8, 0);

			sprintf(szImgPath, "%.2f %.2f %s", Res[i].score, Res[i].angle, Res[i].label.c_str() );
			text = szImgPath;
			cvPutText( img, text.c_str(), cvPoint(Res[i].rect[0]+1, Res[i].rect[1]+20), &font, color );

			for (j=0;j<5;j++)
				cvCircle( img, cvPoint(Res[i].annoation[2*j], Res[i].annoation[2*j+1]), 2, colors[j%8], 2, 8, 0 );

			sprintf(tPath, "%s%s-%.2f_", tPath, Res[i].label.c_str(), Res[i].score );

			if ( Res[i].label == "face" )
				nCountFace++;
		}
		sprintf( savePath, "%s%s.jpg", tPath, strImageID.c_str() );
		cvSaveImage( savePath, img );
			
		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld img...,time:%.4f\n",nCount,run.time());
		
		cvReleaseImage(&img);img = 0;
	}

	/*********************************close file*************************************/
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	

	/*********************************Release*************************************/
	api_face_annoation.Release();

	/*********************************Print Info*********************************/
	if ( nCount != 0 ) 
	{
		printf( "nCount:%ld,nCountFace:%ld_%.4f,PredictTime:%.4fms\n", 
			nCount, nCountFace, nCountFace*1.0/nCount, allPredictTime*1000.0/nCount );
	}
	
	cout<<"Done!! "<<endl;
	
	return nRet;	
}


int main(int argc, char* argv[])
{
	int  ret = 0;
	char szKeyFiles[256],szSavePath[256];
	API_COMMEN api_commen;

	if (argc == 2 && strcmp(argv[1],"-test") == 0) {
		ret = test();
	}
	else
	{
		cout << "usage:\n" << endl;
		cout << "\tDemo_textdetect -test\n" << endl;
		return ret;
	}
	return ret;
}

