#define _MAIN

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
#include "TErrorCode.h"

#include <opencv/highgui.h>

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

#include "region.h"
#include "agglomerative_clustering.h"
#include "utils.h"

using namespace std;
using namespace cv;

/* Diversivication Configurations :                                     */
/* These are boolean values, indicating whenever to use a particular    */
/*                                   diversification strategy or not    */

#define PYRAMIDS     1 // Use spatial pyramids
#define CUE_D        1 // Use Diameter grouping cue
#define CUE_FGI      1 // Use ForeGround Intensity grouping cue
#define CUE_BGI      1 // Use BackGround Intensity grouping cue
#define CUE_G        1 // Use Gradient magnitude grouping cue
#define CUE_S        1 // Use Stroke width grouping cue
#define CHANNEL_I    1 // Use Intensity color channel
#define CHANNEL_R    1 // Use Red color channel
#define CHANNEL_G    1 // Use Green color channel
#define CHANNEL_B    1 // Use Blue color channel


int ads_textdetect( char *szQueryList, int inputDelta, char *svPath  )
{
	
	
	/*****************************Init*****************************/
	char loadImgPath[256];
	char szImgPath[256];
	int i, j, svImg, tank, nRet = 0;
	long inputLabel, nCount;
	unsigned long long ImageID = 0;
	FILE *fpListFile = 0 ;

	API_COMMEN api_commen;
	string keyfile = "/home/chigo/working/caffe_img_classification/test/src/API_adstextdetect/trained_boost_groups.xml";

	// Pipeline configuration
    bool conf_channels[4]={CHANNEL_R,CHANNEL_G,CHANNEL_B,CHANNEL_I};
    bool conf_cues[5]={CUE_D,CUE_FGI,CUE_BGI,CUE_G,CUE_S};

    /* initialize random seed: */
    srand (time(NULL));

    Mat src, grey, lab_img, gradient_magnitude;

	int rWidth,rHeight;
	int MaxLen = 320;	//maxlen:720-same with online
	char imgPath[1024] = {0};

	CvRect tmpRect;
	double tmpProbability;
	map< CvRect ,double > mapRectScore;
	map< CvRect ,double >::iterator itRectScore;
	
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
		mapRectScore.clear();
		api_commen.getRandomID( ImageID );

/*		if( (img->width>size) || (img->height>size) ) 
		{	
			sprintf(szImgPath, "%s/%ld.jpg", svPath, ImageID);
			cvSaveImage( szImgPath,img );
		}	*/

		if(!img || (img->width<16) || (img->height<16) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
		{	
			cout<<"image err!!" << endl;
			return TEC_INVALID_PARAM;
		}	

		/***********************************Resize Image width && height*****************************/
		IplImage *imgResize;
		if( ( img->width>MaxLen ) || ( img->height>MaxLen ) )
		{
			nRet = api_commen.GetReWH( img->width, img->height, MaxLen, rWidth, rHeight );	
			if (nRet != 0)
			{
			   	cout<<"GetReWH err!!" << endl;
				return TEC_INVALID_PARAM;
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
			
		/*****************************push data*****************************/
		Mat imgMat( imgResize );
	    //imgMat.copyTo(src);

	    int delta = inputDelta;
	    int img_area = imgMat.cols*imgMat.rows;
	    cv::MSER cv_mser(delta,(int)(0.00002*img_area),(int)(0.11*img_area),55,0.);

	    cvtColor(imgMat, grey, CV_BGR2GRAY);
	    cvtColor(imgMat, lab_img, CV_BGR2Lab);
	    gradient_magnitude = Mat_<double>(imgMat.size());
	    get_gradient_magnitude( grey, gradient_magnitude);

	    vector<Mat> channels;
	    split(imgMat, channels);
	    channels.push_back(grey);
	    int num_channels = channels.size();

	    if (PYRAMIDS)
	    {
	      for (int c=0; c<num_channels; c++)
	      {
	        Mat pyr;
	        resize(channels[c],pyr,Size(channels[c].cols/2,channels[c].rows/2));
	        //resize(pyr,pyr,Size(channels[c].cols,channels[c].rows));
	        channels.push_back(pyr);
	        resize(channels[c],pyr,Size(channels[c].cols/4,channels[c].rows/4));
	        //resize(pyr,pyr,Size(channels[c].cols,channels[c].rows));
	        channels.push_back(pyr);
	      }
	    }

	    for (int c=0; c<channels.size(); c++)
	    {

	        if (!conf_channels[c%4]) continue;

	        if (channels[c].size() != grey.size()) // update sizes for smaller pyramid lvls
	        {
	          resize(grey,grey,Size(channels[c].cols,channels[c].rows));
	          resize(lab_img,lab_img,Size(channels[c].cols,channels[c].rows));
	          resize(gradient_magnitude,gradient_magnitude,Size(channels[c].cols,channels[c].rows));
	        }

	        /* Initial over-segmentation using MSER algorithm */
	        vector<vector<Point> > contours;
	        //t = (double)getTickCount();
	        cv_mser(channels[c], contours);
	        //cout << " OpenCV MSER found " << contours.size() << " regions in " << ((double)getTickCount() - t)*1000/getTickFrequency() << " ms." << endl;
	   

	        /* Extract simple features for each region */ 
	        vector<Region> regions;
	        Mat mask = Mat::zeros(grey.size(), CV_8UC1);
	        double max_stroke = 0;
	        for (int i=contours.size()-1; i>=0; i--)
	        {
	            Region region;
	            region.pixels_.push_back(Point(0,0)); //cannot swap an empty vector
	            region.pixels_.swap(contours[i]);
	            region.extract_features(lab_img, grey, gradient_magnitude, mask, conf_cues);
	            max_stroke = max(max_stroke, region.stroke_mean_);
	            regions.push_back(region);
	        }

	        /* Single Linkage Clustering for each individual cue */
	        for (int cue=0; cue<5; cue++)
	        {

	          if (!conf_cues[cue]) continue;
	    
	          int f=0;
	          unsigned int N = regions.size();
	          if (N<3) continue;
	          int dim = 3;
	          t_float *data = (t_float*)malloc(dim*N * sizeof(t_float));
	          int count = 0;
	          for (int i=0; i<regions.size(); i++)
	          {
	            data[count] = (t_float)(regions.at(i).bbox_.x+regions.at(i).bbox_.width/2)/imgMat.cols*0.25;
	            data[count+1] = (t_float)(regions.at(i).bbox_.y+regions.at(i).bbox_.height/2)/imgMat.rows;
	            switch(cue)
	            {
	              case 0:
	                data[count+2] = (t_float)max(regions.at(i).bbox_.height, regions.at(i).bbox_.width)/max(imgMat.rows,imgMat.cols);
	                break;
	              case 1:
	                data[count+2] = (t_float)regions.at(i).intensity_mean_/255;
	                break;
	              case 2:
	                data[count+2] = (t_float)regions.at(i).boundary_intensity_mean_/255;
	                break;
	              case 3:
	                data[count+2] = (t_float)regions.at(i).gradient_mean_/255;
	                break;
	              case 4:
	                data[count+2] = (t_float)regions.at(i).stroke_mean_/max_stroke;
	                break;
	            }
	            count = count+dim;
	          }

	          HierarchicalClustering h_clustering(regions, keyfile);
	          vector<HCluster> dendrogram;
	          h_clustering(data, N, dim, (unsigned char)0, (unsigned char)3, dendrogram);
	      
	          for (int k=0; k<dendrogram.size(); k++)
	          {
	             int ml = 1;
	             if (c>=num_channels) ml=2;// update sizes for smaller pyramid lvls
	             if (c>=2*num_channels) ml=4;// update sizes for smaller pyramid lvls

	             //cout << dendrogram[k].rect.x*ml << " " << dendrogram[k].rect.y*ml << " "
	             //     << dendrogram[k].rect.width*ml << " " << dendrogram[k].rect.height*ml << " "
	             //     << (float)dendrogram[k].probability*-1 << endl;
	             //     << (float)dendrogram[k].nfa << endl;
	             //     << (float)(k) * ((float)rand()/RAND_MAX) << endl;
	             //     << (float)dendrogram[k].nfa * ((float)rand()/RAND_MAX) << endl;

				 tmpRect = cvRect(dendrogram[k].rect.x*ml, dendrogram[k].rect.y*ml, 
				 				dendrogram[k].rect.width*ml, dendrogram[k].rect.height*ml);
				 tmpProbability = (double)dendrogram[k].probability;

				//find
				itRectScore = mapRectScore.find(tmpRect);
				if (itRectScore == mapRectScore.end()) // not find
				{
					mapRectScore[tmpRect] = tmpProbability;
				}
				else
				{
					if ( itRectScore->second < tmpProbability )
					{
						mapRectScore[tmpRect] = tmpProbability;
					}
				}
	          }
	  
	          free(data);
	        }

	    }

		/************************Save GetFeat*****************************/	
		for(itRectScore = mapRectScore.begin(); itRectScore != mapRectScore.end(); itRectScore++)
		{
			printf("x-%d,y-%d,w-%d,h-%d,score:%.6f\n",itRectScore->first.x,itRectScore->first.y,
						itRectScore->first.width,itRectScore->first.height,itRectScore->second);
		}
		
		/*********************************Release*************************************/
		cvReleaseImage(&img);img = 0;
		cvReleaseImage(&imgResize);imgResize = 0;
	}

	/*********************************close file*************************************/
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	
	
	cout<<"Done!! "<<endl;
	
	return nRet;
}

int main(int argc, char* argv[])
{
	int  ret = 0;

	if (argc == 5 && strcmp(argv[1],"-adstextdetect") == 0)
	{
		ads_textdetect(argv[2], atoi(argv[3]), argv[4] );
	}
	else
	{
		cout << "usage:\n" << endl;
		cout << "\tDemo_adstextdetect -adstextdetect queryList.txt delta svPath\n" << endl;
		return ret;
	}
	return ret;
}

