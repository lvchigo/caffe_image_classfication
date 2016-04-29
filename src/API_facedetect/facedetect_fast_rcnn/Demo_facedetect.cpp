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

#include "API_xml.h"	//read xml
#include "API_facedetect.h"

//BING
//#include "kyheader.h"
//#include "Objectness_predict.h"
//#include "ValStructVec.h"
//#include "CmShow.h"
//BINGpp
//#include "stdafx.h"
//#include "Objectness_predict.h"
//#include "ValStructVec.h"

using namespace cv;
using namespace std;

int Get_Img2Xml( char *szQueryList, char* xmlSavePath, char* imgSavePath, char* labelname )
{
	char tPath[256];
	char loadImgPath[256];
	char szImgPath[256];
	string strImageID;
	int i, j, label, svImg, nRet = 0;
	unsigned long long nCount,ImageID = 0;
	FILE *fpListFile = 0;

	/********************************Init*****************************/
	API_XML api_xml;
	API_COMMEN api_commen;
	vector< pair< string, Vec4i > > vecLabelRect;
	
	/********************************Open Query List*****************************/
	fpListFile = fopen(szQueryList,"r");
	if (!fpListFile) 
	{
		cout << "0.can't open " << szQueryList << endl;
		return TEC_INVALID_PARAM;
	}

	FILE *fpListFile_trainval = fopen("trainval.txt","at+");
	if (!fpListFile_trainval) 
	{
		cout << "0.can't open " << "trainval.txt" << endl;
		return TEC_INVALID_PARAM;
	}

	/*****************************Process one by one*****************************/
	nCount = 0;
	while(EOF != fscanf(fpListFile, "%s", &loadImgPath ))
	{
		IplImage *img = cvLoadImage(loadImgPath);
		if(!img || (img->width<128) || (img->height<128) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
		{	
			cout<<"Can't open " << loadImgPath << endl;
			continue;
		}	
		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld img...\n",nCount);

		strImageID = api_commen.GetStringIDFromFilePath( loadImgPath );

		//write_xml
		Vec4i rect;
		rect[0] = 1;
		rect[1] = 1;
		rect[2] = img->width-1;
		rect[3] = img->height-1;
		
		vecLabelRect.clear();
		vecLabelRect.push_back( std::make_pair( labelname, rect ) );
		nRet = api_xml.write_xml( strImageID, xmlSavePath, vecLabelRect );
		if (nRet!=0)
		{
			cout << "Err to write_xml!" << endl;
			continue;
		}

		//write trainval.txt
		fprintf(fpListFile_trainval, "%s\n", strImageID.c_str() );

		//cp img file
		//sprintf( tPath, "cp %s %s/%s.jpg", loadImgPath, imgSavePath, strImageID.c_str() );
		//system( tPath );

		cvReleaseImage(&img);img = 0;
	}
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	
	if (fpListFile_trainval) {fclose(fpListFile_trainval);fpListFile_trainval = 0;}	

	sleep(1);
	cout<<"Done!! "<<endl;
	
	return nRet;
}	

int Get_CelebA_FaceLabel( char* imgPath, char* loadAnnotations, char* xmlSavePath, char* imgSavePath )
{
	char tPath[256];
	char loadImgPath[256];
	char szImgPath[256];
	char loadIID[256];
	int i, j, svImg, nRet = 0;
	unsigned long long nCount, labelCount, ImageID = 0;
	FILE *fpListFile = 0;

	/********************************Init*****************************/
	string strLabel;
	string filename,tmpPath,iid,strImageID;
	int loadPoint[10] = {0};
	vector< Vec4i > FacePoint;

	API_XML api_xml;
	API_COMMEN api_commen;
	vector< pair< string, Vec4i > > 	vecLabelRect;

	Scalar color;
	const static Scalar colors[] =  { 	
		CV_RGB(0,0,255),	CV_RGB(0,255,255),	CV_RGB(0,255,0),	
		CV_RGB(255,255,0),	CV_RGB(255,0,0),	CV_RGB(255,0,255),
		CV_RGB(0,128,255), 	CV_RGB(255,128,0)} ;
	string Dict_6class[6] = {"face","eye","mouse","nose","hair","bread"};
	
	/********************************Open Query List*****************************/
	fpListFile = fopen(loadAnnotations,"r");
	if (!fpListFile) 
	{
		cout << "0.can't open " << loadAnnotations << endl;
		return TEC_INVALID_PARAM;
	}

	FILE *fpListFile_trainval = fopen("res/trainval.txt","at+");
	if (!fpListFile_trainval) 
	{
		cout << "0.can't open " << "res/trainval.txt" << endl;
		return TEC_INVALID_PARAM;
	}

	/*****************************Process one by one*****************************/
	nCount = 0;
	while(EOF != fscanf(fpListFile, "%s %d %d %d %d %d %d %d %d %d %d", 
		&loadIID,&loadPoint[0],&loadPoint[1],&loadPoint[2],&loadPoint[3],&loadPoint[4],
		&loadPoint[5],&loadPoint[6],&loadPoint[7],&loadPoint[8],&loadPoint[9]))
	{	
		iid = loadIID;
		sprintf( loadImgPath, "%s/%s", imgPath, iid.c_str() );
		IplImage *img = cvLoadImage(loadImgPath);
		if(!img || (img->width<32) || (img->height<32) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
		{	
			cout<<"Can't open " << loadImgPath << endl;
			cvReleaseImage(&img);img = 0;
			continue;
		}	

		//get iid
		strImageID = api_commen.GetStringIDFromFilePath(loadImgPath);

		//Count_FaceRectFromPoint
		FacePoint.clear();
		nRet = api_commen.Count_FaceRectFromPoint(loadPoint, img->width, img->height, FacePoint);
		if ( ( nRet!= 0 ) || (FacePoint.size()<1) )
		{
			cout<<"unnormal face:" << loadImgPath << endl;

			//cp img file
			sprintf( tPath, "cp %s res/unnormal_face/%s", loadImgPath, iid.c_str() );
			system( tPath );
			
			cvReleaseImage(&img);img = 0;
			continue;
		}

		/************************save one img with muti label*****************************/
		//if( nCount%50 == 0 )
		{
			Mat matImg(img);
			{				
				//point
				for (i=0;i<5;i++)
				{
					color = colors[i];
					rectangle( matImg, cvPoint(loadPoint[i*2]-2, loadPoint[i*2+1]-2),
			                   cvPoint(loadPoint[i*2]+2, loadPoint[i*2+1]+2), color, 2, 8, 0);
				}

				//face rectangle
				for (i=0;i<FacePoint.size();i++)
				{
					color = colors[i];
					rectangle( matImg, cvPoint(FacePoint[i][0], FacePoint[i][1]),
						cvPoint(FacePoint[i][2], FacePoint[i][3]), color, 2, 8, 0);
				}
			}
			sprintf( szImgPath, "%s/%s", imgSavePath, iid.c_str() );
			imwrite( szImgPath, matImg );
		}
		cvReleaseImage(&img);img = 0;

		//send data
		vecLabelRect.clear();
		for(i=0;i<FacePoint.size();i++)
		{
			vecLabelRect.push_back(make_pair(Dict_6class[i],FacePoint[i]));
		}

		//cp xml file
		nRet = api_xml.write_xml( strImageID, xmlSavePath, vecLabelRect );
		if (nRet!=0)
		{
			cout << "Err to write_xml!" << endl;
			continue;
		}

		//write trainval.txt
		fprintf(fpListFile_trainval, "%s\n", strImageID.c_str() );

		nCount++;
		if( nCount%50 == 0 )
		{
			printf("Loaded %ld img...\n",nCount);
			printf("load %s:",iid.c_str());
			for(i=0;i<10;i++)
			{
				printf("%d_%d ",i,loadPoint[i]);
			}
			printf("\n");
		}

		//set 0
		for(i=0;i<10;i++)
			loadPoint[i]=0;
	}
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	
	if (fpListFile_trainval) {fclose(fpListFile_trainval);fpListFile_trainval = 0;}	

	//print
	printf("All Loaded %ld img!!\n",nCount);

	return nRet;
}	

int Check_SenceTimeData( char* imgPath, char* loadAnnotations, char* xmlSavePath, char* imgSavePath )
{
	/*****************************Init:Commen*****************************/
	char tPath[256];
	char loadPath[4096];
	char loadImgPath[256];
	char szImgPath[256];
	int faceNum,nRet = 0;
	char tmpChar;
	char vocab[4096];
	string word,strImageID;
	long long nCount, QueryAllSize, idQuary, labelQuary, i, j, a, b, w2v_words, w2v_size, wordCount;
	FILE *fpListFile = 0, *fpFeatOut = 0;

	vector< string > vecQuaryString;
	const long long max_w = 4096;
	int x1,y1,x2,y2;
	int loadPoint[10] = {0};
	vector< Vec4i > FacePoint;

	API_XML api_xml;
	API_COMMEN api_commen;
	vector< pair< string, Vec4i > > 	vecLabelRect;

	Scalar color;
	const static Scalar colors[] =  { 	
		CV_RGB(0,0,255),	CV_RGB(0,255,255),	CV_RGB(0,255,0),	
		CV_RGB(255,255,0),	CV_RGB(255,0,0),	CV_RGB(255,0,255),
		CV_RGB(0,128,255), 	CV_RGB(255,128,0)} ;
	string Dict_6class[6] = {"face","eye","mouse","nose","hair","bread"};
	
	/********************************Load:Open Query List*****************************/
	fpListFile = fopen(loadAnnotations,"r");
	if (!fpListFile) 
	{
		cout << "Can't open " << loadAnnotations << endl;
		return -1;
	}

	FILE *fpListFile_trainval = fopen("res/trainval.txt","at+");
	if (!fpListFile_trainval) 
	{
		cout << "0.can't open " << "res/trainval.txt" << endl;
		return TEC_INVALID_PARAM;
	}

	/*****************************Load Query*****************************/
	nCount = 0;
	//fscanf(fpListFile, "%lld", &w2v_words);
	w2v_words = api_commen.doc2vec_CountLines(loadAnnotations);
	printf("Load Query.txt...w2v_words-%lld\n",w2v_words);
	for (b = 0; b < w2v_words; b++) {
		//load word to vector
		a = 0;
		vecQuaryString.clear();

		while (1) {
			vocab[a] = fgetc(fpListFile);	
			tmpChar = vocab[a];
			if ((a < max_w) && (vocab[a] != ' ') && (vocab[a] != '\n')) a++;
			if ( ( vocab[a] == ' ' ) || feof(fpListFile) || (vocab[a] == '\n') )
			{
				if ( ( vocab[0] != ' ' ) && (vocab[0] != '\n') )
				{
					vocab[a] = 0;
					word = vocab;
					vecQuaryString.push_back( word );
					//printf("a:%lld,word:%s\n",a,word.c_str());

					a = 0;
				}
			}
			if (feof(fpListFile) || (tmpChar == '\n')) break;
		}

		faceNum = atoi(vecQuaryString[1].c_str());
		//if ( faceNum < 2 )	//muti people
		if ( faceNum != 1 )	//single people
		{
			//printf( "vecQuaryString.size():%ld, do not find info!!b:%lld\n", vecQuaryString.size(), b );
			continue;
		}

		//load img
		sprintf( loadImgPath, "%s/%s", imgPath, vecQuaryString[0].c_str() );
		IplImage *img = cvLoadImage(loadImgPath);
		if(!img || (img->width<32) || (img->height<32) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
		{	
			cout<<"Can't open " << loadImgPath << endl;
			cvReleaseImage(&img);img = 0;
			continue;
		}

		//get iid
		strImageID = api_commen.GetStringIDFromFilePath(loadImgPath);
		
		Mat matImg(img);	
		vecLabelRect.clear();
		for(i=0;i<faceNum;i++)
		{
			//left eye[104]
			loadPoint[0] = int(atof(vecQuaryString[217+i*219].c_str())+0.5);
			loadPoint[1] = int(atof(vecQuaryString[218+i*219].c_str())+0.5);

			//right eye[105]
			loadPoint[2] = int(atof(vecQuaryString[219+i*219].c_str())+0.5);
			loadPoint[3] = int(atof(vecQuaryString[220+i*219].c_str())+0.5);

			//nose[46]
			loadPoint[4] = int(atof(vecQuaryString[101+i*219].c_str())+0.5);
			loadPoint[5] = int(atof(vecQuaryString[102+i*219].c_str())+0.5);

			//left mouse[84]
			loadPoint[6] = int(atof(vecQuaryString[177+i*219].c_str())+0.5);
			loadPoint[7] = int(atof(vecQuaryString[178+i*219].c_str())+0.5);

			//right mouse[100]
			loadPoint[8] = int(atof(vecQuaryString[209+i*219].c_str())+0.5);
			loadPoint[9] = int(atof(vecQuaryString[210+i*219].c_str())+0.5);

			//Count_FaceRectFromPoint
			FacePoint.clear();
			nRet = api_commen.Count_FaceRectFromPoint(loadPoint, img->width, img->height, FacePoint);
			if ( ( nRet!= 0 ) || (FacePoint.size()<1) )
			{
				cout<<"unnormal face:" << loadImgPath << endl;

				//cp img file
				sprintf( tPath, "cp %s res/unnormal_face/%s", loadImgPath, vecQuaryString[0].c_str() );
				system( tPath );

				//rm Annoation xml file
				vecLabelRect.clear();

				//delete
				cvReleaseImage(&img);img = 0;
				break;
			}

			if( nCount%50 == 0 )
			{
				//point
				for (j=0;j<5;j++)
				{
					color = colors[j];
					rectangle( matImg, cvPoint(loadPoint[j*2]-1, loadPoint[j*2+1]-1),
			                   cvPoint(loadPoint[j*2]+1, loadPoint[j*2+1]+1), color, 1, 8, 0);
				}

				//face rectangle
				for (j=0;j<FacePoint.size();j++)
				{
					color = colors[j];
					rectangle( matImg, cvPoint(FacePoint[j][0], FacePoint[j][1]),
						cvPoint(FacePoint[j][2], FacePoint[j][3]), color, 2, 8, 0);
				}
			}

			//send data
			for(j=0;j<FacePoint.size();j++)
			{
				vecLabelRect.push_back(make_pair(Dict_6class[j],FacePoint[j]));
			}

			//set 0
			for(j=0;j<10;j++)
				loadPoint[j]=0;
			
		}

		//cp xml file
		nRet = api_xml.write_xml( strImageID, xmlSavePath, vecLabelRect );
		if (nRet!=0)
		{
			cout << "Err to write_xml!" << endl;
			continue;
		}

		/************************save one img with muti label*****************************/
		if( nCount%50 == 0 )
		{
			sprintf( szImgPath, "%s/%s", imgSavePath, vecQuaryString[0].c_str() );
			imwrite( szImgPath, matImg );
		}
		cvReleaseImage(&img);img = 0;

		//write trainval.txt
		fprintf(fpListFile_trainval, "%s\n", strImageID.c_str() );

		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld img...\n",nCount);
	}

	/*********************************close file*************************************/
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}
	if (fpListFile_trainval) {fclose(fpListFile_trainval);fpListFile_trainval = 0;}	

	/*********************************Print Info*********************************/
	if ( ( nCount != 0 ) && ( b != 0 ) )
	{
		printf( "All Loaded string:%lld,img:%lld!!\nDone!!\n", b, nCount);
	}
	
	return nRet;
}

int ReGet_SenceTimeData( char *szQueryList, char *inImgPath, char *inAnnoationPath, char* xmlSavePath, char* imgSavePath )
{
	char tPath[256];
	char loadImgPath[256];
	char szImgPath[256];
	string strImageID;
	int i, j, label, svImg, nRet = 0;
	unsigned long long nCount,ImageID = 0;
	FILE *fpListFile = 0;

	/********************************Init*****************************/
	API_XML api_xml;
	API_COMMEN api_commen;
	vector< pair< string, Vec4i > > vecLabelRect;

	string xmlImageID;
	vector< pair< string, Vec4i > > vecXmlLabelRect;
	
	/********************************Open Query List*****************************/
	fpListFile = fopen(szQueryList,"r");
	if (!fpListFile) 
	{
		cout << "0.can't open " << szQueryList << endl;
		return TEC_INVALID_PARAM;
	}

	FILE *fpListFile_trainval = fopen("res/trainval.txt","at+");
	if (!fpListFile_trainval) 
	{
		cout << "0.can't open " << "res/trainval.txt" << endl;
		return TEC_INVALID_PARAM;
	}

	/*****************************Process one by one*****************************/
	nCount = 0;
	while(EOF != fscanf(fpListFile, "%s", &loadImgPath ))
	{
		//get iid
		strImageID = api_commen.GetStringIDFromFilePath( loadImgPath );

		sprintf( szImgPath, "%s/%s.jpg", inImgPath, strImageID.c_str() );
		IplImage *img = cvLoadImage(szImgPath);
		if(!img || (img->width<32) || (img->height<32) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
		{	
			cout<<"Can't open " << szImgPath << endl;
			continue;
		}		
		cvReleaseImage(&img);img = 0;

		/************************loadXml*****************************/
		vecXmlLabelRect.clear();
		sprintf(szImgPath, "%s/%s.xml",inAnnoationPath,strImageID.c_str());
		nRet = api_xml.load_xml( szImgPath, xmlImageID, vecXmlLabelRect );
		if ( (nRet!=0) || (vecXmlLabelRect.size()<1) )
		{
			LOOGE<<"[loadXml Err!!loadXml:]"<<szImgPath;
			continue;
		}

		//write trainval.txt
		fprintf(fpListFile_trainval, "%s\n", strImageID.c_str() );

		//cp img file
		sprintf( tPath, "cp %s/%s.jpg %s/%s.jpg", inImgPath, strImageID.c_str(), imgSavePath, strImageID.c_str() );
		system( tPath );

		//cp xml file
		sprintf( tPath, "cp %s/%s.xml %s/%s.xml", inAnnoationPath, strImageID.c_str(), xmlSavePath, strImageID.c_str() );
		system( tPath );

		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld img...\n",nCount);
	}
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	
	if (fpListFile_trainval) {fclose(fpListFile_trainval);fpListFile_trainval = 0;}	

	/*********************************Print Info*********************************/
	if ( nCount != 0 )
	{
		printf( "All Loaded img:%lld!!\nDone!!\n", nCount);
	}
	
	return nRet;
}	

int Get_Xml_Bing_ROI( char *szQueryList, char* loadXmlPath, char* KeyFilePath, char *layerName, int binGPU, int deviceID )
{
	char tPath[256];
	char loadImgPath[256];
	char szImgPath[256];
	int i, j, label, svImg, nRet = 0;
	long inputLabel, nCount, svBingPos, svBingNeg;
	unsigned long long ImageID = 0;
	string strImageID;
	double allPredictTime;
	FILE *fpListFile = 0;

	RunTimer<double> run;
	API_XML api_xml;
	API_COMMEN api_commen;
	API_FACE_DETECT api_face_detect;

	vector < pair < string,float > > Res;

	/***********************************Init*************************************/
	vector< pair<float, Vec4i> > boxHypothese;

	plog::init(plog::info, "plog.txt"); 

	string xmlImageID;
	vector< pair< string, Vec4i > > vecXmlLabelRect;
	vector< pair< string, Vec4i > > vecOutXmlRect;
	
	/********************************Open Query List*****************************/
	fpListFile = fopen(szQueryList,"r");
	if (!fpListFile) 
	{
		cout << "0.can't open " << szQueryList << endl;
		return TEC_INVALID_PARAM;
	}

	FILE *fpListFile_training = fopen("res/BING_ROI_TRAINING.txt","w");
	if (!fpListFile_training) 
	{
		cout << "0.can't open " << "res/BING_ROI_TRAINING.txt" << endl;
		return TEC_INVALID_PARAM;
	}

	FILE *fpListFile_trainval = fopen("res/trainval.txt","w");
	if (!fpListFile_trainval) 
	{
		cout << "0.can't open " << "res/trainval.txt" << endl;
		return TEC_INVALID_PARAM;
	}

	/***********************************Init*************************************/
	nRet = api_face_detect.Init( KeyFilePath, layerName, binGPU, deviceID ); 
	if (nRet != 0)
	{
	   cout<<"Fail to initialization "<<endl;
	   return TEC_INVALID_PARAM;
	}

	nCount = 0;
	allPredictTime = 0.0;
	/*****************************Process one by one*****************************/
	while(EOF != fscanf(fpListFile, "%s", loadImgPath))
	{
		IplImage *img = cvLoadImage(loadImgPath);
		if(!img || (img->width<32) || (img->height<32) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
		{	
			cout<<"Can't open " << loadImgPath << endl;
			continue;
		}	

		/************************getRandomID*****************************/
		strImageID = api_commen.GetStringIDFromFilePath( loadImgPath );

		/************************loadXml*****************************/
		vecXmlLabelRect.clear();
		sprintf(szImgPath, "%s/%s.xml",loadXmlPath,strImageID.c_str());
		nRet = api_xml.load_xml( szImgPath, xmlImageID, vecXmlLabelRect );
		if ( (nRet!=0) || (vecXmlLabelRect.size()<1) )
		{
			LOOGE<<"[loadXml Err!!loadXml:]"<<szImgPath;
			continue;
		}

		/************************get xml Hypothese*****************************/
		vecOutXmlRect.clear();
		nRet = api_face_detect.Get_Xml_Hypothese( xmlImageID, img->width, img->height, vecXmlLabelRect, vecOutXmlRect );
		if ( (nRet!=0) || (vecOutXmlRect.size()<1) )
		{
			LOOGE<<"[get_xml_Hypothese Err!!xmlImageID:]"<<xmlImageID;
			continue;
		}

		/************************Get_Bing_Hypothese*****************************/	
		boxHypothese.clear();
		run.start();
		//BinTraining:2-NO Remove Rectfor Training;1-Remove small Rect for Training;0-Remove small Rect for Test
		nRet = api_face_detect.Get_Bing_Hypothese( img, boxHypothese, 2 );
		if ( (nRet!=0) || (boxHypothese.size()<1) )
		{
			LOOGE<<"[Get_Bing_Hypothese Err!!loadImgPath:]"<<loadImgPath;
			continue;
		}
		run.end();
		LOOGI<<"[Get_Bing_Hypothese] time:"<<run.time();
		allPredictTime += run.time();

		/************************Get_iou_cover*****************************/
		vector< pair<Vec4i, pair< double, int > > > vectorIouCover;
		nRet = api_face_detect.Get_iou_cover( boxHypothese, vecOutXmlRect, vectorIouCover );
		if ( (nRet!=0) || (vectorIouCover.size()<1) )
		{
			LOOGE<<"[Get_iou_cover Err!!loadImgPath:]"<<loadImgPath;
			continue;
		}

		/************************save roi data*****************************/
		fprintf(fpListFile_training, "%s %d", strImageID.c_str(), vecOutXmlRect.size()+vectorIouCover.size() );
		for ( j=0;j<vecOutXmlRect.size();j++ )
		{
			fprintf(fpListFile_training, " %d %d %d %d", vecOutXmlRect[j].second[0], vecOutXmlRect[j].second[1],
	        	vecOutXmlRect[j].second[2], vecOutXmlRect[j].second[3] );
		}

		svBingPos = 0;
		svBingNeg = 0;
		for ( j=0;j<vectorIouCover.size();j++ )
		{
			fprintf(fpListFile_training, " %d %d %d %d", vectorIouCover[j].first[0], vectorIouCover[j].first[1],
	        	vectorIouCover[j].first[2], vectorIouCover[j].first[3] );

			if (vectorIouCover[j].second.first>=0.5)	//pos
				svBingPos++;
			if ( vectorIouCover[j].second.first<=0.3 )	//neg
				svBingNeg++;		
		}
		fprintf(fpListFile_training, "\n");

		//write trainval.txt
		fprintf(fpListFile_trainval, "%s\n", strImageID.c_str() );

		nCount++;
		if( nCount%50 == 0 )
		{
			printf("Loaded %ld img...\n",nCount);
			//printf("%s,xml:%d,bing:pos-%d,neg-%d\n", 
			//	strImageID.c_str(), vecOutXmlRect.size(), svBingPos, svBingNeg );
		}
		printf("%s,xml:%d,bing:pos-%d,neg-%d\n", 
			strImageID.c_str(), vecOutXmlRect.size(), svBingPos, svBingNeg );

		cvReleaseImage(&img);img = 0;
	}

	/*********************************close file*************************************/
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	
	if (fpListFile_training) {fclose(fpListFile_training);fpListFile_training = 0;}	
	if (fpListFile_trainval) {fclose(fpListFile_trainval);fpListFile_trainval = 0;}	

	/*********************************Release*************************************/
	api_face_detect.Release();

	/*********************************Print Info*********************************/
	if ( nCount != 0 ) 
	{
		printf( "nCount:%ld,PredictTime:%.4fms\n", nCount, allPredictTime*1000.0/nCount );
	}
	
	cout<<"Done!! "<<endl;
	
	return nRet;
}

int frcnn_test( char *szQueryList, char* KeyFilePath, char *layerName, int binGPU, int deviceID, int saveImg=0 )
{
	char tPath[256];
	char loadImgPath[256];
	char szImgPath[256];
	char savePath[256];
	string text;
	vector<string> vecText;
	int i, j, label, svImg, nRet = 0;
	long inputLabel, nCount;
	unsigned long long ImageID = 0;
	string strImageID;
	double allPredictTime;
	FILE *fpListFile = 0;

	RunTimer<double> run;
	API_COMMEN api_commen;
	API_FACE_DETECT api_face_detect;

	vector< pair< pair< string, Vec4i >, float > > 	Res;

	/***********************************Init**********************************/
	plog::init(plog::info, "plog.txt"); 

	const static Scalar colors[] =  { 	CV_RGB(0,0,255),	CV_RGB(0,255,255),	CV_RGB(0,255,0),	
										CV_RGB(255,255,0),	CV_RGB(255,0,0),	CV_RGB(255,0,255),
										CV_RGB(0,128,255), 	CV_RGB(255,128,0)} ;
	
	/********************************Open Query List*****************************/
	fpListFile = fopen(szQueryList,"r");
	if (!fpListFile) 
	{
		cout << "0.can't open " << szQueryList << endl;
		return TEC_INVALID_PARAM;
	}

	/***********************************Init*************************************/
	nRet = api_face_detect.Init( KeyFilePath, layerName, binGPU, deviceID ); 
	if (nRet != 0)
	{
	   cout<<"Fail to initialization "<<endl;
	   return TEC_INVALID_PARAM;
	}

	nCount = 0;
	allPredictTime = 0.0;
	/*****************************Process one by one*****************************/
	while(EOF != fscanf(fpListFile, "%s", loadImgPath))
	{
		IplImage *img = cvLoadImage(loadImgPath);
		if(!img || (img->width<32) || (img->height<32) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
		{	
			cout<<"Can't open " << loadImgPath << endl;
			continue;
		}	

		/************************getRandomID*****************************/
		//api_commen.getRandomID( ImageID );
		//ImageID = api_commen.GetIDFromFilePath( loadImgPath );
		strImageID = api_commen.GetStringIDFromFilePath( loadImgPath );

		/************************ResizeImg*****************************/
		IplImage* imgResize = api_commen.ResizeImg( img );
		
		/************************Predict*****************************/	
		Res.clear();
		run.start();
		nRet = api_face_detect.Predict( imgResize, strImageID, Res );
		if ( (nRet!=0) || (Res.size()<1) )
		{
			LOOGE<<"[Predict Err!!loadImgPath:]"<<loadImgPath;
			continue;
		}
		run.end();
		LOOGI<<"[Predict] time:"<<run.time();
		allPredictTime += run.time();
		
		/************************save img data*****************************/
		Mat matImg(imgResize);
		sprintf( tPath, "res_predict/" );
		for(i=0;i<Res.size();i++)  
		{				
			Scalar color = colors[i%8];
			rectangle( matImg, cvPoint(Res[i].first.second[0], Res[i].first.second[1]),
	                   cvPoint(Res[i].first.second[2], Res[i].first.second[3]), color, 2, 8, 0);

			sprintf(szImgPath, "%.2f %s", Res[i].second, Res[i].first.first.c_str() );
			text = szImgPath;
			putText(matImg, text, cvPoint(Res[i].first.second[0]+1, Res[i].first.second[1]+20), 
				FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
			//putText(matImg, text, cvPoint(1, i*20+20), FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
				
			sprintf(tPath, "%s%s-%.2f_", tPath, Res[i].first.first.c_str(), Res[i].second );
		}
		if ( saveImg == 0 )
		{
			sprintf( savePath, "%s%s.jpg", tPath, strImageID.c_str() );
			imwrite( savePath, matImg );
		}
		else if ( saveImg == 1 )
		{
			sprintf( savePath, "%s%s.jpg", tPath, strImageID.c_str() );
			cvSaveImage( savePath, imgResize );

			sprintf( savePath, "%s%s_predict.jpg", tPath, strImageID.c_str() );
			imwrite( savePath, matImg );
		}

		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld img...\n",nCount);
		
		cvReleaseImage(&imgResize);imgResize = 0;
		cvReleaseImage(&img);img = 0;
	}

	/*********************************close file*************************************/
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	

	/*********************************Release*************************************/
	api_face_detect.Release();

	/*********************************Print Info*********************************/
	if ( nCount != 0 ) 
	{
		printf( "nCount:%ld,PredictTime:%.4fms\n", nCount, allPredictTime*1000.0/nCount );
	}
	
	cout<<"Done!! "<<endl;
	
	return nRet;	
}


int main(int argc, char* argv[])
{
	int  ret = 0;
	char szKeyFiles[256],szSavePath[256];
	API_COMMEN api_commen;

	if (argc == 6 && strcmp(argv[1],"-get_img2xml") == 0) {
		ret = Get_Img2Xml( argv[2], argv[3], argv[4], argv[5] );
	}
	else if (argc == 6 && strcmp(argv[1],"-Get_CelebA_FaceLabel") == 0) {
		ret = Get_CelebA_FaceLabel( argv[2], argv[3], argv[4], argv[5] );
	}
	else if (argc == 6 && strcmp(argv[1],"-check_sencetime") == 0) {
		ret = Check_SenceTimeData( argv[2], argv[3], argv[4], argv[5] );
	}
	else if (argc == 7 && strcmp(argv[1],"-reget_sencetime") == 0) {
		ret = ReGet_SenceTimeData( argv[2], argv[3], argv[4], argv[5], argv[6] );
	}
	else if (argc == 8 && strcmp(argv[1],"-get_xml_bing_roi") == 0) {
		ret = Get_Xml_Bing_ROI( argv[2], argv[3], argv[4], argv[5], atol(argv[6]), atol(argv[7]) );
	}
	else if (argc == 8 && strcmp(argv[1],"-frcnn") == 0) {
		ret = frcnn_test( argv[2], argv[3], argv[4], atol(argv[5]), atol(argv[6]), atol(argv[7]) );
	}
	else
	{
		cout << "usage:\n" << endl;
		cout << "\tDemo_facedetect -get_img2xml queryList.txt xmlSavePath imgSavePath labelname\n" << endl;
		cout << "\tDemo_facedetect -Get_CelebA_FaceLabel imgPath loadAnnotations xmlSavePath imgSavePath\n" << endl;
		cout << "\tDemo_facedetect -check_sencetime imgPath loadAnnotations xmlSavePath imgSavePath\n" << endl;
		cout << "\tDemo_facedetect -reget_sencetime queryList.txt imgPath loadAnnotations xmlSavePath imgSavePath\n" << endl;
		cout << "\tDemo_facedetect -get_xml_bing_roi queryList.txt loadXmlPath keyFilePath layerName binGPU deviceID\n" << endl;
		cout << "\tDemo_facedetect -frcnn queryList.txt keyFilePath layerName binGPU deviceID saveImg\n" << endl;
		return ret;
	}
	return ret;
}
