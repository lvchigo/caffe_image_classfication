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

using namespace cv;
using namespace std;

struct Sencetime_fddb_Info
{
    float 	score;
    int		x1;
	int		y1;
	int		x2;
	int 	y2;
};

int Change_Name( char *szQueryList, char* imgPath )
{
	char tPath[256];
	char loadImgPath[256];
	char szImgPath[256];
	string strImageID;
	int i, j, label, svImg, nRet = 0;
	unsigned long long nCount,ImageID = 0;
	FILE *fpListFile = 0;
	
	/********************************Open Query List*****************************/
	fpListFile = fopen(szQueryList,"r");
	if (!fpListFile) 
	{
		cout << "0.can't open " << szQueryList << endl;
		return TEC_INVALID_PARAM;
	}

	FILE *fpListFile_trainval = fopen("ChangeName.txt","at+");
	if (!fpListFile_trainval) 
	{
		cout << "0.can't open " << "ChangeName.txt" << endl;
		return TEC_INVALID_PARAM;
	}

	/*****************************Process one by one*****************************/
	nCount = 0;
	while(EOF != fscanf(fpListFile, "%s", &loadImgPath ))
	{
		//change name
		long ID = 0;
		int  atom =0;
		string tmpPath = loadImgPath;
		long start = tmpPath.find_last_of('_');
		long end = tmpPath.find_last_of('.');
		if ( (start>0) && (end>0) && ( end>start ) )
			strImageID = tmpPath.substr(start+1,end-start-1);
		else
			continue;
		//printf("strImageID:%s\n",strImageID.c_str());

		sprintf( tPath, "%s/%s.jpg", imgPath, strImageID.c_str() );
		IplImage *img = cvLoadImage(tPath);
		if(!img || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
		{	
			cout<<"Can't open " << tPath << endl;
			continue;
		}	
		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld img...\n",nCount);

		//write trainval.txt
		fprintf(fpListFile_trainval, "%s\n", tPath );

		cvReleaseImage(&img);img = 0;
	}
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	
	if (fpListFile_trainval) {fclose(fpListFile_trainval);fpListFile_trainval = 0;}	

	cout<<"Done!! "<<endl;
	
	return nRet;
}	


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
	vector< pair<string, Vec4i> > FacePoint;

	API_XML api_xml;
	API_COMMEN api_commen;

	Scalar color;
	const static Scalar colors[] =  { 	
		CV_RGB(0,0,255),	CV_RGB(0,255,255),	CV_RGB(0,255,0),	
		CV_RGB(255,255,0),	CV_RGB(255,0,0),	CV_RGB(255,0,255),
		CV_RGB(0,128,255), 	CV_RGB(255,128,0)} ;
	
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
		if( nCount%50 == 0 )
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
					rectangle( matImg, cvPoint(FacePoint[i].second[0], FacePoint[i].second[1]),
						cvPoint(FacePoint[i].second[2], FacePoint[i].second[3]), color, 2, 8, 0);
				}
			}
			sprintf( szImgPath, "%s/%s", imgSavePath, iid.c_str() );
			imwrite( szImgPath, matImg );
		}
		cvReleaseImage(&img);img = 0;

		//cp xml file
		nRet = api_xml.write_xml( strImageID, xmlSavePath, FacePoint );
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

//change celeba unnormal face xml to useful xml
int Change_XML_Name( char *szQueryList, char* imgPath, char* xmlSavePath, char* imgSavePath )
{
	char tPath[256];
	char loadImgPath[256];
	char szImgPath[256];
	int i, j, svImg, bin_Bad_Label, nRet = 0;
	unsigned long long nCount, labelCount, ImageID = 0;
	FILE *fpListFile = 0;

	/********************************Init*****************************/
	string strLabel;
	string filename,tmpPath,iid;
	long end = 0;

	map< string,int > 				mapLabel;
	map< string, int >::iterator 	itLabel;

	API_XML api_xml;
	vector< pair< string, Vec4i > > 	vecLabelRect;
	
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
	while(EOF != fscanf(fpListFile, "%s", &loadImgPath))
	{	
		// reading the xml files
		nRet = api_xml.load_xml( string(loadImgPath), filename, vecLabelRect );
		if ( (nRet!=0) || (vecLabelRect.size()<1) )
		{
			cout << "Err to load_xml!" << endl;
			continue;
		}

		bin_Bad_Label = 0;
		for(i=0;i<vecLabelRect.size();i++)
		{
			strLabel = vecLabelRect[i].first;

			//err label
			if ( (strLabel == "Copy of face") || (strLabel == "eye") )
			{
				bin_Bad_Label = 1;
				break;
			}
			
			//write map
			itLabel = mapLabel.find(strLabel);		
			if (itLabel != mapLabel.end()) // find it
				mapLabel[itLabel->first] = itLabel->second+1;		//[In]dic code-words
			else
				mapLabel[strLabel] = 1;
		}

		if ( bin_Bad_Label == 1 )
			continue;

		//cp xml file
		nRet = api_xml.write_xml( filename, xmlSavePath, vecLabelRect );
		if (nRet!=0)
		{
			cout << "Err to write_xml!" << endl;
			continue;
		}

		//cp img file
		sprintf( tPath, "cp %s/%s.jpg %s/%s.jpg", imgPath, filename.c_str(), imgSavePath, filename.c_str() );
		system( tPath );

		//write trainval.txt
		fprintf(fpListFile_trainval, "%s\n", filename.c_str() );

		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld img...\n",nCount);
	}
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	
	if (fpListFile_trainval) {fclose(fpListFile_trainval);fpListFile_trainval = 0;}	

	//write label file
	sprintf( tPath, "res/Dict_FRCNN_%dlabel.txt", mapLabel.size() );
	FILE *fpListFile_training = fopen(tPath,"w");
	if (!fpListFile_training) 
	{
		cout << "0.can't open " << "BING_ROI_TRAINING.txt" << endl;
		return TEC_INVALID_PARAM;
	}

	labelCount = 0;
	for(itLabel = mapLabel.begin(); itLabel != mapLabel.end(); itLabel++)
	{
		fprintf(fpListFile_training, "%s\n", itLabel->first.c_str());
		printf("%s %d\n", itLabel->first.c_str(), itLabel->second);

		labelCount += itLabel->second;
	}
	if (fpListFile_training) {fclose(fpListFile_training);fpListFile_training = 0;}	

	printf("Load xml num:%lld, all label count:%lld\n", nCount, labelCount );
	printf("Total Class num:%d\n", mapLabel.size() );
	cout<<"Done to cp xml file && write label file!! "<<endl;
	
	return nRet;
}	


int Get_wider_FaceLabel( char* szQueryList, char* loadAnnotations, char* xmlSavePath, char* srcImgSavePath, char* imgSavePath )
{
	char tPath[256];
	char loadImgPath[256];
	char szImgPath[256];
	char loadIID[256];
	int svImg, labelCount, nRet = 0;
	unsigned long long nCount, nCountFace, ImageID = 0;
	long long QueryAllSize, idQuary, labelQuary, i, j, a, b, w2v_words, w2v_size, wordCount;
	FILE *fpListFile = 0;

	char tmpChar;
	char vocab[4096];
	vector< string > vecQuaryString;
	const long long max_w = 4096;

	/********************************Init*****************************/
	string strLabel,word;
	string filename,tmpPath,iid,strImageID;
	vector< pair< string, Vec4i > >  FacePoint;

	API_XML api_xml;
	API_COMMEN api_commen;
	vector< pair< string, Vec4i > > 	vecLabelRect;
	map<string, string> mapIDPath;
	map<string, string>::iterator itIDPath;
	int x1,y1,x2,y2;

	Scalar color;
	const static Scalar colors[] =  { 	
		CV_RGB(0,0,255),	CV_RGB(0,255,255),	CV_RGB(0,255,0),	
		CV_RGB(255,255,0),	CV_RGB(255,0,0),	CV_RGB(255,0,255),
		CV_RGB(0,128,255), 	CV_RGB(255,128,0)} ;
	//string Dict_6class[6] = {"face","eye","mouse","nose","hair","bread"};
	string Dict_7class[7] = {"face","halfface","eye","mouse","nose","hair","bread"};
	
	/********************************Open Query List*****************************/
	FILE *fpQListFile = fopen(szQueryList,"r");
	if (!fpQListFile) 
	{
		cout << "0.can't open " << szQueryList << endl;
		return TEC_INVALID_PARAM;
	}
	
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

	/*****************************imglist:Process one by one*****************************/
	printf("Loading imglist...\n");
	while(EOF != fscanf(fpQListFile, "%s", &loadImgPath ))
	{	
		IplImage *img = cvLoadImage(loadImgPath);
		if(!img || (img->width<32) || (img->height<32) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
		{	
			cout<<"Can't open " << loadImgPath << endl;
			cvReleaseImage(&img);img = 0;
			continue;
		}	

		//get iid
		strImageID = api_commen.GetStringIDFromFilePath(loadImgPath);
		mapIDPath[strImageID] = string(loadImgPath);
		cvReleaseImage(&img);img = 0;
	}
	printf("End Loaded imglist!!\n");
	if (fpQListFile) {fclose(fpQListFile);fpQListFile = 0;}

	/*****************************Load Query*****************************/
	nCount = 0;
	nCountFace = 0;
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

		//remove big face
		labelCount = atoi(vecQuaryString[1].c_str());
		if ( labelCount < 2 )	//muti people
		//if ( labelCount != 1 )	//single people
			continue;

		//find path
		strImageID = vecQuaryString[0];
		itIDPath = mapIDPath.find(strImageID);
		if (itIDPath == mapIDPath.end())
		{	
			cout<<"Can't open " << strImageID << endl;
			continue;
		}	

		//find it
		tmpPath = itIDPath->second;
		vecLabelRect.clear();
		for(i=0;i<labelCount;i++)
		{
			//read vect
			x1 = atoi(vecQuaryString[2+4*i].c_str());
			y1 = atoi(vecQuaryString[3+4*i].c_str());
			x2 = atoi(vecQuaryString[4+4*i].c_str());
			y2 = atoi(vecQuaryString[5+4*i].c_str());

			if ( (x1<0)||(y1<0)||(x2<0)||(y2<0)||(x1>=x2)||(y1>=y2) )
				continue;

			//remove too small face
			if ( (x2-x1<20)||(y2-y1<20) )
				continue;

			//send data
			Vec4i tmpRect(x1, y1, x2, y2);
			vecLabelRect.push_back(make_pair(Dict_7class[0],tmpRect));
		}

		if (vecLabelRect.size()<1)
			continue;

		//cp xml file
		nRet = api_xml.write_xml( strImageID, xmlSavePath, vecLabelRect );
		if (nRet!=0)
		{
			cout << "Err to write_xml!" << endl;
			continue;
		}

		//cp img file
		sprintf( tPath, "cp %s %s/", tmpPath.c_str(), srcImgSavePath );
		system( tPath );

		//write trainval.txt
		fprintf(fpListFile_trainval, "%s\n", strImageID.c_str() );

		nCount++;
		nCountFace += vecLabelRect.size();
		/************************save one img with muti label*****************************/
		if( nCount%50 == 0 )
		{
			printf("Loaded %ld img...\n",nCount);
		}
		{
			IplImage *img = cvLoadImage(tmpPath.c_str());
			Mat matImg(img);
			{				
				//face rectangle
				for (j=0;j<vecLabelRect.size();j++)
				{
					color = colors[j%8];
					rectangle( matImg, cvPoint(vecLabelRect[j].second[0], vecLabelRect[j].second[1]),
						cvPoint(vecLabelRect[j].second[2], vecLabelRect[j].second[3]), color, 1, 8, 0);
				}
			}
			sprintf( szImgPath, "%s/%s.jpg", imgSavePath, strImageID.c_str() );
			imwrite( szImgPath, matImg );
			cvReleaseImage(&img);img = 0;
		}

	}
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	
	if (fpListFile_trainval) {fclose(fpListFile_trainval);fpListFile_trainval = 0;}	

	//print
	printf("All Loaded %ld img!!\n",nCount);
	printf("All Loaded %ld face!!\n",nCountFace);

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
	vector< pair< string, Vec4i > > FacePoint;

	API_XML api_xml;
	API_COMMEN api_commen;
	vector< pair< string, Vec4i > > 	vecLabelRect;

	Scalar color;
	const static Scalar colors[] =  { 	
		CV_RGB(0,0,255),	CV_RGB(0,255,255),	CV_RGB(0,255,0),	
		CV_RGB(255,255,0),	CV_RGB(255,0,0),	CV_RGB(255,0,255),
		CV_RGB(0,128,255), 	CV_RGB(255,128,0)} ;
	
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
		if ( faceNum < 1 )	//muti people
		//if ( faceNum != 1 )	//single people
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
					rectangle( matImg, cvPoint(FacePoint[j].second[0], FacePoint[j].second[1]),
						cvPoint(FacePoint[j].second[2], FacePoint[j].second[3]), color, 2, 8, 0);
				}
			}

			//send data
			for(j=0;j<FacePoint.size();j++)
			{
				vecLabelRect.push_back(make_pair(FacePoint[j].first,FacePoint[j].second));
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

		if( nCount>50000 )
			break;
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
		//sprintf( tPath, "cp %s/%s.jpg %s/%s.jpg", inImgPath, strImageID.c_str(), imgSavePath, strImageID.c_str() );
		//system( tPath );

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

int frcnn_test( char *szQueryList, char* KeyFilePath, char *layerName, int binGPU, int deviceID )
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
	API_FACE_DETECT api_face_detect;

	vector< FaceDetectInfo >	Res;

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

	/***********************************Init*************************************/
	nRet = api_face_detect.Init( KeyFilePath, binGPU, deviceID ); 
	if (nRet != 0)
	{
	   cout<<"Fail to initialization "<<endl;
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

		/************************getRandomID*****************************/
		//api_commen.getRandomID( ImageID );
		//ImageID = api_commen.GetIDFromFilePath( loadImgPath );
		strImageID = api_commen.GetStringIDFromFilePath( loadImgPath );
		
		/************************Predict*****************************/	
		Res.clear();
		run.start();
		nRet = api_face_detect.Predict( img, Res );
		if ( (nRet!=0) || (Res.size()<1) )
		{
			LOOGE<<"[Predict Err!!loadImgPath:]"<<loadImgPath;
			cvReleaseImage(&img);img = 0;
			continue;
		}
		run.end();
		//LOOGI<<"[Predict] time:"<<run.time();
		allPredictTime += run.time();
		
		/************************save img data*****************************/
		{
			name = Res[0].label;
			if ( (name=="face")||(name=="halfface")||(name=="eye")||(name=="mouse")||(name=="nose")||(name=="hair")||(name=="beard") )
				sprintf(tPath, "res_predict/face/", tPath );
			else
				sprintf(tPath, "res_predict/noface/", tPath );
		}
		for(i=0;i<Res.size();i++)  
		{				
			Scalar color = colors[i%8];
			cvRectangle( img, cvPoint(Res[i].rect[0], Res[i].rect[1]),
	                   cvPoint(Res[i].rect[2], Res[i].rect[3]), color, 2, 8, 0);

			sprintf(szImgPath, "%.2f %s", Res[i].score, Res[i].label.c_str() );
			text = szImgPath;
			cvPutText( img, text.c_str(), cvPoint(Res[i].rect[0]+1, Res[i].rect[1]+20), &font, color );

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
	api_face_detect.Release();

	/*********************************Print Info*********************************/
	if ( nCount != 0 ) 
	{
		printf( "nCount:%ld,nCountFace:%ld_%.4f,PredictTime:%.4fms\n", 
			nCount, nCountFace, nCountFace*1.0/nCount, allPredictTime*1000.0/nCount );
	}
	
	cout<<"Done!! "<<endl;
	
	return nRet;	
}

int frcnn_test_shuying( char *szQueryList, char* KeyFilePath, char *layerName, int binGPU, int deviceID )
{
	char tPath[256];
	char loadImgPath[256];
	char szImgPath[256];
	char savePath[256];
	vector<string> vecText;
	int i, j, label, svImg, rw, rh, nRet = 0;
	long inputLabel, nCount, nCountFace;
	string strImageID,text,name;
	double allPredictTime;
	FILE *fpListFile = 0;

	RunTimer<double> run;
	API_COMMEN api_commen;
	API_FACE_DETECT api_face_detect;

	vector< FaceDetectInfo >	Res;

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

	FILE *fpListFile_shuying = fopen("res_predict/res_facedetect.txt","wt+");
	if (!fpListFile_shuying) 
	{
		cout << "0.can't open " << "res_predict/res_facedetect.txt" << endl;
		return TEC_INVALID_PARAM;
	}

	/***********************************Init*************************************/
	nRet = api_face_detect.Init( KeyFilePath, binGPU, deviceID ); 
	if (nRet != 0)
	{
	   cout<<"Fail to initialization "<<endl;
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

		/************************getRandomID*****************************/
		//api_commen.getRandomID( ImageID );
		//ImageID = api_commen.GetIDFromFilePath( loadImgPath );
		strImageID = api_commen.GetStringIDFromFilePath( loadImgPath );
		
		/************************Predict*****************************/	
		Res.clear();
		run.start();
		nRet = api_face_detect.Predict( img, Res );
		if ( (nRet!=0) || (Res.size()<1) )
		{
			LOOGE<<"[Predict Err!!loadImgPath:]"<<loadImgPath;
			cvReleaseImage(&img);img = 0;
			continue;
		}
		run.end();
		//LOOGI<<"[Predict] time:"<<run.time();
		allPredictTime += run.time();
		
		/************************save img data*****************************/
		for(i=0;i<Res.size();i++)  
		{				
			Scalar color = colors[i%8];
			cvRectangle( img, cvPoint(Res[i].rect[0], Res[i].rect[1]),
	                   cvPoint(Res[i].rect[2], Res[i].rect[3]), color, 2, 8, 0);

			sprintf(szImgPath, "%.2f %s %d", Res[i].score, Res[i].label.c_str(), i );
			text = szImgPath;
			cvPutText( img, text.c_str(), cvPoint(Res[i].rect[0]+1, Res[i].rect[1]+20), &font, color );

			if ( Res[i].label == "face" )
				nCountFace++;

			rw = Res[i].rect[2]-Res[i].rect[0];
			rh = Res[i].rect[3]-Res[i].rect[1];
			fprintf(fpListFile_shuying, "%s %s %d %.4f %.4f\n", 
				strImageID.c_str(), Res[i].label.c_str(), i, Res[i].score, (rw*rh*1.0/(img->width*img->height)) );
		}
		sprintf( savePath, "res_predict/face/%s.jpg", strImageID.c_str() );
		cvSaveImage( savePath, img );
			
		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld img...,time:%.4f\n",nCount,run.time());
		
		cvReleaseImage(&img);img = 0;
	}

	/*********************************close file*************************************/
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	
	if (fpListFile_shuying) {fclose(fpListFile_shuying);fpListFile_shuying = 0;}	

	/*********************************Release*************************************/
	api_face_detect.Release();

	/*********************************Print Info*********************************/
	if ( nCount != 0 ) 
	{
		printf( "nCount:%ld,nCountFace:%ld_%.4f,PredictTime:%.4fms\n", 
			nCount, nCountFace, nCountFace*1.0/nCount, allPredictTime*1000.0/nCount );
	}
	
	cout<<"Done!! "<<endl;
	
	return nRet;	
}


int frcnn_addSample( char *szQueryList, char* KeyFilePath, char *layerName, int binGPU, int deviceID )
{
	char tPath[256];
	char loadImgPath[256];
	char szImgPath[256];
	char savePath[256];
	vector<string> vecText;
	int i, j, label, svImg, nRet = 0;
	long inputLabel, nCount;
	string strImageID,text,name;
	double allPredictTime;
	FILE *fpListFile = 0;

	RunTimer<double> run;
	API_COMMEN api_commen;
	API_FACE_DETECT api_face_detect;

	vector< FaceDetectInfo >	Res;

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
	nRet = api_face_detect.Init( KeyFilePath, binGPU, deviceID ); 
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
			cvReleaseImage(&img);img = 0;
			continue;
		}	

		/************************getRandomID*****************************/
		strImageID = api_commen.GetStringIDFromFilePath( loadImgPath );
		
		/************************Predict*****************************/	
		Res.clear();
		run.start();
		nRet = api_face_detect.Predict( img, Res );
		if ( (nRet!=0) || (Res.size()<1) )
		{
			LOOGE<<"[Predict Err!!loadImgPath:]"<<loadImgPath;
			cvReleaseImage(&img);img = 0;
			continue;
		}
		run.end();
		//LOOGI<<"[Predict] time:"<<run.time();
		allPredictTime += run.time();
		
		/************************save img data*****************************/
		svImg = 0;
		{
			name = Res[0].label;
			if ( (name=="other")||(name=="pet")||(name=="puppet")||(name=="puppet") )
			{
				sprintf(tPath, "res_predict/noface/", tPath );
				svImg = 1;
			}
		}
		
		if (svImg == 1)
		{
			//cp img file
			sprintf( savePath, "%s%s.jpg", tPath, strImageID.c_str() );
			sprintf( tPath, "cp %s %s", loadImgPath, savePath );
			system( tPath );
		}
			
		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld img...,time:%.4f\n",nCount,run.time());
		
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


int test_fddb( char *szQueryList, char *szQueryPath, char* KeyFilePath, char *layerName, int binGPU, int deviceID )
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
	API_FACE_DETECT api_face_detect;

	vector< FaceDetectInfo >	Res;
	vector< FaceDetectInfo >	FDDB_Res;
	int position;

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

	FILE *fpListFile_fddb = fopen("res_predict/res_fddb.txt","wt+");
	if (!fpListFile_fddb) 
	{
		cout << "0.can't open " << "res_predict/res_fddb.txt" << endl;
		return TEC_INVALID_PARAM;
	}

	/***********************************Init*************************************/
	nRet = api_face_detect.Init( KeyFilePath, binGPU, deviceID ); 
	if (nRet != 0)
	{
	   cout<<"Fail to initialization "<<endl;
	   return TEC_INVALID_PARAM;
	}

	nCount = 0;
	nCountFace = 0;
	allPredictTime = 0.0;
	/*****************************Process one by one*****************************/
	while(EOF != fscanf(fpListFile, "%s", loadImgPath))
	{
		strImageID = loadImgPath;
		sprintf(szImgPath, "%s/%s.jpg", szQueryPath, loadImgPath );
		IplImage *img = cvLoadImage(szImgPath);
		if(!img || (img->width<32) || (img->height<32) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
		{	
			cout<<"Can't open " << szImgPath << endl;
			cvReleaseImage(&img);img = 0;
			continue;
		}	

		/************************getRandomID*****************************/
		//api_commen.getRandomID( ImageID );
		//ImageID = api_commen.GetIDFromFilePath( loadImgPath );
		//strImageID = api_commen.GetStringIDFromFilePath( loadImgPath );

		/************************ResizeImg*****************************/
		//IplImage* imgResize = api_commen.ResizeImg( img, 320 );
		
		/************************Predict*****************************/	
		Res.clear();
		run.start();
		nRet = api_face_detect.Predict( img, Res );
		if ( (nRet!=0) || (Res.size()<1) )
		{
			LOOGE<<"[Predict Err!!loadImgPath:]"<<loadImgPath;
			//cvReleaseImage(&imgResize);imgResize = 0;
			cvReleaseImage(&img);img = 0;
			continue;
		}
		run.end();
		//LOOGI<<"[Predict] time:"<<run.time();
		allPredictTime += run.time();

		//save face label
		FDDB_Res.clear();
		for(i=0;i<Res.size();i++)  
		{
			if ( Res[i].label == "face" )
			{
				FDDB_Res.push_back( Res[i] );
			}
		}

		//write trainval.txt
		//if (FDDB_Res.size()>0)
		{
			fprintf(fpListFile_fddb, "%s\n", strImageID.c_str() );
			fprintf(fpListFile_fddb, "%d\n", FDDB_Res.size() );
			for(i=0;i<FDDB_Res.size();i++)  
				fprintf(fpListFile_fddb, "%d %d %d %d %.4f\n", FDDB_Res[i].rect[0], FDDB_Res[i].rect[1],
					FDDB_Res[i].rect[2]-FDDB_Res[i].rect[0], FDDB_Res[i].rect[3]-FDDB_Res[i].rect[1],
					FDDB_Res[i].score);
		}

		//replace word
		position = strImageID.find( "/" ); // find first period
		while ( position != string::npos )
		{
		  	strImageID.replace( position, 1, "_");
		  	position = strImageID.find( "/", position+1 );
		}

		/************************save img data*****************************/
		{
			name = Res[0].label;
			if ( (name=="face")||(name=="halfface")||(name=="eye")||(name=="mouse")||(name=="nose")||(name=="hair")||(name=="beard") )
				sprintf(tPath, "res_predict/face/", tPath );
			else
				sprintf(tPath, "res_predict/noface/", tPath );
		}
		for(i=0;i<Res.size();i++)  
		{				
			Scalar color = colors[i%8];
			cvRectangle( img, cvPoint(Res[i].rect[0], Res[i].rect[1]),
	                   cvPoint(Res[i].rect[2], Res[i].rect[3]), color, 2, 8, 0);

			sprintf(szImgPath, "%.2f %s", Res[i].score, Res[i].label.c_str() );
			text = szImgPath;
			cvPutText( img, text.c_str(), cvPoint(Res[i].rect[0]+1, Res[i].rect[1]+20), &font, color );

			//sprintf(tPath, "%s%s-%.2f_", tPath, Res[i].label.c_str(), Res[i].score );

			if ( Res[i].label == "face" )
				nCountFace++;
		}
		sprintf( savePath, "%s%s.jpg", tPath, strImageID.c_str() );
		cvSaveImage( savePath, img );
			
		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld img...,time:%.4f\n",nCount,run.time());
		
		//cvReleaseImage(&imgResize);imgResize = 0;
		cvReleaseImage(&img);img = 0;
	}

	/*********************************close file*************************************/
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	
	if (fpListFile_fddb) {fclose(fpListFile_fddb);fpListFile_fddb = 0;}	

	/*********************************Release*************************************/
	api_face_detect.Release();

	/*********************************Print Info*********************************/
	if ( nCount != 0 ) 
	{
		printf( "nCount:%ld,nCountFace:%ld_%.4f,PredictTime:%.4fms\n", 
			nCount, nCountFace, nCountFace*1.0/nCount, allPredictTime*1000.0/nCount );
	}
	
	cout<<"Done!! "<<endl;
	
	return nRet;	
}

int Check_Sencetime_fddb( char *loadAnnotations, char *szQueryList, char *szQueryPath )
{
	char tPath[256];
	char loadImgPath[256];
	char szImgPath[256];
	char savePath[256];
	vector<string> vecText;
	int label, svImg, nRet = 0;
	long inputLabel;
	string text,name;
	FILE *fpListFile = 0;

	string word,strImageID;
	long long nCount, QueryAllSize, idQuary, labelQuary, i, j, a, b, w2v_words;
	vector< string > vecQuaryString;
	int faceNum;
	char tmpChar;
	char vocab[4096];
	const long long max_w = 4096;

	RunTimer<double> run;
	API_COMMEN api_commen;

	vector<Sencetime_fddb_Info> vecSenseTimeFddb;
	map<string, vector<Sencetime_fddb_Info> > mapIDPath;
	map<string, vector<Sencetime_fddb_Info> >::iterator itIDPath;

	int position;

	/***********************************Init**********************************/
	plog::init(plog::info, "plog.txt"); 

	const static Scalar colors[] =  { 	CV_RGB(0,0,255),	CV_RGB(0,255,255),	CV_RGB(0,255,0),	
										CV_RGB(255,255,0),	CV_RGB(255,0,0),	CV_RGB(255,0,255),
										CV_RGB(0,128,255), 	CV_RGB(255,128,0)} ;
	CvFont font;
 	cvInitFont(&font,CV_FONT_HERSHEY_TRIPLEX,0.35f,0.7f,0,1,CV_AA);
	
	/********************************Open Query List*****************************/
	fpListFile = fopen(loadAnnotations,"r");
	if (!fpListFile) 
	{
		cout << "Can't open " << loadAnnotations << endl;
		return -1;
	}

	FILE *fpListFile_fddb = fopen("res_predict/res_fddb.txt","wt+");
	if (!fpListFile_fddb) 
	{
		cout << "0.can't open " << "res_predict/res_fddb.txt" << endl;
		return TEC_INVALID_PARAM;
	}

	FILE *fpQListFile = fopen(szQueryList,"r");
	if (!fpQListFile) 
	{
		cout << "0.can't open " << szQueryList << endl;
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
		
		//change name
		long ID = 0;
		int  atom =0;
		string tmpPath = vecQuaryString[0];
		long start = tmpPath.find_first_of('/');
		long end = tmpPath.find_last_of('.');
		if ( (start>0) && (end>0) && ( end>start ) )
			strImageID = tmpPath.substr(start+1,end-start-1);
		else
			continue;
		//printf("strImageID:%s\n",strImageID.c_str());

		//get face num
		faceNum = atoi(vecQuaryString[1].c_str());

		//get face info
		vecSenseTimeFddb.clear();
		for(i=0;i<faceNum;i++)
		{
			Sencetime_fddb_Info tmpSencetimeFddb;
			tmpSencetimeFddb.x1 = int(atof(vecQuaryString[2+2+i*220].c_str())+0.5);
			tmpSencetimeFddb.y1 = int(atof(vecQuaryString[2+1+i*220].c_str())+0.5);
			tmpSencetimeFddb.x2 = int(atof(vecQuaryString[2+3+i*220].c_str())+0.5);
			tmpSencetimeFddb.y2 = int(atof(vecQuaryString[2+i*220].c_str())+0.5);
			tmpSencetimeFddb.score = atof(vecQuaryString[2+4+i*220].c_str());
			vecSenseTimeFddb.push_back( tmpSencetimeFddb );
		}	

		//get iid
		mapIDPath[strImageID] = vecSenseTimeFddb;
			
		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld img...\n",nCount);
	}

	/*****************************imglist:Process one by one*****************************/
	printf("Loading imglist...\n");
	nCount = 0;
	while(EOF != fscanf(fpQListFile, "%s", &tPath ))
	{	
		//find iid
		strImageID = tPath;
		itIDPath = mapIDPath.find(strImageID);
		if (itIDPath == mapIDPath.end())
		{	
			continue;
		}	

		//load img
		sprintf( loadImgPath, "%s/%s.jpg", szQueryPath, strImageID.c_str() );
		IplImage *img = cvLoadImage(loadImgPath);
		if(!img || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
		{	
			cout<<"Can't open " << loadImgPath << endl;
			cvReleaseImage(&img);img = 0;
			continue;
		}

		//write trainval.txt
		fprintf(fpListFile_fddb, "%s\n", strImageID.c_str() );
		fprintf(fpListFile_fddb, "%d\n", itIDPath->second.size() );

		//replace word
		position = strImageID.find( "/" ); // find first period
		while ( position != string::npos )
		{
		  	strImageID.replace( position, 1, "_");
		  	position = strImageID.find( "/", position+1 );
		}

		/************************save img data*****************************/
		if ( itIDPath->second.size() > 0 )
			sprintf(tPath, "res_predict/face/" );
		else
			sprintf(tPath, "res_predict/noface/" );
		
		for(i=0;i<itIDPath->second.size();i++)
		{
			int x1 = itIDPath->second[i].x1;
			int y1 = itIDPath->second[i].y1;
			int x2 = itIDPath->second[i].x2;
			int y2 = itIDPath->second[i].y2;
			float score = itIDPath->second[i].score;
			fprintf(fpListFile_fddb, "%d %d %d %d %.4f\n", x1, y1,(x2-x1),(y2-y1),score);

			Scalar color = colors[i%8];
			cvRectangle( img, cvPoint(x1, y1), cvPoint(x2, y2), color, 2, 8, 0);

			sprintf(szImgPath, "%.2f", score );
			text = szImgPath;
			cvPutText( img, text.c_str(), cvPoint(x1+1, y1+20), &font, color );
		}
		sprintf( savePath, "%s/%s.jpg", tPath, strImageID.c_str() );
		cvSaveImage( savePath, img );
		cvReleaseImage(&img);img = 0;

		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld img...\n",nCount);
		
	}
	printf("End Loaded imglist!!\n");

	/*********************************close file*************************************/
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	
	if (fpQListFile) {fclose(fpQListFile);fpQListFile = 0;}
	if (fpListFile_fddb) {fclose(fpListFile_fddb);fpListFile_fddb = 0;}	

	/*********************************Print Info*********************************/
	if ( nCount != 0 ) 
	{
		printf( "nCount:%ld\n", nCount );
	}
	
	cout<<"Done!! "<<endl;
	
	return nRet;	
}

int Check_Sencetime_in( char *loadAnnotations, char *szQueryList, char *szQueryPath )
{
	char tPath[256];
	char loadImgPath[256];
	char szImgPath[256];
	char savePath[256];
	vector<string> vecText;
	int label, svImg, nRet = 0;
	long inputLabel;
	string text,name;
	FILE *fpListFile = 0;

	string word,strImageID;
	long long nCount, faceCount, QueryAllSize, idQuary, labelQuary, i, j, a, b, w2v_words;
	vector< string > vecQuaryString;
	int faceNum;
	char tmpChar;
	char vocab[4096];
	const long long max_w = 4096;

	RunTimer<double> run;
	API_COMMEN api_commen;

	vector<Sencetime_fddb_Info> vecSenseTimeFddb;
	map<string, vector<Sencetime_fddb_Info> > mapIDPath;
	map<string, vector<Sencetime_fddb_Info> >::iterator itIDPath;

	int position;

	/***********************************Init**********************************/
	plog::init(plog::info, "plog.txt"); 

	const static Scalar colors[] =  { 	CV_RGB(0,0,255),	CV_RGB(0,255,255),	CV_RGB(0,255,0),	
										CV_RGB(255,255,0),	CV_RGB(255,0,0),	CV_RGB(255,0,255),
										CV_RGB(0,128,255), 	CV_RGB(255,128,0)} ;
	CvFont font;
 	cvInitFont(&font,CV_FONT_HERSHEY_TRIPLEX,0.35f,0.7f,0,1,CV_AA);
	
	/********************************Open Query List*****************************/
	fpListFile = fopen(loadAnnotations,"r");
	if (!fpListFile) 
	{
		cout << "Can't open " << loadAnnotations << endl;
		return -1;
	}

	FILE *fpQListFile = fopen(szQueryList,"r");
	if (!fpQListFile) 
	{
		cout << "0.can't open " << szQueryList << endl;
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
		
		//change name
		long ID = 0;
		int  atom =0;
		string tmpPath = vecQuaryString[0];
		long start = tmpPath.find_first_of('/');
		long end = tmpPath.find_last_of('.');
		if ( (start>0) && (end>0) && ( end>start ) )
			strImageID = tmpPath.substr(start+1,end-start-1);
		else
			continue;
		//printf("strImageID:%s\n",strImageID.c_str());

		//get face num
		faceNum = atoi(vecQuaryString[1].c_str());

		//get face info
		vecSenseTimeFddb.clear();
		for(i=0;i<faceNum;i++)
		{
			Sencetime_fddb_Info tmpSencetimeFddb;
			tmpSencetimeFddb.x1 = int(atof(vecQuaryString[2+2+i*220].c_str())+0.5);
			tmpSencetimeFddb.y1 = int(atof(vecQuaryString[2+1+i*220].c_str())+0.5);
			tmpSencetimeFddb.x2 = int(atof(vecQuaryString[2+3+i*220].c_str())+0.5);
			tmpSencetimeFddb.y2 = int(atof(vecQuaryString[2+i*220].c_str())+0.5);
			tmpSencetimeFddb.score = atof(vecQuaryString[2+4+i*220].c_str());
			vecSenseTimeFddb.push_back( tmpSencetimeFddb );
		}	

		//get iid
		mapIDPath[strImageID] = vecSenseTimeFddb;
			
		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld img...\n",nCount);
	}

	/*****************************imglist:Process one by one*****************************/
	printf("Loading imglist...\n");
	nCount = 0;
	faceCount = 0;
	while(EOF != fscanf(fpQListFile, "%s", &tPath ))
	{	
		//find iid
		strImageID = api_commen.GetStringIDFromFilePath( tPath );
		itIDPath = mapIDPath.find(strImageID);
		if (itIDPath == mapIDPath.end())
		{	
			continue;
		}	

		//load img
		sprintf( loadImgPath, "%s/%s.jpg", szQueryPath, strImageID.c_str() );
		IplImage *img = cvLoadImage(loadImgPath);
		if(!img || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
		{	
			cout<<"Can't open " << loadImgPath << endl;
			cvReleaseImage(&img);img = 0;
			continue;
		}

		//add faceCount
		faceCount += itIDPath->second.size();

		/************************save img data*****************************/
		if ( itIDPath->second.size() > 0 )
			sprintf(tPath, "res_predict/face/" );
		else
			sprintf(tPath, "res_predict/noface/" );
		
		for(i=0;i<itIDPath->second.size();i++)
		{
			int x1 = itIDPath->second[i].x1;
			int y1 = itIDPath->second[i].y1;
			int x2 = itIDPath->second[i].x2;
			int y2 = itIDPath->second[i].y2;
			float score = itIDPath->second[i].score;

			Scalar color = colors[i%8];
			cvRectangle( img, cvPoint(x1, y1), cvPoint(x2, y2), color, 2, 8, 0);

			sprintf(szImgPath, "%.2f", score );
			text = szImgPath;
			cvPutText( img, text.c_str(), cvPoint(x1+1, y1+20), &font, color );
		}
		sprintf( savePath, "%s/%s.jpg", tPath, strImageID.c_str() );
		cvSaveImage( savePath, img );
		cvReleaseImage(&img);img = 0;

		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld img...\n",nCount);
		
	}
	printf("End Loaded imglist!!\n");

	/*********************************close file*************************************/
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	
	if (fpQListFile) {fclose(fpQListFile);fpQListFile = 0;}	

	/*********************************Print Info*********************************/
	if ( nCount != 0 ) 
	{
		printf( "nCount:%ld,faceCount:%ld\n", nCount, faceCount );
	}
	
	cout<<"Done!! "<<endl;
	
	return nRet;	
}


int main(int argc, char* argv[])
{
	int  ret = 0;
	char szKeyFiles[256],szSavePath[256];
	API_COMMEN api_commen;

	if (argc == 4 && strcmp(argv[1],"-change_name") == 0) {
		ret = Change_Name( argv[2], argv[3] );
	}
	else if (argc == 6 && strcmp(argv[1],"-get_img2xml") == 0) {
		ret = Get_Img2Xml( argv[2], argv[3], argv[4], argv[5] );
	}
	else if (argc == 6 && strcmp(argv[1],"-Get_CelebA_FaceLabel") == 0) {
		ret = Get_CelebA_FaceLabel( argv[2], argv[3], argv[4], argv[5] );
	}
	else if (argc == 6 && strcmp(argv[1],"-ch_xml_name") == 0) {
		ret = Change_XML_Name( argv[2], argv[3], argv[4], argv[5] );
	}
	else if (argc == 7 && strcmp(argv[1],"-Get_wider_FaceLabel") == 0) {
		ret = Get_wider_FaceLabel( argv[2], argv[3], argv[4], argv[5], argv[6] );
	}
	else if (argc == 6 && strcmp(argv[1],"-check_sencetime") == 0) {
		ret = Check_SenceTimeData( argv[2], argv[3], argv[4], argv[5] );
	}
	else if (argc == 7 && strcmp(argv[1],"-reget_sencetime") == 0) {
		ret = ReGet_SenceTimeData( argv[2], argv[3], argv[4], argv[5], argv[6] );
	}
	else if (argc == 7 && strcmp(argv[1],"-frcnn") == 0) {
		ret = frcnn_test( argv[2], argv[3], argv[4], atol(argv[5]), atol(argv[6]) );
	}
	else if (argc == 7 && strcmp(argv[1],"-frcnn_shuying") == 0) {
		ret = frcnn_test_shuying( argv[2], argv[3], argv[4], atol(argv[5]), atol(argv[6]) );
	}
	else if (argc == 7 && strcmp(argv[1],"-frcnn_addsample") == 0) {
		ret = frcnn_addSample( argv[2], argv[3], argv[4], atol(argv[5]), atol(argv[6]) );
	}
	else if (argc == 8 && strcmp(argv[1],"-fddb") == 0) {
		ret = test_fddb( argv[2], argv[3], argv[4], argv[5], atol(argv[6]), atol(argv[7]) );
	}
	else if (argc == 5 && strcmp(argv[1],"-fddb_sencetime") == 0) {
		ret = Check_Sencetime_fddb( argv[2], argv[3], argv[4] );
	}
	else if (argc == 5 && strcmp(argv[1],"-sencetime_in") == 0) {
		ret = Check_Sencetime_in( argv[2], argv[3], argv[4] );
	}
	else
	{
		cout << "usage:\n" << endl;
		cout << "\tDemo_facedetect -change_name queryList.txt imgPath\n" << endl;
		cout << "\tDemo_facedetect -get_img2xml queryList.txt xmlSavePath imgSavePath labelname\n" << endl;
		cout << "\tDemo_facedetect -Get_CelebA_FaceLabel imgPath loadAnnotations xmlSavePath imgSavePath\n" << endl;
		cout << "\tDemo_facedetect -ch_xml_name queryList.txt imgPath xmlSavePath imgSavePath\n" << endl;
		cout << "\tDemo_facedetect -Get_wider_FaceLabel queryList.txt loadAnnotations xmlSavePath srcImgSavePath imgSavePath\n" << endl;
		cout << "\tDemo_facedetect -check_sencetime imgPath loadAnnotations xmlSavePath imgSavePath\n" << endl;
		cout << "\tDemo_facedetect -reget_sencetime queryList.txt imgPath loadAnnotations xmlSavePath imgSavePath\n" << endl;
		cout << "\tDemo_facedetect -frcnn queryList.txt keyFilePath layerName binGPU deviceID\n" << endl;
		cout << "\tDemo_facedetect -frcnn_shuying queryList.txt keyFilePath layerName binGPU deviceID\n" << endl;
		cout << "\tDemo_facedetect -frcnn_addsample queryList.txt keyFilePath layerName binGPU deviceID\n" << endl;
		cout << "\tDemo_facedetect -fddb queryList.txt queryPath keyFilePath layerName binGPU deviceID\n" << endl;
		cout << "\tDemo_facedetect -fddb_sencetime loadAnnotations szQueryList queryPath\n" << endl;
		cout << "\tDemo_facedetect -sencetime_in loadAnnotations szQueryList queryPath\n" << endl;
		return ret;
	}
	return ret;
}
