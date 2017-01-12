#include <string>
#include <vector>
#include <stdio.h>
#include <stdlib.h>		//do shell
#include <sys/stat.h> 
#include <sys/types.h> 
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <map>

#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>

#include "API_commen/API_commen.h"
#include "API_mutilabel/v2.0.0/API_mutilabel.h"
#include "API_commen/TErrorCode.h"
#include "plog/Log.h"
#include "API_xml/API_xml.h"	//read xml

using namespace cv;
using namespace std;

#define VOC_LABEL_NUM 20
#define COCO_LABEL_NUM 80
#define OLDIN_LABEL_NUM 47
#define Filter_LABEL_NUM 15

struct Patch_Info
{
	string 			label;
    float 			score;
    Vec4i 			rect;
};

//inLabelClass:0-voc,1-coco,2-old in;
int Get_DataCH2InMutiLabel( char *loadXMLPath, char *loadImagePath, char *inDict,  char *outDict, 
		char* svXmlPath, char* svImgPath, int inLabelClass )
{
	char tPath[256];
	char loadImgPath[256];
	char loadXmlPath[256];
	char szImgPath[256];
	int i, j, svImg, bin_write, bin_write_full, nRet = 0;
	int Min_Size = 20;
	unsigned long long nCount, labelCount, fileterCount;
	
	API_COMMEN api_commen;
	API_XML api_xml;

	/***********************************dic2dic index**********************************/
	const int vocName[VOC_LABEL_NUM] = {
		52,53, 0,54,35,55,56, 1,27, 2,	30, 3, 4,57,50,28, 7,29,58,13};
	const int cocoName[COCO_LABEL_NUM] = {
		50,53,56,57,52,55,58,59,54,64,	48,48,48,27, 0, 1, 3, 4, 7, 2,
		 8, 8, 8, 8,32,48,32,48,40,48,	48,48,33,48,48,48,48,48,48,35,
		39,39,48,48,48,25,20,20,25,20,	25,25,25,23,25,17,27,29,28,26,
		30,31,13,14,11,15,12,10,15,15,	15,15,15,34,36,35,48,46,15,48};
	const int oldInName[OLDIN_LABEL_NUM] = {
		16,17,19,20,21,22,25,52,32,35,	 9,56,58,37,38,41,42,43,45,38,
		51,54,36,47,48,51,50,51,50,51,	50,50,50,50, 1, 3, 8,64,64,64,
		64,64,64,64,62,63,60};
	const string labelFilter[Filter_LABEL_NUM] = { 
		"animal.cat", "animal.dog", 
		"electronics.monitor", 
		"furniture.pottedplant", "furniture.sofa", "furniture.table", 
		"goods.cup", "goods.flower",
		"person.body", 
		"vehicle.airplane", "vehicle.bicycle", "vehicle.boat", "vehicle.bus", "vehicle.car", 
		"vehicle.motorbike" };

	/***********************************Load in dic File**********************************/
	vector< string > inDicString;
	sprintf(tPath, "%s",inDict);
	api_commen.loadWordDict(tPath,inDicString);
	printf( "load indic:%s,dict:size-%d\n", tPath, int(inDicString.size()) );

	/***********************************Load out dic File**********************************/
	vector< string > outDicString;
	sprintf(tPath, "%s",outDict);
	api_commen.loadWordDict(tPath,outDicString);
	printf( "load target dic:%s,dict:size-%d\n", tPath, int(outDicString.size()) );

	//Init
	string label;
	string tmpPath,iid,strID, ImageID;
	long end = 0;

	map< string, string > 				mapInName;
	map< string, string >::iterator 	itInName;
	vector < pair < string,Vec4i > > 	vecInLabelRect;
	vector < pair < string,Vec4i > > 	vecOutLabelRect;

	//write voc_in_name
	for(i=0;i<inDicString.size();i++)
	{
		if ( inLabelClass == 2 )
			mapInName[inDicString[i]] = outDicString[oldInName[i]];	//old in
		else if ( inLabelClass == 1 )
			mapInName[inDicString[i]] = outDicString[cocoName[i]];	//coco
		else
			mapInName[inDicString[i]] = outDicString[vocName[i]];	//voc
	}
	
	//Open Query List
	FILE *fpListFile = fopen(loadXMLPath,"r");
	if (!fpListFile) 
	{
		cout << "0.can't open " << loadXMLPath << endl;
		return TEC_INVALID_PARAM;
	}

	FILE *fpListFile_trainval = fopen("res/trainval.txt","wt+");
	if (!fpListFile_trainval) 
	{
		cout << "0.can't open " << "res/trainval.txt" << endl;
		return TEC_INVALID_PARAM;
	}

	//Process one by one
	nCount = 0;
	while(EOF != fscanf(fpListFile, "%s", &loadXmlPath))
	{
		/************************getRandomID*****************************/
		strID = api_commen.GetStringIDFromFilePath( loadXmlPath );

		//check exit img
		sprintf( loadImgPath, "%s/%s.jpg", loadImagePath, strID.c_str() );
		IplImage *img = cvLoadImage(loadImgPath);
		if(!img || (img->width<32) || (img->height<32) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
		{	
			cout<<"Can't open " << loadImgPath << endl;
			cvReleaseImage(&img);img = 0;
			continue;
		}	
		cvReleaseImage(&img);img = 0;
		
		//read_xml
		vecInLabelRect.clear();
		nRet = api_xml.load_xml( string(loadXmlPath), ImageID, vecInLabelRect );
		if ( (nRet!=0) || (vecInLabelRect.size()<1) )
		{
			cout << "Err to load_xml!" << endl;
			continue;
		}

		//change data
		fileterCount = 0;
		vecOutLabelRect.clear();
		for(i=0;i<vecInLabelRect.size();i++)
		{
			/************************find label*****************************/
			itInName = mapInName.find( vecInLabelRect[i].first );
			if (itInName == mapInName.end())
				continue;

			Vec4i tmpVec4i = vecInLabelRect[i].second;
			if ( ((tmpVec4i[2]-tmpVec4i[0])<Min_Size) || ((tmpVec4i[3]-tmpVec4i[1])<Min_Size) )
				continue;

			//filter label
			for(j=0;j<Filter_LABEL_NUM;j++)
			{
				if (itInName->second == labelFilter[j])
					fileterCount++;

				if (itInName->second == "person.body")
				{
					fileterCount = vecInLabelRect.size();
					break;
				}
			}

			vecOutLabelRect.push_back( std::make_pair( itInName->second, vecInLabelRect[i].second ) );
		}

		if ((vecOutLabelRect.size()<1) || (fileterCount>0.4*vecOutLabelRect.size()))
			continue;
		
		//write_xml
		nRet = api_xml.write_xml( ImageID, svXmlPath, vecOutLabelRect );
		if (nRet!=0)
		{
			cout << "Err to write_xml!" << endl;
			continue;
		}

		//cp img file
		sprintf( tPath, "cp %s/%s.jpg %s/%s.jpg", loadImagePath, ImageID.c_str(), svImgPath, ImageID.c_str() );
		system( tPath );

		//write trainval.txt
		fprintf(fpListFile_trainval, "%s\n", ImageID.c_str() );

		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld xml...\n",nCount);
	}
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	
	if (fpListFile_trainval) {fclose(fpListFile_trainval);fpListFile_trainval = 0;}	

	//print info
	printf("All load img:%lld!!\n", nCount );

	cout<<"Done!! "<<endl;
	
	return nRet;
}

int Change_label( char *loadXMLPath, char *loadImagePath, char* svPath )
{
	char tPath[256];
	char cmd[256];
	char loadImgPath[256];
	char loadXmlPath[256];
	char szImgPath[256];
	int i, j, svImg, bin_write, bin_write_full, nRet = 0;
	int Min_Size = 20;
	unsigned long long nCount, labelCount;
	
	API_COMMEN api_commen;
	API_XML api_xml;

	//Init
	string label;
	string tmpPath,iid,strID, ImageID;
	long end = 0;

	map< string, long >					mapCount;
	map< string, long >::iterator 		itCount;
	vector < pair < string,Vec4i > > 	vecInLabelRect;
	vector < pair < string,Vec4i > > 	vecOutLabelRect;
	
	/***********************************Open Query List**********************************/
	FILE *fpListFile = fopen(loadXMLPath,"r");
	if (!fpListFile) 
	{
		cout << "0.can't open " << loadXMLPath << endl;
		return TEC_INVALID_PARAM;
	}

	//Process one by one
	nCount = 0;
	while(EOF != fscanf(fpListFile, "%s", &loadXmlPath))
	{
		vecOutLabelRect.clear();
		/************************getRandomID*****************************/
		strID = api_commen.GetStringIDFromFilePath( loadXmlPath );

		//check exit img
		sprintf( loadImgPath, "%s/%s.jpg", loadImagePath, strID.c_str() );
		IplImage *img = cvLoadImage(loadImgPath);
		if(!img || (img->width<32) || (img->height<32) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
		{	
			cout<<"Can't open " << loadImgPath << endl;
			cvReleaseImage(&img);img = 0;
			continue;
		}	
		cvReleaseImage(&img);img = 0;
		
		//read_xml
		vecInLabelRect.clear();
		nRet = api_xml.load_xml( string(loadXmlPath), ImageID, vecInLabelRect );
		if ( (nRet!=0) || (vecInLabelRect.size()<1) )
		{
			cout << "Err to load_xml!" << endl;
			continue;
		}

		//remove label<T image
		if ( vecInLabelRect.size()<1)
			continue;

		/************************find label*****************************/
		int tmp_sv = 0;
		for(i=0;i<vecInLabelRect.size();i++)
		{		
			if (vecInLabelRect[i].first == "furniture.diningtable")
			{
				vecOutLabelRect.push_back( std::make_pair( "furniture.table", vecInLabelRect[i].second ) );
				tmp_sv = 1;
			}
			else if (vecInLabelRect[i].first == "None")
			{
				vecOutLabelRect.push_back( std::make_pair( "other.other", vecInLabelRect[i].second ) );
				tmp_sv = 1;
			}
			else if (vecInLabelRect[i].first == "cat")
			{
				vecOutLabelRect.push_back( std::make_pair( "animal.cat", vecInLabelRect[i].second ) );
				tmp_sv = 1;
			}
			else if (vecInLabelRect[i].first == "dog")
			{
				vecOutLabelRect.push_back( std::make_pair( "animal.dog", vecInLabelRect[i].second ) );
				tmp_sv = 1;
			}
			else if (vecInLabelRect[i].first == "person.other")
			{
				tmp_sv = 1;
				continue;
			}
			else
				vecOutLabelRect.push_back( std::make_pair( vecInLabelRect[i].first, vecInLabelRect[i].second ) );
		}

		if ( tmp_sv == 1 )
		{
			if (vecOutLabelRect.size() < 1)
			{
				sprintf(cmd, "rm %s", loadXmlPath );
				system(cmd);
				sprintf(cmd, "rm %s", loadImgPath );
				system(cmd);
				continue;
			}
			
			//write_xml
			sprintf( tPath, "%s/Annotations/", svPath );
			nRet = api_xml.write_xml( ImageID, tPath, vecOutLabelRect );
			if (nRet!=0)
			{
				cout << "Err to write_xml!" << endl;
				continue;
			}
		}

		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld xml...\n",nCount);
	}
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	

	//print info
	printf("All load Img_%lld\n", nCount );

	cout<<"Done!! "<<endl;
	
	return nRet;
}

int Change_voclabel4test( char *loadXMLPath, char *svPath )
{
	char loadXmlPath[256];
	int i, j, nRet = 0;
	unsigned long long nCount;

	API_XML api_xml;
	map<string, string> chLabel;

	/***********************************dic2dic index**********************************/
	const string vocName[20] = {
		"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", 
		"diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};

	const string vocCHName[20] = {
		"vehicle.airplane", "vehicle.bicycle", "animal.bird", "vehicle.boat", "goods.bottle", 
		"vehicle.bus", "vehicle.car", "animal.cat", "furniture.chair", "animal.cow", 
		"furniture.table", "animal.dog", "animal.horse", "vehicle.motorbike", "person.body", 
		"furniture.pottedplant", "animal.sheep", "furniture.sofa", "vehicle.train", "electronics.monitor"};

	for(i=0;i<20;i++)
		chLabel[vocName[i]] = vocCHName[i];
	
	/***********************************Open Query List**********************************/
	FILE *fpListFile = fopen(loadXMLPath,"r");
	if (!fpListFile) 
	{
		cout << "0.can't open " << loadXMLPath << endl;
		return TEC_INVALID_PARAM;
	}

	//Process one by one
	nCount = 0;
	while(EOF != fscanf(fpListFile, "%s", &loadXmlPath))
	{	
		nRet = api_xml.change_label(string(loadXmlPath), chLabel, string(svPath) );
		if (nRet!=0)
		{
			cout << "Err to change_label!" << endl;
			continue;
		}

		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld xml...\n",nCount);
	}
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	

	//print info
	printf("All load XML_%lld\n", nCount );

	cout<<"Done!! "<<endl;
	
	return nRet;
}

int Count_Data( char *loadXMLPath, char *loadImagePath, char* svPath, int svImgXml )
{
	char tPath[2560];
	char loadImgPath[256];
	char loadXmlPath[256];
	char szImgPath[256];
	int i, j, svImg, bin_write, bin_write_full, IsExist, nRet = 0;
	int Min_Size = 20;
	unsigned long long nCount, nCount_trainval, nCount_test, labelCount;
	
	API_COMMEN api_commen;
	API_XML api_xml;

	//Init
	string label;
	string tmpPath,iid,strID, ImageID;
	long end = 0;

	map< string, long >					mapCount;
	map< string, long >::iterator 		itCount;
	vector < pair < string,Vec4i > > 	vecInLabelRect;
	vector < pair < string,Vec4i > > 	vecOutLabelRect;
	
	/***********************************Open Query List**********************************/
	FILE *fpListFile = fopen(loadXMLPath,"r");
	if (!fpListFile) 
	{
		cout << "0.can't open " << loadXMLPath << endl;
		return TEC_INVALID_PARAM;
	}

	/***********************************Save File**********************************/
	sprintf( tPath, "%s/trainval.txt", svPath );
	FILE *fpListFile_trainval = fopen(tPath,"w");
	if (!fpListFile_trainval) 
	{
		cout << "0.can't open " << tPath << endl;
		return TEC_INVALID_PARAM;
	}

	sprintf( tPath, "%s/test.txt", svPath );
	FILE *fpListFile_test = fopen(tPath,"w");
	if (!fpListFile_test) 
	{
		cout << "0.can't open " << tPath << endl;
		return TEC_INVALID_PARAM;
	}

	//Process one by one
	nCount = 0;
	nCount_test = 0;
	nCount_trainval = 0; 
	while(EOF != fscanf(fpListFile, "%s", &loadXmlPath))
	{
		/************************getRandomID*****************************/
		strID = api_commen.GetStringIDFromFilePath( loadXmlPath );

		//check exit img
		sprintf( loadImgPath, "%s/%s.jpg", loadImagePath, strID.c_str() );
		IplImage *img = cvLoadImage(loadImgPath);
		if(!img || (img->width<32) || (img->height<32) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
		{	
			cout<<"Can't open " << loadImgPath << endl;
			cvReleaseImage(&img);img = 0;
			continue;
		}	
		cvReleaseImage(&img);img = 0;
		
		//read_xml
		vecInLabelRect.clear();
		nRet = api_xml.load_xml( string(loadXmlPath), ImageID, vecInLabelRect );
		if ( (nRet!=0) || (vecInLabelRect.size()<1) )
		{
			cout << "Err to load_xml!" << endl;
			continue;
		}

		IsExist = 0;
		nRet = api_xml.find_name( string(loadXmlPath), "size", ImageID, IsExist );
		if (nRet!=0)
		{
			cout << "Err to find_name!" << endl;
			continue;
		}

		//remove label<T image
		if ( vecInLabelRect.size()<1)
			continue;

		/************************find label*****************************/
		//int t_sv = 1;
		for(i=0;i<vecInLabelRect.size();i++)
		{		
			//remove not need labels:food.fruit/animal.cat/animal.dog
			//if ( (vecInLabelRect[i].first == "animal.cat") || (vecInLabelRect[i].first == "animal.dog") 
			//	 || (vecInLabelRect[i].first == "other.sticker") )
			//{
			//	t_sv = 0;
			//	break;
			//}
			
			itCount = mapCount.find( vecInLabelRect[i].first );
			if (itCount == mapCount.end())
				mapCount[vecInLabelRect[i].first] = 1;
			else
				mapCount[vecInLabelRect[i].first]++;
		}

		//remove not need labels
		//if (t_sv == 0)
		//{
			//change file name 
		//	sprintf( tPath, "rm %s;rm %s", loadImgPath, loadXmlPath );
		//	system( tPath );
		
		//	continue;
		//}

		//write trainval.txt
		if ( ((nCount+1)%10 == 0) && (IsExist == 1) )
		{
			nCount_test++;
			fprintf(fpListFile_test, "%s\n", ImageID.c_str() );	//for test-faster rcnn
		}
		else
		{
			nCount_trainval++; 
			fprintf(fpListFile_trainval, "%s\n", ImageID.c_str() );//for train
		}

		if ( svImgXml == 1 )
		{
			//cp img file
			sprintf( tPath, "cp %s %s/JPEGImages/", loadImgPath, svPath );
			system( tPath );

			//cp xml file
			sprintf( tPath, "cp %s %s/Annotations/", loadXmlPath, svPath );
			system( tPath );
		}

		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld xml...\n",nCount);
	}
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	
	if (fpListFile_test) {fclose(fpListFile_test);fpListFile_test = 0;}
	if (fpListFile_trainval) {fclose(fpListFile_trainval);fpListFile_trainval = 0;}	

	/*********************************write label file*********************************/
	sprintf( tPath, "%s/Dict_FRCNN_%d_label.txt", svPath, mapCount.size() );
	FILE *fpListFile_training = fopen(tPath,"w");
	if (!fpListFile_training) 
	{
		cout << "0.can't open " << tPath << endl;
		return TEC_INVALID_PARAM;
	}

	labelCount = 0;
	fprintf(fpListFile_training, "Categories Size_%d:\n", mapCount.size() );
	for(itCount = mapCount.begin(); itCount != mapCount.end(); itCount++)
	{
		fprintf(fpListFile_training, "%s\n", itCount->first.c_str());

		labelCount += itCount->second;
	}
	fprintf(fpListFile_training, "\n\n");
	
	fprintf(fpListFile_training, "Categories for Python:\n");
	long tmpCount = 0;
	for(itCount = mapCount.begin(); itCount != mapCount.end(); itCount++)
	{
		fprintf(fpListFile_training, "'%s', ", itCount->first.c_str());
		tmpCount++;
		if ( (tmpCount%5) == 0 )
			fprintf(fpListFile_training, "\n");
	}
	fprintf(fpListFile_training, "\n\n\n");

	fprintf(fpListFile_training, "Categories Img_%lld, Label_%lld, Label/Img_%.2f\n",
		nCount,labelCount,labelCount*1.0/nCount);
	for(itCount = mapCount.begin(); itCount != mapCount.end(); itCount++)
	{
		fprintf(fpListFile_training, "%s %lld %.4f\n", itCount->first.c_str(), itCount->second, 
			itCount->second*1.0/labelCount);
	}
	fprintf(fpListFile_training, "\n\n");
	if (fpListFile_training) {fclose(fpListFile_training);fpListFile_training = 0;}	

	//change file name 
	sprintf( tPath, "mv %s/trainval.txt %s/trainval_%lld.txt", svPath, svPath, nCount_trainval );
	system( tPath );
	sprintf( tPath, "mv %s/test.txt %s/test_%lld.txt", svPath, svPath, nCount_test );
	system( tPath );

	//print info
	printf("All load Img_%lld, Label_%lld, Label/Img_%.2f\n", nCount,labelCount,labelCount*1.0/nCount );
	printf("trainval_%lld, test_%lld, rio_%.2f\n", nCount_trainval,nCount_test,nCount_test*1.0/(nCount_trainval+nCount_test) );

	cout<<"Done!! "<<endl;
	
	return nRet;
}

int Count_Data_shengdan( char *loadXMLPath, char *loadImagePath, char* svPath, int svImgXml, int svROI )
{
	char tPath[25600];
	char loadImgPath[256];
	char loadXmlPath[256];
	char szImgPath[256];
	int i, j, svImg, bin_write, bin_write_full, IsExist, nRet = 0;
	int Min_Size = 20;
	unsigned long long nCount, labelCount;
	
	API_COMMEN api_commen;
	API_XML api_xml;

	//Init
	string label;
	string tmpPath,iid,strID, ImageID;
	long end = 0;

	vector < pair < string,Vec4i > > 	vecInLabelRect;
	vector < pair < string,Vec4i > > 	vecOutLabelRect;
	
	/***********************************Open Query List**********************************/
	FILE *fpListFile = fopen(loadXMLPath,"r");
	if (!fpListFile) 
	{
		cout << "0.can't open " << loadXMLPath << endl;
		return TEC_INVALID_PARAM;
	}

	/***********************************Save File**********************************/
	sprintf( tPath, "%s/annotation.txt", svPath );
	FILE *fpAnnotation = fopen(tPath,"w");
	if (!fpAnnotation) 
	{
		cout << "0.can't open " << tPath << endl;
		return TEC_INVALID_PARAM;
	}

	//Process one by one
	nCount = 0;
	labelCount = 0;
	while(EOF != fscanf(fpListFile, "%s", &loadXmlPath))
	{
		/************************getRandomID*****************************/
		strID = api_commen.GetStringIDFromFilePath( loadXmlPath );

		//check exit img
		sprintf( loadImgPath, "%s/%s.jpg", loadImagePath, strID.c_str() );
		IplImage *img = cvLoadImage(loadImgPath);
		if(!img || (img->width<32) || (img->height<32) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
		{	
			cout<<"Can't open " << loadImgPath << endl;
			cvReleaseImage(&img);img = 0;
			continue;
		}	
		
		
		//read_xml
		vecInLabelRect.clear();
		nRet = api_xml.load_xml( string(loadXmlPath), ImageID, vecInLabelRect );
		if ( (nRet!=0) || (vecInLabelRect.size()<1) )
		{
			cout << "Err to load_xml!" << endl;
			cvReleaseImage(&img);img = 0;
			continue;
		}

		IsExist = 0;
		nRet = api_xml.find_name( string(loadXmlPath), "size", ImageID, IsExist );
		if (nRet!=0)
		{
			cout << "Err to find_name!" << endl;
			cvReleaseImage(&img);img = 0;
			continue;
		}

		//remove label<T image
		if ( vecInLabelRect.size()<1)
		{
			cvReleaseImage(&img);img = 0;
			continue;
		}

		/************************find label*****************************/
		vecOutLabelRect.clear();
		for(i=0;i<vecInLabelRect.size();i++)
		{		
			//get need labels:food.fruit/animal.cat/animal.dog
			if (vecInLabelRect[i].first == "animal.dog")
			{
				vecOutLabelRect.push_back( std::make_pair( vecInLabelRect[i].first, vecInLabelRect[i].second ) );

				if (svROI==1)
				{
					cvSetImageROI( img,cvRect(vecOutLabelRect[i].second[0], vecOutLabelRect[i].second[1], 
						(vecOutLabelRect[i].second[2]-vecOutLabelRect[i].second[0]),
						(vecOutLabelRect[i].second[3]-vecOutLabelRect[i].second[1])) );
					IplImage* roi = cvCreateImage(cvGetSize(img),img->depth,img->nChannels);
	                cvCopy( img, roi, NULL );
	                cvResetImageROI(img);
					sprintf( szImgPath, "%s/roi/%s_%d.jpg", svPath, ImageID.c_str(), i );
					cvSaveImage( szImgPath, roi );
					cvReleaseImage(&roi);roi = 0;
				}
			}
		}

		if ( vecOutLabelRect.size()<1)
		{
			cvReleaseImage(&img);img = 0;
			continue;
		}
		
		sprintf( tPath, "%s;%d;", strID.c_str(),vecOutLabelRect.size() );
		for(i=0;i<vecOutLabelRect.size();i++)
			sprintf( tPath, "%s%d,%d,%d,%d;", tPath,vecOutLabelRect[i].second[0],vecOutLabelRect[i].second[1],
							vecOutLabelRect[i].second[2],vecOutLabelRect[i].second[3]);
		fprintf(fpAnnotation, "%s\n", tPath );

		if ( svImgXml == 1 )
		{
			//cp img file
			sprintf( tPath, "cp %s %s/JPEGImages/", loadImgPath, svPath );
			system( tPath );

			//cp xml file
			sprintf( tPath, "cp %s %s/Annotations/", loadXmlPath, svPath );
			system( tPath );
		}

		cvReleaseImage(&img);img = 0;

		nCount++;
		labelCount += vecOutLabelRect.size();
		if( nCount%50 == 0 )
			printf("Loaded %ld xml...\n",nCount);
	}
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	
	if (fpAnnotation) {fclose(fpAnnotation);fpAnnotation = 0;}	

	printf("All load Img_%lld, Label_%lld, Label/Img_%.2f\n", nCount, labelCount, labelCount*1.0/nCount );
	cout<<"Done!! "<<endl;
	
	return nRet;
}


int frcnn_ReSample( char *szQueryList, char* svPath, char* KeyFilePath, float MutiLabel_T, int binGPU, int deviceID )
{
	char tPath[256];
	char loadImgPath[256];
	char szImgPath[256];
	char savePath[256];
	vector<string> vecText;
	int i, j, label, svImg, nRet = 0;
	long inputLabel, nCount, nCountObj;
	string strImageID,text,name;
	double allPredictTime;
	FILE *fpListFile = 0;

	RunTimer<double> run;
	API_COMMEN api_commen;
	API_MUTI_LABEL api_muti_label;
	API_XML api_xml;
	vector < pair < string,Vec4i > > 	vecOutLabelRect;

	vector< MutiLabelInfo > Res;

	/***********************************Init**********************************/
	unsigned long long		imageID;				//[In]:image ID for CheckData
	unsigned long long		childID;				//[In]:image child ID for CheckData
	sprintf( tPath, "res/plog.log" );
	plog::init(plog::error, tPath); 
	sprintf( tPath, "res/module-in-logo-detection.log" );
	plog::init<enum_module_in_logo>(plog::info, tPath, 100000000, 100000);

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
	nRet = api_muti_label.Init( KeyFilePath, binGPU, deviceID ); 
	if (nRet != 0)
	{
	   cout<<"Fail to initialization "<<endl;
	   return TEC_INVALID_PARAM;
	}

	nCount = 0;
	nCountObj = 0;
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
		sprintf( tPath, "%d_%d", nCount, nCountObj );
		nRet = api_muti_label.Predict( img, string(tPath), nCount, nCountObj, MutiLabel_T, Res );
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
		vecOutLabelRect.clear();
		for(i=0;i<Res.size();i++)  
		{			
			//person.body/goods.puppet/animal.cat/animal.dog
			if (Res[i].label == "animal.dog")
				svImg = 1;
			
			vecOutLabelRect.push_back( std::make_pair( Res[i].label, Res[i].rect ) );
			
			Scalar color = colors[i%8];
			cvRectangle( img, cvPoint(Res[i].rect[0], Res[i].rect[1]),
	                   cvPoint(Res[i].rect[2], Res[i].rect[3]), color, 2, 8, 0);

			sprintf(szImgPath, "%.2f %s", Res[i].score, Res[i].label.c_str() );
			text = szImgPath;
			cvPutText( img, text.c_str(), cvPoint(Res[i].rect[0]+1, Res[i].rect[1]+20), &font, color );
		}

		//do not save image
		if (svImg == 0 )
		{
			cvReleaseImage(&img);img = 0;
			continue;
		}

		//save checkimg
		sprintf( savePath, "%s/CheckImg/%s.jpg", svPath, strImageID.c_str() );
		cvSaveImage( savePath, img );

		//write_xml
		sprintf( tPath, "%s/Annotations/", svPath );
		nRet = api_xml.write_xml( strImageID, tPath, vecOutLabelRect );
		if (nRet!=0)
		{
			cout << "Err to write_xml!" << endl;
			cvReleaseImage(&img);img = 0;
			continue;
		}

		//cp img file
		sprintf( tPath, "cp %s %s/JPEGImages/", loadImgPath, svPath );
		system( tPath );

		nCountObj += Res.size();
		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld img...,time:%.4f\n",nCount,run.time());
		
		cvReleaseImage(&img);img = 0;
	}

	/*********************************close file*************************************/
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	

	/*********************************Release*************************************/
	api_muti_label.Release();

	/*********************************Print Info*********************************/
	if ( nCount != 0 ) 
	{
		printf( "nCount:%ld,nCountObj:%ld_%.4f,PredictTime:%.4fms\n", 
			nCount, nCountObj, nCountObj*1.0/nCount, allPredictTime*1000.0/nCount );
	}
	
	cout<<"Done!! "<<endl;
	
	return nRet;	
}

int cut_word(string input, char cutWord, string &formal, string &later)
{
	string strImageID;
	
	long end = input.find_first_of(cutWord);
	if (end>0)
	{
		formal = input.substr(0,end-1);
		later = input.substr(end+1,input.length());
	}
	else
		return -1;

	return 0;
}


//get list-->download img-->save res;
int frcnn_Check_OnlineTest( char *szQueryList, char* svPath, int svNum_T )
{
	char tPath[256];
	char charImageID[256];
	char szImgPath[256];
	char savePath[256];
	vector<string> vecText;
	int i, j, label, svImg, nRet = 0;
	long inputLabel, nCount, nCountObj, objEnough;
	string strImageID,text,name;
	double allPredictTime;
	FILE *fpListFile = 0;

	API_COMMEN api_commen;
	API_XML api_xml;

	vector < pair < string,Vec4i > > 	vecOutLabelRect;
	map< string, long >					mapCount;
	map< string, long >::iterator 		itCount;

	char tmpChar;
	char vocab[409600];
	string word;
	const long long max_w = 409600;
	vector< string > vecQuaryString;

	/***********************************Init**********************************/
	unsigned long long		imageIDLast;			//[In]:image ID for CheckData
	unsigned long long		imageID;				//[In]:image ID for CheckData
	unsigned long long		childID;				//[In]:image child ID for CheckData
	string					url;					//[In]:image url for CheckData
	int						objNum;

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
	nCountObj = 0;
	allPredictTime = 0.0;

	/*****************************Load Query*****************************/
	//fscanf(fpListFile, "%lld", &w2v_words);
	long CountLines = api_commen.doc2vec_CountLines(szQueryList);
	printf("Load Query.txt...%lld...\n",CountLines);
	for (long b = 0; b < CountLines; b++) {
		//load word to vector
		long a = 0;
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
		svImg = 0;
		imageID = atol(vecQuaryString[5].c_str());
		childID = atol(vecQuaryString[6].c_str());
		url = vecQuaryString[7];
		objNum = atoi(vecQuaryString[8].c_str());
		//printf("imageID:%lld,childID:%lld,url:%s,objNum:%d\n",imageID,childID,url.c_str(),objNum);

		//get high value image
		if ( ( objNum<2 ) || (imageIDLast == imageID) )
			continue;
		
		for(i=0;i<objNum;i++)
		{
		    Vec4i rect;
			string label = vecQuaryString[9+i*6];
			float score = atof(vecQuaryString[10+i*6].c_str());
			rect[0] = atoi(vecQuaryString[11+i*6].c_str());
			rect[1] = atoi(vecQuaryString[12+i*6].c_str());
			rect[2] = atoi(vecQuaryString[13+i*6].c_str());
			rect[3] = atoi(vecQuaryString[14+i*6].c_str());

			if (score<0.8)
				continue;

			/************************find label*****************************/		
			itCount = mapCount.find( label );
			if (itCount == mapCount.end())
				mapCount[label] = 1;
			else if ( (itCount != mapCount.end()) && (itCount->second<svNum_T) )
				mapCount[label]++;
			else if ( (itCount != mapCount.end()) && (itCount->second>=svNum_T) )
				svImg++;
		}

		//do not save image(all out of num_T)
		if (svImg == objNum )
			continue;

		//save enough
		objEnough = 0;
		if (mapCount.size() == 64 )
		{
			for(itCount = mapCount.begin(); itCount != mapCount.end(); itCount++)
			{
				if (itCount->second>=svNum_T)
					objEnough++;
				else
					break;
			}

			if ( objEnough == mapCount.size() )
				break;
		}

		/*****************************wget_image_file*****************************/
		sprintf( szImgPath, "%s/%lld_%lld.jpg", svPath, imageID, childID );
		std::string wgetFile = "wget -q -T 3 -t 1 " + url + " -O " + string(szImgPath);
		nRet = system( wgetFile.c_str() );

		/*****************************cvLoadImage*****************************/
		IplImage *img = cvLoadImage(szImgPath);
		if(!img || (img->width<16) || (img->height<16) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
		{	
			cout<<"Can't open " << szImgPath << endl;
			
			//rm img file
			sprintf(tPath, "rm %s", szImgPath );
			nRet = system(tPath);

			//Release
			cvReleaseImage(&img);img = 0;
			
			continue;
		}	
		//printf("szImgPath:%s\n",szImgPath);
		
		//save img data
		vecOutLabelRect.clear();
		for(i=0;i<objNum;i++)
		{
		    Vec4i rect;
			string label = vecQuaryString[9+i*6];
			float score = atof(vecQuaryString[10+i*6].c_str());
			rect[0] = atoi(vecQuaryString[11+i*6].c_str());
			rect[1] = atoi(vecQuaryString[12+i*6].c_str());
			rect[2] = atoi(vecQuaryString[13+i*6].c_str());
			rect[3] = atoi(vecQuaryString[14+i*6].c_str());

			if (score<0.9)
				continue;

			vecOutLabelRect.push_back( std::make_pair( label, rect ) );

			Scalar color = colors[i%8];
			cvRectangle( img, cvPoint(rect[0], rect[1]), cvPoint(rect[2], rect[3]), color, 2, 8, 0);

			sprintf(tPath, "%.2f %s", score, label.c_str() );
			text = tPath;
			cvPutText( img, text.c_str(), cvPoint(rect[0]+1, rect[1]+20), &font, color );
		}

		if (vecOutLabelRect.size()<1)
		{	
			//rm img file
			sprintf(tPath, "rm %s", szImgPath );
			nRet = system(tPath);

			//Release
			cvReleaseImage(&img);img = 0;
			
			continue;
		}	
		
		//save checkimg
		sprintf( savePath, "%s/CheckImg/%lld_%lld.jpg", svPath, imageID, childID );
		cvSaveImage( savePath, img );

		//write_xml
		sprintf( tPath, "%s/Annotations/", svPath );
		sprintf(charImageID, "%lld_%lld", imageID, childID );
		strImageID = charImageID;
		nRet = api_xml.write_xml( strImageID, tPath, vecOutLabelRect );
		if (nRet!=0)
		{
			//rm img file
			sprintf(tPath, "rm %s", szImgPath );
			nRet = system(tPath);
			
			cout << "Err to write_xml!" << endl;
			cvReleaseImage(&img);img = 0;
			continue;
		}

		//mv img file
		sprintf( tPath, "mv %s %s/JPEGImages/", szImgPath, svPath );
		system( tPath );

		//renew last imageID
		imageIDLast = imageID;

		nCountObj += objNum;
		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld img...\n",nCount);
		
		cvReleaseImage(&img);img = 0;
	}

	/*********************************close file*************************************/
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	

	/*********************************Print Info*********************************/
	if ( nCount != 0 ) 
	{
		printf( "nCount:%ld,nCountObj:%ld_%.4f,PredictTime:%.4fms\n", 
			nCount, nCountObj, nCountObj*1.0/nCount, allPredictTime*1000.0/nCount );
	}
	
	cout<<"Done!! "<<endl;
	
	return nRet;	
}

int frcnn_test( char *szQueryList, char* svPath, char* KeyFilePath, float MutiLabel_T, int binGPU, int deviceID )
{
	char tPath[256];
	char loadImgPath[256];
	char szImgPath[256];
	char savePath[256];
	vector<string> vecText;
	int i, j, label, svImg, nRet = 0;
	long inputLabel, nCount, nCountObj;
	string strImageID,text,name;
	double allPredictTime;
	FILE *fpListFile = 0;

	RunTimer<double> run;
	API_COMMEN api_commen;
	API_MUTI_LABEL api_muti_label;

	vector< MutiLabelInfo > Res;

	/***********************************Init**********************************/
	unsigned long long		imageID;				//[In]:image ID for CheckData
	unsigned long long		childID;				//[In]:image child ID for CheckData
	sprintf( tPath, "res/plog.log" );
	plog::init(plog::error, tPath); 
	sprintf( tPath, "res/module-in-logo-detection.log" );
	plog::init<enum_module_in_logo>(plog::info, tPath, 100000000, 100000);

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
	nRet = api_muti_label.Init( KeyFilePath, binGPU, deviceID ); 
	if (nRet != 0)
	{
	   cout<<"Fail to initialization "<<endl;
	   return TEC_INVALID_PARAM;
	}

	nCount = 0;
	nCountObj = 0;
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
		sprintf( tPath, "%d_%d", nCount, nCountObj );
		nRet = api_muti_label.Predict( img, string(tPath), nCount, nCountObj, MutiLabel_T, Res );
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

			sprintf(szImgPath, "%.2f %s", Res[i].score, Res[i].label.c_str() );
			text = szImgPath;
			cvPutText( img, text.c_str(), cvPoint(Res[i].rect[0]+1, Res[i].rect[1]+20), &font, color );

			//if (i<3)
			//	printf("Res:%.2f_%.2f_%.2f!!\n",Res[i].feat[0],Res[i].feat[1],Res[i].feat[2]);
		}
		sprintf( savePath, "%s/%s.jpg", svPath, strImageID.c_str() );
		cvSaveImage( savePath, img );

		nCountObj += Res.size();
		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld img...,time:%.4f\n",nCount,run.time());
		
		cvReleaseImage(&img);img = 0;
	}

	/*********************************close file*************************************/
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	

	/*********************************Release*************************************/
	api_muti_label.Release();

	/*********************************Print Info*********************************/
	if ( nCount != 0 ) 
	{
		printf( "nCount:%ld,nCountObj:%ld_%.4f,PredictTime:%.4fms\n", 
			nCount, nCountObj, nCountObj*1.0/nCount, allPredictTime*1000.0/nCount );
	}
	
	cout<<"Done!! "<<endl;
	
	return nRet;	
}

int frcnn_test_face( char *szQueryList, char* KeyFilePath, float MutiLabel_T, int binGPU, int deviceID )
{
	char tPath[256];
	char loadImgPath[256];
	char szImgPath[256];
	char savePath[256];
	vector<string> vecText;
	int i, j, label, svImg, nRet = 0;
	long inputLabel, nCount, nCountObj, nCountFace;
	string strImageID,text,name;
	double allPredictTime;
	FILE *fpListFile = 0;

	RunTimer<double> run;
	API_COMMEN api_commen;
	API_MUTI_LABEL api_muti_label;

	vector< MutiLabelInfo > Res_MultiLabel;
	vector< FaceAnnoationInfo > Res_FaceAnnoation;

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
	nRet = api_muti_label.Init( KeyFilePath, binGPU, deviceID ); 
	if (nRet != 0)
	{
	   cout<<"Fail to initialization "<<endl;
	   return TEC_INVALID_PARAM;
	}

	nCount = 0;
	nCountObj = 0;
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
		Res_MultiLabel.clear();
		run.start();
		sprintf( tPath, "%d_%d", nCount, nCountObj );
		nRet = api_muti_label.Predict( img, string(tPath), nCount, nCountObj, MutiLabel_T, Res_MultiLabel );
		if ( (nRet!=0) || (Res_MultiLabel.size()<1) )
		{
			LOOGE<<"[Predict Err!!loadImgPath:]"<<loadImgPath;
			cvReleaseImage(&img);img = 0;
			continue;
		}
		run.end();
		//LOOGI<<"[Predict] time:"<<run.time();
		allPredictTime += run.time();

		/***********************************Face:Predict**********************************/
		Res_FaceAnnoation.clear();
		run.start();
		nRet = api_muti_label.Face_Predict( img, Res_MultiLabel, Res_FaceAnnoation );
		if ( (nRet!=0) || (Res_FaceAnnoation.size()<1) )
		{
			LOOGE<<"[Face:Predict Err!!loadImgPath:]"<<loadImgPath;
			cvReleaseImage(&img);img = 0;
			continue;
		}
		run.end();
		//LOOGI<<"[Face:Predict] time:"<<run.time();
		allPredictTime += run.time();
		
		/************************save img data*****************************/
		{
			name = Res_FaceAnnoation[0].label;
			if (name=="person.face")
				sprintf(tPath, "res_predict/face/", tPath );
			else
				sprintf(tPath, "res_predict/noface/", tPath );
		}
		for(i=0;i<Res_FaceAnnoation.size();i++)  
		{						
			Scalar color = colors[i%8];

			if (Res_FaceAnnoation[i].label == "person.face" )
			{
				api_muti_label.Face_Draw_Rotate_Box( img, Res_FaceAnnoation[i], color );

				IplImage* Rotate_ROI = api_muti_label.Face_Get_Rotate_ROI( img, Res_FaceAnnoation[i] );
				sprintf( savePath, "res_predict/roi/%s_%d.jpg", strImageID.c_str(), i );
				cvSaveImage( savePath, Rotate_ROI );
				cvReleaseImage(&Rotate_ROI);Rotate_ROI = 0;
				nCountFace++;
			}
			
			cvRectangle( img, cvPoint(Res_FaceAnnoation[i].rect[0], Res_FaceAnnoation[i].rect[1]),
	                   cvPoint(Res_FaceAnnoation[i].rect[2], Res_FaceAnnoation[i].rect[3]), color, 2, 8, 0);

			sprintf(szImgPath, "%.2f %.2f %s", Res_FaceAnnoation[i].score, Res_FaceAnnoation[i].angle, Res_FaceAnnoation[i].label.c_str() );
			text = szImgPath;
			cvPutText( img, text.c_str(), cvPoint(Res_FaceAnnoation[i].rect[0]+1, Res_FaceAnnoation[i].rect[1]+20), &font, color );

			for (j=0;j<5;j++)
				cvCircle( img, cvPoint(Res_FaceAnnoation[i].annoation[2*j], Res_FaceAnnoation[i].annoation[2*j+1]), 2, colors[j%8], 2, 8, 0 );

			sprintf(tPath, "%s%s-%.2f_", tPath, Res_FaceAnnoation[i].label.c_str(), Res_FaceAnnoation[i].score );
		}
		sprintf( savePath, "%s%s.jpg", tPath, strImageID.c_str() );
		cvSaveImage( savePath, img );

		nCountObj += Res_MultiLabel.size();
		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld img...,time:%.4f\n",nCount,run.time());
		
		cvReleaseImage(&img);img = 0;
	}

	/*********************************close file*************************************/
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	

	/*********************************Release*************************************/
	api_muti_label.Release();

	/*********************************Print Info*********************************/
	if ( nCount != 0 ) 
	{
		printf( "nCount:%ld,nCountObj:%ld_%.4f,nCountFace:%ld_%.4f,PredictTime:%.4fms\n", 
			nCount, nCountObj, nCountObj*1.0/nCount, nCountFace, nCountFace*1.0/nCount, allPredictTime*1000.0/nCount );
	}
	
	cout<<"Done!! "<<endl;
	
	return nRet;	
}

int frcnn_test_checkface( char *szQueryList, char* svPath, char* KeyFilePath, float MutiLabel_T, int binGPU, int deviceID )
{
	char tPath[256];
	char loadImgPath[256];
	char szImgPath[256];
	char savePath[256];
	vector<string> vecText;
	int i, j, label, svImg, nRet = 0;
	long inputLabel, nCount, nCountObj;
	string strImageID,text,name;
	double allPredictTime;
	FILE *fpListFile = 0;

	RunTimer<double> run;
	API_COMMEN api_commen;
	API_MUTI_LABEL api_muti_label;

	vector< MutiLabelInfo > Res;

	/***********************************Init**********************************/
	unsigned long long		imageID;				//[In]:image ID for CheckData
	unsigned long long		childID;				//[In]:image child ID for CheckData
	sprintf( tPath, "res/plog.log" );
	plog::init(plog::error, tPath); 
	sprintf( tPath, "res/module-in-logo-detection.log" );
	plog::init<enum_module_in_logo>(plog::info, tPath, 100000000, 100000);

	const static Scalar colors[] =	{	CV_RGB(0,0,255),	CV_RGB(0,255,255),	CV_RGB(0,255,0),	
										CV_RGB(255,255,0),	CV_RGB(255,0,0),	CV_RGB(255,0,255),
										CV_RGB(0,128,255),	CV_RGB(255,128,0)} ;

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
	nRet = api_muti_label.Init( KeyFilePath, binGPU, deviceID ); 
	if (nRet != 0)
	{
	   cout<<"Fail to initialization "<<endl;
	   return TEC_INVALID_PARAM;
	}

	nCount = 0;
	nCountObj = 0;
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
		sprintf( tPath, "%d_%d", nCount, nCountObj );
		nRet = api_muti_label.Predict( img, string(tPath), nCount, nCountObj, MutiLabel_T, Res );
		
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
		int bFace = 0;
		for(i=0;i<Res.size();i++) 
		{
			if (Res[i].label == "person.face" )
			{
				bFace = 1;
				break;
			}
		}
		
		{
			if (bFace==1)
				sprintf(tPath, "res_predict/face/", tPath );
			else
				sprintf(tPath, "res_predict/noface/", tPath );
		}
		
		for(i=0;i<Res.size();i++)  
		{		
			if (Res[i].label == "person.face" )
			{
				cvSetImageROI( img,cvRect(Res[i].rect[0], Res[i].rect[1], 
					(Res[i].rect[2]-Res[i].rect[0]),(Res[i].rect[3]-Res[i].rect[1])) );
				IplImage* faceROI = cvCreateImage(cvGetSize(img),img->depth,img->nChannels);
                cvCopy( img, faceROI, NULL );
                cvResetImageROI(img);
				sprintf( savePath, "res_predict/roi/%s_%d.jpg", strImageID.c_str(), i );
				cvSaveImage( savePath, faceROI );
				cvReleaseImage(&faceROI);faceROI = 0;
			}
		
			Scalar color = colors[i%8];
			cvRectangle( img, cvPoint(Res[i].rect[0], Res[i].rect[1]),
					   cvPoint(Res[i].rect[2], Res[i].rect[3]), color, 2, 8, 0);

			sprintf(szImgPath, "%.2f %s", Res[i].score, Res[i].label.c_str() );
			text = szImgPath;
			cvPutText( img, text.c_str(), cvPoint(Res[i].rect[0]+1, Res[i].rect[1]+20), &font, color );

			//if (i<3)
			//	printf("Res:%.2f_%.2f_%.2f!!\n",Res[i].feat[0],Res[i].feat[1],Res[i].feat[2]);
		}
		sprintf( savePath, "%s/%s.jpg", tPath, strImageID.c_str() );
		cvSaveImage( savePath, img );

		nCountObj += Res.size();
		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld img...,time:%.4f\n",nCount,run.time());
		
		cvReleaseImage(&img);img = 0;
	}

	/*********************************close file*************************************/
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	

	/*********************************Release*************************************/
	api_muti_label.Release();

	/*********************************Print Info*********************************/
	if ( nCount != 0 ) 
	{
		printf( "nCount:%ld,nCountObj:%ld_%.4f,PredictTime:%.4fms\n", 
			nCount, nCountObj, nCountObj*1.0/nCount, allPredictTime*1000.0/nCount );
	}
	
	cout<<"Done!! "<<endl;
	
	return nRet;	
}

int frcnn_test_class( char *szQueryList, char* svPath, char* KeyFilePath, float MutiLabel_T, int binGPU, int deviceID )
{
	char tPath[256];
	char loadImgPath[256];
	char szImgPath[256];
	char savePath[256];
	vector<string> vecText;
	int i, j, label, svImg, nRet = 0;
	long inputLabel, nCount, nCountObj;
	string strImageID,text,name;
	double allPredictTime;
	FILE *fpListFile = 0;

	RunTimer<double> run;
	API_COMMEN api_commen;
	API_MUTI_LABEL api_muti_label;

	vector< MutiLabelInfo > Res;

	/***********************************Init**********************************/
	unsigned long long		imageID;				//[In]:image ID for CheckData
	unsigned long long		childID;				//[In]:image child ID for CheckData
	sprintf( tPath, "res/plog.log" );
	plog::init(plog::error, tPath); 
	sprintf( tPath, "res/module-in-logo-detection.log" );
	plog::init<enum_module_in_logo>(plog::info, tPath, 100000000, 100000);

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
	nRet = api_muti_label.Init( KeyFilePath, binGPU, deviceID ); 
	if (nRet != 0)
	{
	   cout<<"Fail to initialization "<<endl;
	   return TEC_INVALID_PARAM;
	}

	nCount = 0;
	nCountObj = 0;
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
		sprintf( tPath, "%d_%d", nCount, nCountObj );
		nRet = api_muti_label.Predict( img, string(tPath), nCount, nCountObj, MutiLabel_T, Res );
		if ( (nRet!=0) || (Res.size()<1) )
		{
			LOOGE<<"[Predict Err!!loadImgPath:]"<<loadImgPath;
			cvReleaseImage(&img);img = 0;
			continue;
		}
		run.end();
		//LOOGI<<"[Predict] time:"<<run.time();
		allPredictTime += run.time();

		int sv = 0;
		/************************save img data*****************************/
		for(i=0;i<Res.size();i++)  
		{						
			Scalar color = colors[i%8];
			cvRectangle( img, cvPoint(Res[i].rect[0], Res[i].rect[1]),
	                   cvPoint(Res[i].rect[2], Res[i].rect[3]), color, 2, 8, 0);

			sprintf(szImgPath, "%.2f %s", Res[i].score, Res[i].label.c_str() );
			text = szImgPath;
			cvPutText( img, text.c_str(), cvPoint(Res[i].rect[0]+1, Res[i].rect[1]+20), &font, color );

			//if (i<3)
			//	printf("Res:%.2f_%.2f_%.2f!!\n",Res[i].feat[0],Res[i].feat[1],Res[i].feat[2]);

			//if ((Res[i].label == "animal.cat") || (Res[i].label == "animal.dog") || (Res[i].label == "food.fruit") )
			if (Res[i].label == "food.fruit")
				sv = 1;
		}
		if ( sv == 1 )
		{
			sprintf( savePath, "%s/CheckImg/%s.jpg", svPath, strImageID.c_str() );
			cvSaveImage( savePath, img );

			//cp img file
			sprintf( tPath, "cp %s %s/JPEGImages/", loadImgPath, svPath );
			system( tPath );
		}

		nCountObj += Res.size();
		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld img...,time:%.4f\n",nCount,run.time());
		
		cvReleaseImage(&img);img = 0;
	}

	/*********************************close file*************************************/
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	

	/*********************************Release*************************************/
	api_muti_label.Release();

	/*********************************Print Info*********************************/
	if ( nCount != 0 ) 
	{
		printf( "nCount:%ld,nCountObj:%ld_%.4f,PredictTime:%.4fms\n", 
			nCount, nCountObj, nCountObj*1.0/nCount, allPredictTime*1000.0/nCount );
	}
	
	cout<<"Done!! "<<endl;
	
	return nRet;	
}

int main(int argc, char* argv[])
{
	int  ret = 0;
	char szKeyFiles[256],szSavePath[256];
	API_COMMEN api_commen;
	
	//inLabelClass:0-voc,1-coco,2-old in;
	if (argc == 9 && strcmp(argv[1],"-dataCH2InMutiLabel") == 0) {
		ret = Get_DataCH2InMutiLabel( argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], atoi(argv[8]) );
	}
	else if (argc == 5 && strcmp(argv[1],"-change_label") == 0) {
		ret = Change_label( argv[2], argv[3], argv[4] );
	}
	else if (argc == 4 && strcmp(argv[1],"-change_voclabel4test") == 0) {
		ret = Change_voclabel4test( argv[2], argv[3] );
	}
	else if (argc == 6 && strcmp(argv[1],"-count_data") == 0) {
		ret = Count_Data( argv[2], argv[3], argv[4], atoi(argv[5]) );
	}
	else if (argc == 7 && strcmp(argv[1],"-count_data_shengdan") == 0) {
		ret = Count_Data_shengdan( argv[2], argv[3], argv[4], atoi(argv[5]), atoi(argv[6]) );
	}
	else if (argc == 8 && strcmp(argv[1],"-resample") == 0) {
		ret = frcnn_ReSample( argv[2], argv[3], argv[4], atof(argv[5]), atoi(argv[6]), atoi(argv[7]) );
	}
	else if (argc == 5 && strcmp(argv[1],"-check_onlinetest") == 0) {
		ret = frcnn_Check_OnlineTest( argv[2], argv[3], atoi(argv[4]) );
	}
	else if (argc == 8 && strcmp(argv[1],"-test") == 0) {
		ret = frcnn_test( argv[2], argv[3], argv[4], atof(argv[5]), atoi(argv[6]), atoi(argv[7]) );
	}
	else if (argc == 7 && strcmp(argv[1],"-test_face") == 0) {
		ret = frcnn_test_face( argv[2], argv[3], atof(argv[4]), atoi(argv[5]), atoi(argv[6]) );
	}
	else if (argc == 8 && strcmp(argv[1],"-test_checkface") == 0) {
		ret = frcnn_test_checkface( argv[2], argv[3], argv[4], atof(argv[5]), atoi(argv[6]), atoi(argv[7]) );
	}
	else if (argc == 8 && strcmp(argv[1],"-test_class") == 0) {
		ret = frcnn_test_class( argv[2], argv[3], argv[4], atof(argv[5]), atoi(argv[6]), atoi(argv[7]) );
	}
	else
	{
		cout << "usage:\n" << endl;
		cout << "\tDemo_mutilabel -dataCH2InMutiLabel loadXMLPath loadImagePath inDict outDict svXml svImgPath inLabelClass\n" << endl;
		cout << "\tDemo_mutilabel -change_label loadXMLPath loadImagePath svPath\n" << endl;
		cout << "\tDemo_mutilabel -change_voclabel4test loadXMLPath svPath\n" << endl;
		cout << "\tDemo_mutilabel -count_data loadXMLPath loadImagePath svPath svImgXml\n" << endl;
		cout << "\tDemo_mutilabel -count_data_shengdan loadXMLPath loadImagePath svPath svImgXml svROI\n" << endl;
		cout << "\tDemo_mutilabel -resample loadImagePath svPath keyfile MutiLabel_T binGPU deviceID\n" << endl;
		cout << "\tDemo_mutilabel -check_onlinetest loadPath svPath svNum_T\n" << endl;
		cout << "\tDemo_mutilabel -test loadImagePath svPath keyfile MutiLabel_T binGPU deviceID\n" << endl;
		cout << "\tDemo_mutilabel -test_face loadImagePath keyfile MutiLabel_T binGPU deviceID\n" << endl;
		cout << "\tDemo_mutilabel -test_class loadImagePath svPath keyfile MutiLabel_T binGPU deviceID\n" << endl;
		cout << "\tDemo_mutilabel -test_checkface loadImagePath svPath keyfile MutiLabel_T binGPU deviceID\n" << endl;
		return ret;
	}
	return ret;
}

