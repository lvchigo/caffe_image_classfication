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

#include "API_commen.h"
#include "API_multilabel_imgtext.h"
#include "TErrorCode.h"
#include "plog/Log.h"
#include "API_xml.h"	//read xml

using namespace cv;
using namespace std;

int DataChange_ShuffleData( char *input, char *output )
{
	char tPath[4096];
	char loadPath[256];
	int i, nRet = 0;
	long nCount;
	
	/***********************************Load dic File**********************************/
	vector<string> lines;
	
	/********************************Open Query List*****************************/
	FILE *fpInput = fopen(input,"r");
	if (!fpInput) 
	{
		cout << "0.can't open " << input << endl;
		return -1;
	}

	FILE *fpOutput = fopen(output, "wt");
	if (!fpOutput)
	{
		return -1;
	}

	//Process one by one
	lines.clear();
	while(EOF != fscanf(fpInput, "%s", &loadPath))
	{
		lines.push_back(string(loadPath));
	}

	//shuffle
	random_shuffle(lines.begin(), lines.end());

	//write
	nCount = 0;
	for (i=0;i<lines.size();i++)
	{
		fprintf(fpOutput, "%s\n", lines[i].c_str() );

		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld lines...\n",nCount);
	}

	/*********************************close file*************************************/
	if (fpInput) {fclose(fpInput);fpInput = 0;}	
	if (fpOutput) {fclose(fpOutput);fpOutput = 0;}

	/*********************************Print Info*********************************/
	if ( nCount != 0 ) 
	{
		printf( "nCount:%ld\n", nCount );
	}
	
	cout<<"Done!! "<<endl;
	
	return nRet;	
}


int DataChange_inmutilabl64class( char *szXmlList, char *loadImagePath, char *keyword, char* svPath )
{
	char tPath[4096];
	char charImageID[256];
	char szImgPath[256];
	char savePath[256];
	char strPath[4096];
	char imglabelPath[40960];
	char loadImgPath[256];
	char loadXmlPath[256];
	vector<string> vecText;
	int i, j, label, svImg, nRet = 0;
	long inputLabel, nCount, nCountObj, objEnough;
	unsigned long long randomID = 0;
	string strID,text,name;

	API_COMMEN 	api_commen;
	API_XML api_xml;

	vector < pair < string,Vec4i > > 	vecInLabelRect;
	
	/***********************************Load dic File**********************************/
	vector<string> dic;
	map< string,int > mapDic;
	map< string,int >::iterator itDic;

	map< string,int > mapImgText;
	map< string,int >::iterator itImgText;
	
	dic.clear();
	printf("load dic:%s\n",keyword);
	api_commen.loadWordDict(keyword,dic);
	printf( "dict:size-%d,tag:", int(dic.size()) );
	for ( i=0;i<dic.size();i++ )
	{
		printf( "%d-%s ",i+1, dic[i].c_str() );
		mapDic[dic[i]] = i+1;
	}
	printf( "\n" );
	
	/********************************Open Query List*****************************/
	FILE *fpListFile = fopen(szXmlList,"r");
	if (!fpListFile) 
	{
		cout << "0.can't open " << szXmlList << endl;
		return -1;
	}

	sprintf( tPath, "%s/labels_train.txt", svPath );
	FILE *fpLabelsTrain = fopen(tPath, "wt");
	if (!fpLabelsTrain)
	{
		return -1;
	}

	sprintf( tPath, "%s/labels_val.txt", svPath );
	FILE *fpLabelsVal = fopen(tPath, "wt");
	if (!fpLabelsVal)
	{
		return -1;
	}

	//Process one by one
	nCount = 0;
	while(EOF != fscanf(fpListFile, "%s", &loadXmlPath))
	{
		//read_xml
		vecInLabelRect.clear();
		nRet = api_xml.load_xml( string(loadXmlPath), strID, vecInLabelRect );
		if ( (nRet!=0) || (vecInLabelRect.size()<1) )
		{
			cout << "Err to load_xml!" << endl;
			continue;
		}

		/*****************************cvLoadImage*****************************/
		sprintf( loadImgPath, "%s/%s.jpg", loadImagePath, strID.c_str() );
		IplImage *img = cvLoadImage(loadImgPath);
		if(!img || (img->width<32) || (img->height<32) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
		{	
			cout<<"Can't open " << loadImgPath << endl;
			cvReleaseImage(&img);img = 0;
			continue;
		}

		//save img data
		IplImage *ImgResize = cvCreateImage(cvSize(255, 255), img->depth, img->nChannels);
		cvResize( img, ImgResize );
		
		//save checkimg
		sprintf( savePath, "%s/JPEGImages/%s.jpg", svPath, strID.c_str() );
		cvSaveImage( savePath, ImgResize );

		//get keyword-label and stay alone
		mapImgText.clear();//keep word alone
		sprintf( imglabelPath, "%s.jpg", strID.c_str() );
		for(i=0;i<vecInLabelRect.size();i++)
		{	
			//search keyword
			itImgText = mapImgText.find(vecInLabelRect[i].first);
			if (itImgText == mapImgText.end()) // not find
			{
				mapImgText[vecInLabelRect[i].first] = 1;

				itDic = mapDic.find(vecInLabelRect[i].first);
				if (itDic != mapDic.end()) // find
					sprintf( imglabelPath, "%s,%d", imglabelPath, itDic->second );
			}
		}

		//write img-label data
		if( nCount%10 != 0 ) //train
			fprintf(fpLabelsTrain, "%s\n", imglabelPath );
		else	//val
			fprintf(fpLabelsVal, "%s\n", imglabelPath );
		
		cvReleaseImage(&img);img = 0;
		cvReleaseImage(&ImgResize);ImgResize = 0;

		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld img...\n",nCount);
	}

	/*********************************close file*************************************/
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	
	if (fpLabelsTrain) {fclose(fpLabelsTrain);fpLabelsTrain = 0;}
	if (fpLabelsVal) {fclose(fpLabelsVal);fpLabelsVal = 0;}

	/*********************************Print Info*********************************/
	if ( nCount != 0 ) 
	{
		printf( "nCount:%ld\n", nCount );
	}
	
	cout<<"Done!! "<<endl;
	
	return nRet;	
}

int DataChange_itugodoudou383class( char *szQueryList, char *loadImagePath, char *keyword, char* svPath )
{
	char tPath[4096];
	char charImageID[256];
	char szImgPath[256];
	char savePath[256];
	char strPath[4096];
	char imglabelPath[40960];
	char loadImgPath[256];
	vector<string> vecText;
	int i, j, label, svImg, nRet = 0;
	long inputLabel, nCount, nCountObj, objEnough;
	unsigned long long randomID = 0;
	string text,name;

	API_COMMEN 	api_commen;
	
	/***********************************Load dic File**********************************/
	vector<string> dic;
	map< string,int > mapDic;
	map< string,int >::iterator itDic;

	map< string,int > mapImgText;
	map< string,int >::iterator itImgText;
	
	dic.clear();
	printf("load dic:%s\n",keyword);
	api_commen.loadWordDict(keyword,dic);
	printf( "dict:size-%d,tag:", int(dic.size()) );
	for ( i=0;i<dic.size();i++ )
	{
		printf( "%d-%s ",i+1+64, dic[i].c_str() );
		mapDic[dic[i]] = i+1+64;	//add multilabel-64class
	}
	printf( "\n" );
	
	/********************************Open Query List*****************************/
	FILE *fpListFile = fopen(szQueryList,"r");
	if (!fpListFile) 
	{
		cout << "0.can't open " << szQueryList << endl;
		return -1;
	}

	sprintf( tPath, "%s/labels_train.txt", svPath );
	FILE *fpLabelsTrain = fopen(tPath, "wt");
	if (!fpLabelsTrain)
	{
		return -1;
	}

	sprintf( tPath, "%s/labels_val.txt", svPath );
	FILE *fpLabelsVal = fopen(tPath, "wt");
	if (!fpLabelsVal)
	{
		return -1;
	}

	vector < string > des_split;
	vector < string > word;
	vector < string > word_cutword;

	//Process one by one
	nCount = 0;
	while(EOF != fscanf(fpListFile, "%s", &strPath))
	{
		des_split.clear();
		api_commen.split( string(strPath), ";", des_split);
		if (des_split.size()!=3)
			continue;

		//Cut Word
		word_cutword.clear();
		api_commen.split( string(des_split[1]), ",", word_cutword);
		if (word_cutword.size()<1)
			continue;

		//Check Word
		word.clear();
		for(i=0;i<word_cutword.size();i++)
		{
			itDic = mapDic.find(word_cutword[i]);
			if (itDic == mapDic.end()) // no find
				continue;

			word.push_back(word_cutword[i]);
		}
		if (word.size()<1)
			continue;

		/*****************************cvLoadImage*****************************/
		sprintf( loadImgPath, "%s/%s.jpg", loadImagePath, des_split[0].c_str() );
		IplImage *img = cvLoadImage(loadImgPath);
		if(!img || (img->width<32) || (img->height<32) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
		{	
			cout<<"Can't open " << loadImgPath << endl;
			cvReleaseImage(&img);img = 0;
			
			continue;
		}

		//get keyword-label and stay alone
		nCountObj = 0;
		mapImgText.clear();//keep word alone
		sprintf( imglabelPath, "%s.jpg", des_split[0].c_str() );
		for(i=0;i<word.size();i++)
		{	
			//search keyword
			itImgText = mapImgText.find(word[i]);
			if (itImgText == mapImgText.end()) // not find
			{
				mapImgText[word[i]] = 1;

				itDic = mapDic.find(word[i]);
				if (itDic != mapDic.end()) // find
				{
					nCountObj++;
					sprintf( imglabelPath, "%s,%d", imglabelPath, itDic->second );
				}
			}
		}

		if (nCountObj<1)
			continue;

		//write img-label data
		if( nCount%10 != 0 ) //train
			fprintf(fpLabelsTrain, "%s\n", imglabelPath );
		else	//val
			fprintf(fpLabelsVal, "%s\n", imglabelPath );
		
		cvReleaseImage(&img);img = 0;

		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld img...\n",nCount);
	}

	/*********************************close file*************************************/
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	
	if (fpLabelsTrain) {fclose(fpLabelsTrain);fpLabelsTrain = 0;}
	if (fpLabelsVal) {fclose(fpLabelsVal);fpLabelsVal = 0;}

	/*********************************Print Info*********************************/
	if ( nCount != 0 ) 
	{
		printf( "nCount:%ld\n", nCount );
	}
	
	cout<<"Done!! "<<endl;
	
	return nRet;	
}

int DataChange_sogou98class( char *szQueryList, char *loadImagePath, char *keyword, char* svPath )
{
	char tPath[4096];
	char charImageID[256];
	char szImgPath[256];
	char savePath[256];
	char strPath[4096];
	char imglabelPath[40960];
	char loadImgPath[256];
	vector<string> vecText;
	int i, j, label, svImg, nRet = 0;
	long inputLabel, nCount, nCountObj, objEnough;
	unsigned long long randomID = 0;
	string text,name;

	API_COMMEN 	api_commen;

	vector < string > des_split;
	vector < string > word;
	vector < string > word_cutword;
	
	/***********************************Load dic File**********************************/
	vector<string> dic;
	map< string,int > mapDic;	//<key,label_id>
	map< string,int >::iterator itDic;

	map< string,string > mapDicString;	//<tag,key>
	map< string,string >::iterator itDicString;

	map< string,int > mapImgText;
	map< string,int >::iterator itImgText;

	sprintf( tPath, "%s/keyword.txt", svPath );
	FILE *fpKeyword = fopen(tPath, "wt");
	if (!fpKeyword)
	{
		return -1;
	}
	
	dic.clear();
	mapDic.clear();
	mapDicString.clear();
	printf("load dic:%s\n",keyword);
	printf("add tag:");
	api_commen.loadWordDict(keyword,dic);
	for ( i=0;i<dic.size();i++ )
	{
		des_split.clear();
		api_commen.split( dic[i], ";", des_split);
		if (des_split.size()!=3)
			continue;
		
		//Cut Word
		word_cutword.clear();
		api_commen.split( string(des_split[2]), ",", word_cutword);
		if (word_cutword.size()<1)
			continue;

		label = atoi(des_split[0].c_str());

		//set data
		itDic = mapDic.find(des_split[1]);
		if (itDic != mapDic.end()) // find
		{
			printf( "find same keyword:%s\n", des_split[1].c_str() );
			return -1;
		}
		else
			mapDic[des_split[1]] = label;	//add multilabel-64class
				
		for (j=0;j<word_cutword.size();j++)
		{
			itDicString = mapDicString.find(word_cutword[j]);
			if (itDicString != mapDicString.end()) // find
			{
				printf( "find same keyword:%s\n", word_cutword[j].c_str() );
				return -1;
			}
			else
				mapDicString[word_cutword[j]] = des_split[1];	//add multilabel-64class
		}

		//print
		if(label>64)	//print add labels
		{
			fprintf(fpKeyword, "%s\n", des_split[1].c_str() );
			printf( "%d-%s ",label, des_split[1].c_str() );
		}
	}
	printf( "\n" );
	
	/********************************Open Query List*****************************/
	FILE *fpListFile = fopen(szQueryList,"r");
	if (!fpListFile) 
	{
		cout << "0.can't open " << szQueryList << endl;
		return -1;
	}

	sprintf( tPath, "%s/labels_train.txt", svPath );
	FILE *fpLabelsTrain = fopen(tPath, "wt");
	if (!fpLabelsTrain)
	{
		return -1;
	}

	sprintf( tPath, "%s/labels_val.txt", svPath );
	FILE *fpLabelsVal = fopen(tPath, "wt");
	if (!fpLabelsVal)
	{
		return -1;
	}

	//Process one by one
	nCount = 0;
	while(EOF != fscanf(fpListFile, "%s", &strPath))
	{
		des_split.clear();
		api_commen.split( string(strPath), ",", des_split);
		if (des_split.size()<3)
			continue;

		//Check Word
		word.clear();
		for(i=2;i<des_split.size();i++)
		{
			itDicString = mapDicString.find(des_split[i]);
			if (itDicString == mapDicString.end()) // no find
				continue;
			else
				word.push_back(itDicString->second);	//tag-->keyword
		}
		if (word.size()<1)
			continue;

		/*****************************cvLoadImage*****************************/
		sprintf( loadImgPath, "%s/%s.jpg", loadImagePath, des_split[0].c_str() );
		IplImage *img = cvLoadImage(loadImgPath);
		if(!img || (img->width<32) || (img->height<32) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
		{	
			cout<<"Can't open " << loadImgPath << endl;
			cvReleaseImage(&img);img = 0;
			
			continue;
		}
		cvReleaseImage(&img);img = 0;

		//get keyword-label and stay alone
		nCountObj = 0;
		mapImgText.clear();//keep word alone
		sprintf( imglabelPath, "%s.jpg", des_split[0].c_str() );
		for(i=0;i<word.size();i++)
		{	
			//search keyword
			itImgText = mapImgText.find(word[i]);
			if (itImgText == mapImgText.end()) // not find
			{
				mapImgText[word[i]] = 1;

				itDic = mapDic.find(word[i]);
				if (itDic != mapDic.end()) // find
				{
					nCountObj++;
					sprintf( imglabelPath, "%s,%d", imglabelPath, itDic->second );
				}
			}
		}

		if (nCountObj<1)
			continue;

		//write img-label data
		if( nCount%10 != 0 ) //train
			fprintf(fpLabelsTrain, "%s\n", imglabelPath );
		else	//val
			fprintf(fpLabelsVal, "%s\n", imglabelPath );

		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld img...\n",nCount);
	}

	/*********************************close file*************************************/
	if (fpKeyword) {fclose(fpKeyword);fpKeyword = 0;}
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	
	if (fpLabelsTrain) {fclose(fpLabelsTrain);fpLabelsTrain = 0;}
	if (fpLabelsVal) {fclose(fpLabelsVal);fpLabelsVal = 0;}

	/*********************************Print Info*********************************/
	if ( nCount != 0 ) 
	{
		printf( "nCount:%ld\n", nCount );
	}
	
	cout<<"Done!! "<<endl;
	
	return nRet;	
}

int frcnn_test( char *szQueryList, char* svPath, char* KeyFilePath, char* layerName, int binGPU, int deviceID )
{
	char tPath[40960];
	char loadImgPath[4096];
	char szImgPath[4096];
	char savePath[4096];
	vector<string> vecText;
	int i, j, label, svImg, label_add, nRet = 0;
	long inputLabel, nCount, nCountObj;
	string strImageID,text,name;
	double allPredictTime;
	FILE *fpListFile = 0;

	RunTimer<double> run;
	API_COMMEN api_commen;
	API_MUTILABEL_IMGTEXT api_multilabel_imgtext;

	vector< pair< string, float > >	ResProb;			//[Out]:Res:prob layer
	vector< pair< string, float > >	ResScore;			//[Out]:Res:score layer

	/***********************************Init**********************************/
	unsigned long long		imageID;				//[In]:image ID for CheckData
	unsigned long long		childID;				//[In]:image child ID for CheckData
	sprintf( tPath, "res/plog.log" );
	plog::init(plog::error, tPath); 
	//sprintf( tPath, "res/module-in-logo-detection.log" );
	//plog::init<enum_module_in_logo>(plog::info, tPath, 100000000, 100000);

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
	nRet = api_multilabel_imgtext.Init( KeyFilePath, layerName, binGPU, deviceID ); 
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

		/************************getRandomID*****************************/
		strImageID = api_commen.GetStringIDFromFilePath( loadImgPath );
		
		/************************Predict*****************************/	
		ResProb.clear();
		ResScore.clear();
		label_add = 0;
		run.start();
		sprintf( tPath, "%d_%d", nCount, nCountObj );
		nRet = api_multilabel_imgtext.Predict(img, layerName, ResProb, ResScore, label_add);
		if ( (nRet!=0) || (ResProb.size()<1) || (ResScore.size()<1) )
		{
			LOOGE<<"[Predict Err!!loadImgPath:]"<<loadImgPath;
			cvReleaseImage(&img);img = 0;
			continue;
		}
		run.end();
		allPredictTime += run.time();
		
		/************************save img data*****************************/
		for(i=0;i<ResProb.size();i++)  
		{		
			Scalar color = colors[i%8];
			
			if (i==0)
				cvPutText( img, "prob-layer:", cvPoint(1, 20), &font, color );
			sprintf(szImgPath, "%.2f %s", ResProb[i].second, ResProb[i].first.c_str() );
			text = szImgPath;
			cvPutText( img, text.c_str(), cvPoint(1, (i+2)*20), &font, color );
		}

		sprintf( savePath, "%s/img/%s", svPath, strImageID.c_str() );
		for(i=0;i<ResScore.size();i++)  
		{		
			Scalar color = colors[i%8];
			
			if (i==0)
				cvPutText( img, "score-layer:", cvPoint(1, (ResProb.size()+2)*20), &font, color );
			sprintf(szImgPath, "%.2f_%s", ResScore[i].second, ResScore[i].first.c_str() );
			sprintf( savePath, "%s_%s", savePath, szImgPath );
			text = szImgPath;
			cvPutText( img, text.c_str(), cvPoint(1, (ResProb.size()+i+3)*20), &font, color );
		}
		sprintf( savePath, "%s.jpg", savePath );
		//sprintf( savePath, "%s/%s.jpg", svPath, strImageID.c_str() );
		cvSaveImage( savePath, img );

		if (label_add == 1)
		{
			//cp img file
			//printf("savePath:%s\n",savePath);
			sprintf( tPath, "cp %s %s/CheckImage/", savePath, svPath );
			//printf("tPath:%s\n",tPath);
			system( tPath );
		}

		nCountObj += ResScore.size();
		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld img...,time:%.4f\n",nCount,run.time());
		
		cvReleaseImage(&img);img = 0;
	}

	/*********************************close file*************************************/
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	

	/*********************************Release*************************************/
	api_multilabel_imgtext.Release();

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
	
	if (argc == 4 && strcmp(argv[1],"-DataChange_ShuffleData") == 0) {
		ret = DataChange_ShuffleData( argv[2], argv[3] );
	}
	else if (argc == 6 && strcmp(argv[1],"-DataChange_inmutilabl64class") == 0) {
		ret = DataChange_inmutilabl64class( argv[2], argv[3], argv[4], argv[5] );
	}
	else if (argc == 6 && strcmp(argv[1],"-DataChange_itugodoudou383class") == 0) {
		ret = DataChange_itugodoudou383class( argv[2], argv[3], argv[4], argv[5] );
	}
	else if (argc == 6 && strcmp(argv[1],"-DataChange_sogou98class") == 0) {
		ret = DataChange_sogou98class( argv[2], argv[3], argv[4], argv[5] );
	}
	else if (argc == 8 && strcmp(argv[1],"-test") == 0) {
		ret = frcnn_test( argv[2], argv[3], argv[4], argv[5], atoi(argv[6]), atoi(argv[7]) );
	}
	else
	{
		cout << "usage:\n" << endl;
		cout << "\tDemo -DataChange_ShuffleData input output\n" << endl;
		cout << "\tDemo -DataChange_inmutilabl64class szXmlList loadImagePath keyword svPath\n" << endl;
		cout << "\tDemo -DataChange_itugodoudou383class loadData loadImagePath keyword svPath\n" << endl;
		cout << "\tDemo -DataChange_sogou98class loadData loadImagePath keyword svPath\n" << endl;
		cout << "\tDemo -test loadImagePath svPath keyfile layerName binGPU deviceID\n" << endl;
		return ret;
	}
	return ret;
}

