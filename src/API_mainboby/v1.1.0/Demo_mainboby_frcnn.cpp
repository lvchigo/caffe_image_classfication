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
#include "API_mainboby.h"
#include "TErrorCode.h"
#include "plog/Log.h"

#include "kyheader.h"
#include "Objectness_predict.h"
#include "ValStructVec.h"
#include "CmShow.h"

#include "tinyxml2.hpp"	//read xml

using namespace cv;
using namespace std;
using namespace tinyxml2;

int Get_image_Num( char *szQueryList, char* savePath, int MaxSingleClassNum )
{
	char tPath[256];
	char loadImgPath[256];
	char szImgPath[256];
	int i, j, label, svImg, nRet = 0;
	unsigned long long nCount,ImageID = 0;
	string strIID;
	FILE *fpListFile = 0;

	/********************************Init*****************************/
	API_COMMEN api_commen;
	
	/********************************Open Query List*****************************/
	fpListFile = fopen(szQueryList,"r");
	if (!fpListFile) 
	{
		cout << "0.can't open " << szQueryList << endl;
		return TEC_INVALID_PARAM;
	}

	/*****************************Process one by one*****************************/
	nCount = 0;
	while(EOF != fscanf(fpListFile, "%s", &loadImgPath ))
	{
		if (nCount>MaxSingleClassNum)
			continue;

		IplImage *img = cvLoadImage(loadImgPath);
		if(!img || (img->width<128) || (img->height<128) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
		{	
			cout<<"Can't open " << loadImgPath << endl;
			continue;
		}	
		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld img...\n",nCount);

		/************************getRandomID*****************************/
		//api_commen.getRandomID( ImageID );
		strIID = api_commen.GetStringIDFromFilePath( loadImgPath );

		sprintf( szImgPath, "%s/%s.jpg", savePath, strIID.c_str() );
		cvSaveImage( szImgPath, img );

		cvReleaseImage(&img);img = 0;
	}
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	

	printf( "all load img:%lld\n", nCount );
	cout<<"Done!! "<<endl;
	
	return nRet;
}	

int Get_Big_image( char *szQueryList, char* savePath, char* ClassName, int MaxSingleClassNum )
{
	char tPath[256];
	char loadImgPath[256];
	char szImgPath[256];
	int i, j, svImg, nRet = 0;
	unsigned long long nCount,ImageID = 0;
	string strImageID;
	FILE *fpListFile = 0;

	/********************************Init*****************************/
	API_COMMEN api_commen;
	
	/********************************Open Query List*****************************/
	fpListFile = fopen(szQueryList,"r");
	if (!fpListFile) 
	{
		cout << "0.can't open " << szQueryList << endl;
		return TEC_INVALID_PARAM;
	}

	/*****************************Process one by one*****************************/
	nCount = 0;
	while(EOF != fscanf(fpListFile, "%s", &loadImgPath))
	{
		IplImage *img = cvLoadImage(loadImgPath);
		if(!img || (img->width<128) || (img->height<128) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
		{	
			cout<<"Can't open " << loadImgPath << endl;
			cvReleaseImage(&img);img = 0;
			continue;
		}	

		/************************getRandomID*****************************/
		//api_commen.getRandomID( ImageID );
		//ImageID = api_commen.GetIDFromFilePath( loadImgPath );
		strImageID = api_commen.GetStringIDFromFilePath(loadImgPath);

		sprintf( szImgPath, "%s/%s/%s.jpg", savePath, ClassName, strImageID.c_str() );
		cvSaveImage( szImgPath, img );
		cvReleaseImage(&img);img = 0;

		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld img...\n",nCount);
		if (nCount>=MaxSingleClassNum)
			break;
	}
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	
	
	cout<<"Done!! "<<endl;
	
	return nRet;
}	

int Get_VocData( char *szQueryList, char* inImgPath, char* inXmlPath, char* imgSavePath, char* xmlSavePath )
{
	char tPath[256];
	char loadImgPath[256];
	char szImgPath[256];
	int i, j, svImg, nRet = 0;
	unsigned long long nCount,ImageID = 0;
	string strImageID;
	FILE *fpListFile = 0;

	/********************************Init*****************************/
	API_COMMEN api_commen;
	
	/********************************Open Query List*****************************/
	fpListFile = fopen(szQueryList,"r");
	if (!fpListFile) 
	{
		cout << "0.can't open " << szQueryList << endl;
		return TEC_INVALID_PARAM;
	}

	/*****************************Process one by one*****************************/
	nCount = 0;
	while(EOF != fscanf(fpListFile, "%s", &loadImgPath))
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
		strImageID = api_commen.GetStringIDFromFilePath(loadImgPath);

		//cp img file
		sprintf( tPath, "cp %s/%s.jpg %s/%s.jpg", inImgPath, strImageID.c_str(), imgSavePath, strImageID.c_str() );
		system( tPath );

		//cp xml file
		sprintf( tPath, "cp %s/%s.xml %s/%s.xml", inXmlPath, strImageID.c_str(), xmlSavePath, strImageID.c_str() );
		system( tPath );
		
		cvReleaseImage(&img);img = 0;

		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld img...\n",nCount);
	}
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	
	
	cout<<"Done!! "<<endl;
	
	return nRet;
}	


int Change_XML_Name( char *szQueryList, char* imgPath, char* xmlSavePath, char* imgSavePath )
{
	char tPath[256];
	char loadImgPath[256];
	char szImgPath[256];
	int i, j, svImg, nRet = 0;
	unsigned long long nCount, labelCount, ImageID = 0;
	FILE *fpListFile = 0;

	/********************************Init*****************************/
	string strLabel;
	string filename,tmpPath,iid;
	long end = 0;

	map< string,int > 				mapLabel;
	map< string, int >::iterator 	itLabel;

	API_MAINBOBY api_mainboby;
	vector< pair< string, Vec4i > > 	vecLabelRect;
	
	/********************************Open Query List*****************************/
	fpListFile = fopen(szQueryList,"r");
	if (!fpListFile) 
	{
		cout << "0.can't open " << szQueryList << endl;
		return TEC_INVALID_PARAM;
	}

	/*****************************Process one by one*****************************/
	nCount = 0;
	while(EOF != fscanf(fpListFile, "%s", &loadImgPath))
	{	
		// reading the xml files
		nRet = api_mainboby.load_xml( string(loadImgPath), filename, vecLabelRect );
		if ( (nRet!=0) || (vecLabelRect.size()<1) )
		{
			cout << "Err to load_xml!" << endl;
			continue;
		}
		
		for(i=0;i<vecLabelRect.size();i++)
		{
			strLabel = vecLabelRect[i].first;
			//write map
			itLabel = mapLabel.find(strLabel);		
			if (itLabel != mapLabel.end()) // find it
				mapLabel[itLabel->first] = itLabel->second+1;		//[In]dic code-words
			else
				mapLabel[strLabel] = 1;
		}

		//cp xml file
		nRet = api_mainboby.write_xml( filename, xmlSavePath, vecLabelRect );
		if (nRet!=0)
		{
			cout << "Err to write_xml!" << endl;
			continue;
		}

		//cp img file
		sprintf( tPath, "cp %s/%s.jpg %s/%s.jpg", imgPath, filename.c_str(), imgSavePath, filename.c_str() );
		system( tPath );

		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld img...\n",nCount);
	}
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	

	//write label file
	sprintf( tPath, "Dict_FRCNN_%dlabel.txt", mapLabel.size() );
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

#define VOC_LABEL_NUM 20

int Get_Voc2Xml( char *szQueryList, char* imgPath, char* xmlSavePath, char* imgSavePath, int maxNum )
{
	char tPath[256];
	char loadImgPath[256];
	char szImgPath[256];
	int i, j, svImg, bin_write, bin_write_full, nRet = 0;
	unsigned long long nCount, labelCount, ImageID = 0;
	int binSameLabel = 1;

	const string vocName[VOC_LABEL_NUM] = {
		"aeroplane","bicycle","bird","boat","bottle",
		"bus","car","cat","chair","cow",
		"diningtable","dog","horse","motorbike","person",
		"pottedplant","sheep","sofa","train","tvmonitor"};
	const string frcnnName[VOC_LABEL_NUM] = {
		"goods.airplane.airplane","goods.car.bicycle","pet.bird.bird","goods.ship.ship","goods.bottle.bottle",
		"goods.car.bus","goods.car.car","pet.cat.cat","goods.chair.chair","pet.cow.cow",
		"food.diningtable.diningtable","pet.dog.dog","pet.horse.horse","goods.car.motorbike","people.people.people",
		"goods.pottedplant.pottedplant","pet.sheep.sheep","goods.sofa.sofa","goods.car.train","goods.tvmonitor.tvmonitor"};
	
	FILE *fpListFile = 0;

	/********************************Init*****************************/
	string label;
	string filename,tmpPath,iid;
	long end = 0;

	API_MAINBOBY api_mainboby;

	map< string, int > 					mapLabelNum;
	map< string, int >::iterator 		itLabelNum;

	map< string, string > 				mapVocInName;
	map< string, string >::iterator 	itVocInName;
	vector< pair< string, Vec4i > > 	vecLabelRect;

	//write voc_in_name
	for(i=0;i<VOC_LABEL_NUM;i++)
	{
		mapVocInName[vocName[i]] = frcnnName[i];
	}
	
	/********************************Open Query List*****************************/
	fpListFile = fopen(szQueryList,"r");
	if (!fpListFile) 
	{
		cout << "0.can't open " << szQueryList << endl;
		return TEC_INVALID_PARAM;
	}

	/*****************************Process one by one*****************************/
	nCount = 0;
	mapLabelNum.clear();
	while(EOF != fscanf(fpListFile, "%s", &loadImgPath))
	{
		/* reading the xml files */
		XMLDocument doc;
		XMLError eResult = doc.LoadFile(const_cast<char*>(loadImgPath));
		if (eResult != XML_SUCCESS) {
			cout<<"Error opening the xml files! "<<loadImgPath<<endl;
			continue;
		}

		XMLNode* pRoot = doc.FirstChildElement("annotation");
		if (pRoot == nullptr) {
			cout<<"No annotation available!"<<endl;
			continue;
		}
		
		XMLElement* pNameElement = pRoot->FirstChildElement("filename");
		if (pNameElement == nullptr) {
			cout<<"No filename here!"<<endl;
			continue;
		}

		//read name 
		const char * Name = pNameElement->GetText();
		//printf("Name:%s\n",Name);

		filename = Name;
		std::size_t found = filename.find(".jpg");
		if ( (found!=std::string::npos) && (found>0) )
		{
			end = 0;
			tmpPath = filename;
			end = tmpPath.find_last_of(".");
			if (end>0)
			{
				iid = tmpPath.substr(0,end);
				filename = iid;
			}
		}
		//printf("change Name:%s,end:%d\n",iid.c_str(),end);

		//read name
		XMLElement* pObjectElement = pRoot->FirstChildElement("object");
		if (pObjectElement == nullptr) {
			cout<<"No filename here!"<<endl;
			continue;
		}

		//printf("label:");
		vecLabelRect.clear();
		binSameLabel = 1;	//init all same label
		while (pObjectElement)  
		{  
			XMLElement* pLabelElement = pObjectElement->FirstChildElement("name");
			if (pLabelElement == nullptr) {
				cout<<"No label here!"<<endl;
				pObjectElement = pObjectElement->NextSiblingElement("object");  
				continue;
			}
			
			const char * label = pLabelElement->GetText();
			//printf("%s ",label);

			//write map
			itVocInName = mapVocInName.find(string(label));		
			if (itVocInName != mapVocInName.end()) // find it
			{
				XMLElement* pBndboxElement = pObjectElement->FirstChildElement("bndbox");
				if (pBndboxElement == nullptr) {
					cout<<"No bndbox here!"<<endl;
					pObjectElement = pObjectElement->NextSiblingElement("object");  
					continue;
				}

				XMLElement* pXminElement = pBndboxElement->FirstChildElement("xmin");
				if (pXminElement == nullptr) {
					cout<<"No xmin here!"<<endl;
					pObjectElement = pObjectElement->NextSiblingElement("object");  
					continue;
				}
				XMLElement* pYminElement = pBndboxElement->FirstChildElement("ymin");
				if (pYminElement == nullptr) {
					cout<<"No ymin here!"<<endl;
					pObjectElement = pObjectElement->NextSiblingElement("object");  
					continue;
				}
				XMLElement* pXmaxElement = pBndboxElement->FirstChildElement("xmax");
				if (pXmaxElement == nullptr) {
					cout<<"No xmax here!"<<endl;
					pObjectElement = pObjectElement->NextSiblingElement("object");  
					continue;
				}
				XMLElement* pYmaxElement = pBndboxElement->FirstChildElement("ymax");
				if (pYmaxElement == nullptr) {
					cout<<"No ymax here!"<<endl;
					pObjectElement = pObjectElement->NextSiblingElement("object");  
					continue;
				}
				
				const char * xmin = pXminElement->GetText();
				const char * ymin = pYminElement->GetText();
				const char * xmax = pXmaxElement->GetText();
				const char * ymax = pYmaxElement->GetText();

				Vec4i rect;
				rect[0] = (atoi(xmin)<1)?(atoi(xmin)+1):(atoi(xmin));
				rect[1] = (atoi(ymin)<1)?(atoi(ymin)+1):(atoi(ymin));
				rect[2] = (atoi(xmax)<1)?(atoi(xmax)+1):(atoi(xmax));
				rect[3] = (atoi(ymax)<1)?(atoi(ymax)+1):(atoi(ymax));

				if ( ( atoi(xmax)-atoi(xmin) < 32 ) || (atoi(ymax)-atoi(ymin) < 32) )
				{
					pObjectElement = pObjectElement->NextSiblingElement("object");  
					continue;
				}
				
				vecLabelRect.push_back( std::make_pair( itVocInName->second, rect ) );

				if ( (vecLabelRect.size()>1) && ( itVocInName->second != vecLabelRect[0].first ) )
					binSameLabel = 0;	//2or more labels
			}
			else
			{
				vecLabelRect.clear();
				break;
			}
		
			pObjectElement = pObjectElement->NextSiblingElement("object");  
		} 
		//printf("\n");

		if ( vecLabelRect.size() < 1)
			continue;

		//judge full
		bin_write = 1;
		bin_write_full = 0;
		for(itLabelNum = mapLabelNum.begin(); itLabelNum != mapLabelNum.end(); itLabelNum++)
		{
			if (itLabelNum->second > maxNum)
				bin_write_full++;
			else
				break;
		}
		if ( bin_write_full == VOC_LABEL_NUM )
		{
			bin_write = 0;
			break;
		}

		//save maxNum
		for (i=0;i<vecLabelRect.size();i++)
		{
			itLabelNum = mapLabelNum.find(vecLabelRect[i].first);		
			if (itLabelNum != mapLabelNum.end()) // find it
			{
				if ( (itLabelNum->second > maxNum) && ( binSameLabel == 1 ) )
				{
					bin_write = 0;
					break;
				}
				
				mapLabelNum[vecLabelRect[i].first] += 1;
			}
			else
			{
				mapLabelNum[vecLabelRect[i].first] = 1;
			}
		}

		if ( bin_write == 0 )
			continue;
		
		//write_xml
		nRet = api_mainboby.write_xml( filename, xmlSavePath, vecLabelRect );
		if (nRet!=0)
		{
			cout << "Err to write_xml!" << endl;
			continue;
		}

		//cp img file
		sprintf( tPath, "cp %s/%s.jpg %s/%s.jpg", imgPath, filename.c_str(), imgSavePath, filename.c_str() );
		system( tPath );

		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld img...\n",nCount);
	}
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	

	//print info
	printf("label:\n");
	labelCount = 0;
	for(itLabelNum = mapLabelNum.begin(); itLabelNum != mapLabelNum.end(); itLabelNum++)
	{
		printf("%s %d\n", itLabelNum->first.c_str(), itLabelNum->second);

		labelCount += itLabelNum->second;
	}
	printf("All load img:%lld, save label:%lld\n", nCount, labelCount);

	cout<<"Done!! "<<endl;
	
	return nRet;
}	


int Get_ImageNetXml2Xml( char *szQueryList, char* xmlSavePath )
{
	char tPath[256];
	char loadImgPath[256];
	char szImgPath[256];
	int i, j, svImg, nRet = 0;
	unsigned long long nCount, labelCount, ImageID = 0;
	
	FILE *fpListFile = 0;

	/********************************Init*****************************/
	string label;
	string filename,tmpPath,iid;
	long end = 0;
	API_MAINBOBY api_mainboby;

	vector< pair< string, Vec4i > > 	vecLabelRect;
	
	/********************************Open Query List*****************************/
	fpListFile = fopen(szQueryList,"r");
	if (!fpListFile) 
	{
		cout << "0.can't open " << szQueryList << endl;
		return TEC_INVALID_PARAM;
	}

	/*****************************Process one by one*****************************/
	nCount = 0;
	while(EOF != fscanf(fpListFile, "%s", &loadImgPath))
	{
		/* reading the xml files */
		XMLDocument doc;
		XMLError eResult = doc.LoadFile(const_cast<char*>(loadImgPath));
		if (eResult != XML_SUCCESS) {
			cout<<"Error opening the xml files! "<<loadImgPath<<endl;
			continue;
		}

		XMLNode* pRoot = doc.FirstChildElement("annotation");
		if (pRoot == nullptr) {
			cout<<"No annotation available!"<<endl;
			continue;
		}
		
		XMLElement* pNameElement = pRoot->FirstChildElement("filename");
		if (pNameElement == nullptr) {
			cout<<"No filename here!"<<endl;
			continue;
		}

		//read name 
		const char * Name = pNameElement->GetText();
		//printf("Name:%s\n",Name);

		filename = Name;
		std::size_t found = filename.find(".jpg");
		if ( (found!=std::string::npos) && (found>0) )
		{
			end = 0;
			tmpPath = filename;
			end = tmpPath.find_last_of(".");
			if (end>0)
			{
				iid = tmpPath.substr(0,end);
				filename = iid;
			}
		}
		//printf("change Name:%s,end:%d\n",iid.c_str(),end);

		//read name
		XMLElement* pObjectElement = pRoot->FirstChildElement("object");
		if (pObjectElement == nullptr) {
			cout<<"No filename here!"<<endl;
			continue;
		}

		//printf("label:");
		vecLabelRect.clear();
		while (pObjectElement)  
		{  
			XMLElement* pLabelElement = pObjectElement->FirstChildElement("name");
			if (pLabelElement == nullptr) {
				cout<<"No label here!"<<endl;
				continue;
			}
			
			const char * label = pLabelElement->GetText();
			//printf("%s ",label);

			//write map
			{
				XMLElement* pBndboxElement = pObjectElement->FirstChildElement("bndbox");
				if (pBndboxElement == nullptr) {
					cout<<"No bndbox here!"<<endl;
					continue;
				}

				XMLElement* pXminElement = pBndboxElement->FirstChildElement("xmin");
				if (pXminElement == nullptr) {
					cout<<"No xmin here!"<<endl;
					continue;
				}
				XMLElement* pYminElement = pBndboxElement->FirstChildElement("ymin");
				if (pYminElement == nullptr) {
					cout<<"No ymin here!"<<endl;
					continue;
				}
				XMLElement* pXmaxElement = pBndboxElement->FirstChildElement("xmax");
				if (pXmaxElement == nullptr) {
					cout<<"No xmax here!"<<endl;
					continue;
				}
				XMLElement* pYmaxElement = pBndboxElement->FirstChildElement("ymax");
				if (pYmaxElement == nullptr) {
					cout<<"No ymax here!"<<endl;
					continue;
				}
				
				const char * xmin = pXminElement->GetText();
				const char * ymin = pYminElement->GetText();
				const char * xmax = pXmaxElement->GetText();
				const char * ymax = pYmaxElement->GetText();

				Vec4i rect;
				rect[0] = (atoi(xmin)<1)?(atoi(xmin)+1):(atoi(xmin));
				rect[1] = (atoi(ymin)<1)?(atoi(ymin)+1):(atoi(ymin));
				rect[2] = (atoi(xmax)<1)?(atoi(xmax)+1):(atoi(xmax));
				rect[3] = (atoi(ymax)<1)?(atoi(ymax)+1):(atoi(ymax));
				vecLabelRect.push_back( std::make_pair( string(label), rect ) );
			}
		
			pObjectElement = pObjectElement->NextSiblingElement("object");  
		} 
		//printf("\n");

		if ( vecLabelRect.size() < 1)
			continue;
			
		//write_xml
		nRet = api_mainboby.write_xml( filename, xmlSavePath, vecLabelRect );
		if (nRet!=0)
		{
			cout << "Err to write_xml!" << endl;
			continue;
		}

		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld img...\n",nCount);
	}
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	

	cout<<"Done!! "<<endl;
	
	return nRet;
}	

int Get_Img2Xml( char *szQueryList, char* xmlSavePath, char* imgSavePath, char* labelname )
{
	char tPath[256];
	char loadImgPath[256];
	char szImgPath[256];
	int i, j, label, svImg, nRet = 0;
	unsigned long long nCount,ImageID = 0;
	FILE *fpListFile = 0;

	/********************************Init*****************************/
	API_COMMEN api_commen;
	API_MAINBOBY api_mainboby;
	vector< pair< string, Vec4i > > vecLabelRect;
	
	/********************************Open Query List*****************************/
	fpListFile = fopen(szQueryList,"r");
	if (!fpListFile) 
	{
		cout << "0.can't open " << szQueryList << endl;
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

		//strImageID = api_commen.GetStringIDFromFilePath( loadImgPath );
		api_commen.getRandomID( ImageID );
		//ImageID = api_commen.GetIDFromFilePath( loadImgPath );

		//write_xml
		Vec4i rect;
		rect[0] = 1;
		rect[1] = 1;
		rect[2] = img->width-1;
		rect[3] = img->height-1;
		
		vecLabelRect.clear();
		vecLabelRect.push_back( std::make_pair( labelname, rect ) );
		sprintf( tPath, "%lld", ImageID );
		nRet = api_mainboby.write_xml( string(tPath), xmlSavePath, vecLabelRect );
		if (nRet!=0)
		{
			cout << "Err to write_xml!" << endl;
			continue;
		}

		//cp img file
		sprintf( tPath, "cp %s %s/%lld.jpg", loadImgPath, imgSavePath, ImageID );
		system( tPath );

		cvReleaseImage(&img);img = 0;
	}
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	

	sleep(1);
	cout<<"Done!! "<<endl;
	
	return nRet;
}	

int Check_XML_Roi_Img( char *szQueryList, char* imgPath, char* imgSavePath, int binVOC )
{
	char tPath[256];
	char loadImgPath[256];
	char szImgPath[256];
	int i, j, svImg, T_W,T_H, nRet = 0;
	unsigned long long nCount, labelNum, ImageID = 0;
	FILE *fpListFile = 0;

	/********************************Init*****************************/
	string label, text;
	string filename,tmpPath,iid;
	long end = 0;
	vector< pair< int, int > > vecXmlWH;

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

	/*****************************Process one by one*****************************/
	nCount = 0;
	while(EOF != fscanf(fpListFile, "%s", &loadImgPath))
	{
		/* reading the xml files */
		XMLDocument doc;
		XMLError eResult = doc.LoadFile(const_cast<char*>(loadImgPath));
		if (eResult != XML_SUCCESS) {
			cout<<"Error opening the xml files! "<<loadImgPath<<endl;
			continue;
		}

		XMLNode* pRoot = doc.FirstChildElement("annotation");
		if (pRoot == nullptr) {
			cout<<"No annotation available!"<<endl;
			continue;
		}
		
		XMLElement* pNameElement = pRoot->FirstChildElement("filename");
		if (pNameElement == nullptr) {
			cout<<"No filename here!"<<endl;
			continue;
		}

		//read name 
		const char * Name = pNameElement->GetText();
		//printf("Name:%s\n",Name);

		filename = Name;
		std::size_t found = filename.find(".jpg");
		if ( (found!=std::string::npos) && (found>0) )
		{
			end = 0;
			tmpPath = filename;
			end = tmpPath.find_last_of(".");
			if (end>0)
			{
				iid = tmpPath.substr(0,end);
				filename = iid;
			}
		}
		//printf("change Name:%s,end:%d\n",iid.c_str(),end);

		//read name
		XMLElement* pObjectElement = pRoot->FirstChildElement("object");
		if (pObjectElement == nullptr) {
			cout<<"No filename here!"<<endl;
			continue;
		}
		
		sprintf(szImgPath, "%s/%s.jpg",imgPath,filename.c_str());
		IplImage *img = cvLoadImage(szImgPath);
		if(!img || (img->width<64) || (img->height<64) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
		{	
			cout<<"Can't open " << szImgPath << endl;
			cvReleaseImage(&img);img = 0;
			continue;
		}	
		
		T_W = (img->width*0.3>64)?int(img->width*0.3):64;
		T_H = (img->height*0.3>64)?int(img->height*0.3):64;

		/************************change img*****************************/
		Mat matImg(img);

		//printf("label:");
		labelNum = 0;
		vecXmlWH.clear();
		while (pObjectElement)	
		{  
			XMLElement* pLabelElement = pObjectElement->FirstChildElement("name");
			if (pLabelElement == nullptr) {
				cout<<"No label here!"<<endl;
				continue;
			}
			
			const char * label = pLabelElement->GetText();
			//printf("%s ",label);

			//write map
			XMLElement* pBndboxElement = pObjectElement->FirstChildElement("bndbox");
			if (pBndboxElement == nullptr) {
				cout<<"No bndbox here!"<<endl;
				continue;
			}

			XMLElement* pXminElement = pBndboxElement->FirstChildElement("xmin");
			if (pXminElement == nullptr) {
				cout<<"No xmin here!"<<endl;
				continue;
			}
			XMLElement* pYminElement = pBndboxElement->FirstChildElement("ymin");
			if (pYminElement == nullptr) {
				cout<<"No ymin here!"<<endl;
				continue;
			}
			XMLElement* pXmaxElement = pBndboxElement->FirstChildElement("xmax");
			if (pXmaxElement == nullptr) {
				cout<<"No xmax here!"<<endl;
				continue;
			}
			XMLElement* pYmaxElement = pBndboxElement->FirstChildElement("ymax");
			if (pYmaxElement == nullptr) {
				cout<<"No ymax here!"<<endl;
				continue;
			}
			
			const char * xmin = pXminElement->GetText();
			const char * ymin = pYminElement->GetText();
			const char * xmax = pXmaxElement->GetText();
			const char * ymax = pYmaxElement->GetText();

			/************************save img data*****************************/
			Scalar color = colors[labelNum%8];
			rectangle( matImg, cvPoint(atoi(xmin), atoi(ymin)),
	                   cvPoint(atoi(xmax), atoi(ymax)), color, 2, 8, 0);
			putText(matImg, string(label), cvPoint(1, labelNum*20+20), FONT_HERSHEY_SIMPLEX, 0.5, color, 2);	
			if (labelNum == 0)
				sprintf(szImgPath, "%s/%s/%s.jpg", imgSavePath, label, filename.c_str() );

			labelNum++;
			vecXmlWH.push_back( std::make_pair( (atoi(xmax)-atoi(xmin)), (atoi(ymax)-atoi(ymin)) ) );
			pObjectElement = pObjectElement->NextSiblingElement("object");	
		} 
		//printf("\n");

		if ( ( binVOC == 1 ) && (( labelNum!=1) || 
			(( labelNum==1)&&((vecXmlWH[0].first<T_W) || (vecXmlWH[0].second<T_H))))  )
		{
			cvReleaseImage(&img);img = 0;
			continue;
		}
		
		imwrite( szImgPath, matImg );

		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld img...\n",nCount);

		cvReleaseImage(&img);img = 0;
	}
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	

	printf( "nCount:%ld\n", nCount );
	cout<<"Done!! "<<endl;
	
	return nRet;
}


#define GET_BING_ROI_TRAINING 1	//for caffe training net

int Get_Bing_ROI( char *szQueryList, char* KeyFilePath, char *layerName, int binGPU, int deviceID )
{
	char tPath[256];
	char loadImgPath[256];
	char szImgPath[256];
	int i, j, label, svImg, nRet = 0;
	int roiNum, topN=15;
	long inputLabel, nCount;
	unsigned long long ImageID = 0;
	string strImageID;
	double allPredictTime;
	FILE *fpListFile = 0;

	RunTimer<double> run;
	API_COMMEN api_commen;
	API_MAINBOBY api_mainboby;

	vector < pair < string,float > > Res;

	/***********************************Init*************************************/
	vector< pair<float, Vec4i> > boxHypothese;
	const static Scalar colors[] =  { 	CV_RGB(0,0,255),	CV_RGB(0,255,255),	CV_RGB(0,255,0),	
										CV_RGB(255,255,0),	CV_RGB(255,0,0),	CV_RGB(255,0,255),
										CV_RGB(0,128,255), 	CV_RGB(255,128,0)} ;
	/***********************************Init**********************************/
	plog::init(plog::info, "plog.txt"); 
	
	/********************************Open Query List*****************************/
	fpListFile = fopen(szQueryList,"r");
	if (!fpListFile) 
	{
		cout << "0.can't open " << szQueryList << endl;
		return TEC_INVALID_PARAM;
	}

#ifdef GET_BING_ROI_TRAINING
	FILE *fpListFile_training = fopen("BING_ROI_TRAINING.txt","w");
	if (!fpListFile_training) 
	{
		cout << "0.can't open " << "BING_ROI_TRAINING.txt" << endl;
		return TEC_INVALID_PARAM;
	}

	FILE *fpListFile_trainval = fopen("trainval.txt","w");
	if (!fpListFile_trainval) 
	{
		cout << "0.can't open " << "trainval.txt" << endl;
		return TEC_INVALID_PARAM;
	}
#endif

	/***********************************Init*************************************/
	nRet = api_mainboby.Init( KeyFilePath, layerName, binGPU, deviceID ); 
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
		if(!img || (img->width<128) || (img->height<128) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
		{	
			cout<<"Can't open " << loadImgPath << endl;
			continue;
		}	
		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld img...\n",nCount);

		/************************getRandomID*****************************/
#ifdef GET_BING_ROI_TRAINING
		strImageID = api_commen.GetStringIDFromFilePath( loadImgPath );
#else
		api_commen.getRandomID( ImageID );
		//ImageID = api_commen.GetIDFromFilePath( loadImgPath );
#endif

		/************************ResizeImg*****************************/
		IplImage* imgResize = api_mainboby.ResizeImg( img );

		/************************Get_Bing_Hypothese*****************************/	
		boxHypothese.clear();
		run.start();
#ifdef GET_BING_ROI_TRAINING
		//BinTraining:2-NO Remove Rectfor Training;1-Remove small Rect for Training;0-Remove small Rect for Test
		nRet = api_mainboby.Get_Bing_Hypothese( img, boxHypothese, 2 ); 	//(not resize img && return 1000 roi)
#else
		nRet = api_mainboby.Get_Bing_Hypothese( imgResize, boxHypothese );
#endif
		if ( (nRet!=0) || (boxHypothese.size()<1) )
		{
			LOOGE<<"[Get_Bing_Hypothese Err!!loadImgPath:]"<<loadImgPath;
			continue;
		}
		run.end();
		LOOGI<<"[Get_Bing_Hypothese] time:"<<run.time();
		allPredictTime += run.time();

#ifdef GET_BING_ROI_TRAINING
		/************************save roi data*****************************/
		fprintf(fpListFile_training, "%s %d", strImageID.c_str(), boxHypothese.size() );
		for ( j=0;j<boxHypothese.size();j++ )
		{
			fprintf(fpListFile_training, " %d %d %d %d", boxHypothese[j].second[0], boxHypothese[j].second[1],
	        	boxHypothese[j].second[2], boxHypothese[j].second[3] );
		}
		fprintf(fpListFile_training, "\n");

		//write trainval.txt
		fprintf(fpListFile_trainval, "%s\n", strImageID.c_str() );
#else
		Mat matImg(imgResize);
		/************************save img data*****************************/
		roiNum = (boxHypothese.size()>topN)?topN:boxHypothese.size();
		for(i=0;i<roiNum;i++) 
		//for(i=boxTests.size()-1;i>boxTests.size()-1-topN;i--) 
		{	
			Scalar color = colors[i%8];
			rectangle( matImg, cvPoint(boxHypothese[i].second[0], boxHypothese[i].second[1]),
	                   cvPoint(boxHypothese[i].second[2], boxHypothese[i].second[3]), color, 3, 8, 0);
			sprintf(szImgPath, "res_Hypothese/%lld_res.jpg",ImageID);
			imwrite( szImgPath, matImg );
		}
#endif

		cvReleaseImage(&imgResize);imgResize = 0;
		cvReleaseImage(&img);img = 0;
	}

	/*********************************close file*************************************/
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	
	
#ifdef GET_BING_ROI_TRAINING
	if (fpListFile_training) {fclose(fpListFile_training);fpListFile_training = 0;}	
	if (fpListFile_trainval) {fclose(fpListFile_trainval);fpListFile_trainval = 0;}	
#endif

	/*********************************Release*************************************/
	api_mainboby.Release();

	/*********************************Print Info*********************************/
	if ( nCount != 0 ) 
	{
		printf( "nCount:%ld,PredictTime:%.4fms\n", nCount, allPredictTime*1000.0/nCount );
	}
	
	cout<<"Done!! "<<endl;
	
	return nRet;
}	

int Get_Xml_Bing_ROI( char *szQueryList, char* loadXmlPath, char* KeyFilePath, char *layerName, int binGPU, int deviceID )
{
	char tPath[256];
	char loadImgPath[256];
	char szImgPath[256];
	int i, j, label, svImg, nRet = 0;
	int roiNum, topN=15;
	long inputLabel, nCount, svBingPos, svBingNeg, countIOU_0_7;
	unsigned long long ImageID = 0;
	string strImageID;
	double allPredictTime;
	FILE *fpListFile = 0;

	RunTimer<double> run;
	API_COMMEN api_commen;
	API_MAINBOBY api_mainboby;

	vector < pair < string,float > > Res;

	/***********************************Init*************************************/
	vector< pair<float, Vec4i> > boxHypothese;
	const static Scalar colors[] =  { 	CV_RGB(0,0,255),	CV_RGB(0,255,255),	CV_RGB(0,255,0),	
										CV_RGB(255,255,0),	CV_RGB(255,0,0),	CV_RGB(255,0,255),
										CV_RGB(0,128,255), 	CV_RGB(255,128,0)} ;
	/***********************************Init**********************************/
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

	FILE *fpListFile_training = fopen("BING_ROI_TRAINING.txt","w");
	if (!fpListFile_training) 
	{
		cout << "0.can't open " << "BING_ROI_TRAINING.txt" << endl;
		return TEC_INVALID_PARAM;
	}

	FILE *fpListFile_trainval = fopen("trainval.txt","w");
	if (!fpListFile_trainval) 
	{
		cout << "0.can't open " << "trainval.txt" << endl;
		return TEC_INVALID_PARAM;
	}

	/***********************************Init*************************************/
	nRet = api_mainboby.Init( KeyFilePath, layerName, binGPU, deviceID ); 
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
		if(!img || (img->width<128) || (img->height<128) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
		{	
			cout<<"Can't open " << loadImgPath << endl;
			continue;
		}	

		/************************getRandomID*****************************/
		strImageID = api_commen.GetStringIDFromFilePath( loadImgPath );

		/************************loadXml*****************************/
		vecXmlLabelRect.clear();
		sprintf(szImgPath, "%s/%s.xml",loadXmlPath,strImageID.c_str());
		nRet = api_mainboby.load_xml( szImgPath, xmlImageID, vecXmlLabelRect );
		if ( (nRet!=0) || (vecXmlLabelRect.size()<1) )
		{
			LOOGE<<"[loadXml Err!!loadXml:]"<<szImgPath;
			continue;
		}

		/************************get xml Hypothese*****************************/
		vecOutXmlRect.clear();
		int binHypothese = 0; //0-normal,1-for mainbody v1.0.0
		nRet = api_mainboby.Get_Xml_Hypothese( xmlImageID, img->width, img->height, vecXmlLabelRect, vecOutXmlRect, binHypothese );
		if ( (nRet!=0) || (vecOutXmlRect.size()<1) )
		{
			LOOGE<<"[get_xml_Hypothese Err!!xmlImageID:]"<<xmlImageID;
			continue;
		}

		/************************Get_Bing_Hypothese*****************************/	
		boxHypothese.clear();
		run.start();
		//BinTraining:2-NO Remove Rectfor Training;1-Remove small Rect for Training;0-Remove small Rect for Test
		nRet = api_mainboby.Get_Bing_Hypothese( img, boxHypothese, 2 );
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
		nRet = api_mainboby.Get_iou_cover( boxHypothese, vecOutXmlRect, vectorIouCover );
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
		countIOU_0_7 = 0;
/*		for ( j=0;j<boxHypothese.size();j++ )
		{
			fprintf(fpListFile_training, " %d %d %d %d", boxHypothese[j].second[0], boxHypothese[j].second[1],
	        	boxHypothese[j].second[2], boxHypothese[j].second[3] );
	        svBingNeg++;
		}*/
		for ( j=0;j<vectorIouCover.size();j++ )
		{
			//if ( ( (vectorIouCover[j].second.first>=0.7) && (vectorIouCover[j].second.second==1) ) || 	//pos
			//	 ( vectorIouCover[j].second.first<=0.3 ) && (vectorIouCover[j].second.second!=1) )		//neg
			{
				fprintf(fpListFile_training, " %d %d %d %d", vectorIouCover[j].first[0], vectorIouCover[j].first[1],
		        	vectorIouCover[j].first[2], vectorIouCover[j].first[3] );

				if ( (vectorIouCover[j].second.first>=0.7) && (vectorIouCover[j].second.second==1) )	//pos
					svBingPos++;
				if ( ( vectorIouCover[j].second.first<=0.3 ) && (vectorIouCover[j].second.second!=1) )	//neg
					svBingNeg++;
			}

			//if (vectorIouCover[j].second.first>=0.7)
			//	countIOU_0_7++;			
		}
		fprintf(fpListFile_training, "\n");

		//write trainval.txt
		fprintf(fpListFile_trainval, "%s\n", strImageID.c_str() );

		nCount++;
		if( nCount%50 == 0 )
		{
			printf("Loaded %ld img...\n",nCount);
			printf("%s,xml:%d,bing:pos-%d,neg-%d,0.7-%d\n", 
				strImageID.c_str(), vecOutXmlRect.size(), svBingPos, svBingNeg, countIOU_0_7 );
		}
		//printf("%s,xml:%d,bing:pos-%d,neg-%d,0.7-%d\n", 
		//	strImageID.c_str(), vecOutXmlRect.size(), svBingPos, svBingNeg, countIOU_0_7 );

		cvReleaseImage(&img);img = 0;
	}

	/*********************************close file*************************************/
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	
	if (fpListFile_training) {fclose(fpListFile_training);fpListFile_training = 0;}	
	if (fpListFile_trainval) {fclose(fpListFile_trainval);fpListFile_trainval = 0;}	

	/*********************************Release*************************************/
	api_mainboby.Release();

	/*********************************Print Info*********************************/
	if ( nCount != 0 ) 
	{
		printf( "nCount:%ld,PredictTime:%.4fms\n", nCount, allPredictTime*1000.0/nCount );
	}
	
	cout<<"Done!! "<<endl;
	
	return nRet;
}	

int Get_Xml_Bing_ROI_for_1_0_0( char *szQueryList, char* loadXmlPath, char* svImagePath, char* KeyFilePath, char *layerName, int binGPU, int deviceID )
{
	char tPath[256];
	char loadImgPath[256];
	char szImgPath[256];
	char svImgFile[256];
	int i, j, label, svImg, nRet = 0;
	int roiNum, topN=15;
	long inputLabel, nCount, svBingPos, svBingNeg, countIOU_0_7;
	unsigned long long ImageID = 0;
	string strImageID;
	double allPredictTime;
	FILE *fpListFile = 0;

	RunTimer<double> run;
	API_COMMEN api_commen;
	API_MAINBOBY api_mainboby;

	vector < pair < string,float > > Res;

	/***********************************Init*************************************/
	vector< pair<float, Vec4i> > boxHypothese;
	const static Scalar colors[] =  { 	CV_RGB(0,0,255),	CV_RGB(0,255,255),	CV_RGB(0,255,0),	
										CV_RGB(255,255,0),	CV_RGB(255,0,0),	CV_RGB(255,0,255),
										CV_RGB(0,128,255), 	CV_RGB(255,128,0)} ;
	/***********************************Init**********************************/
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

	/***********************************Init*************************************/
	nRet = api_mainboby.Init( KeyFilePath, layerName, binGPU, deviceID ); 
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
		if(!img || (img->width<128) || (img->height<128) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
		{	
			cout<<"Can't open " << loadImgPath << endl;
			continue;
		}	

		/************************getRandomID*****************************/
		strImageID = api_commen.GetStringIDFromFilePath( loadImgPath );

		/************************loadXml*****************************/
		vecXmlLabelRect.clear();
		sprintf(szImgPath, "%s/%s.xml",loadXmlPath,strImageID.c_str());
		nRet = api_mainboby.load_xml( szImgPath, xmlImageID, vecXmlLabelRect );
		if ( (nRet!=0) || (vecXmlLabelRect.size()<1) )
		{
			LOOGE<<"[loadXml Err!!loadXml:]"<<szImgPath;
			continue;
		}

		/************************get xml Hypothese*****************************/
		vecOutXmlRect.clear();
		int binHypothese = 1; //0-normal,1-for mainbody v1.0.0
		nRet = api_mainboby.Get_Xml_Hypothese( xmlImageID, img->width, img->height, vecXmlLabelRect, vecOutXmlRect, binHypothese );
		if ( (nRet!=0) || (vecOutXmlRect.size()<1) )
		{
			LOOGE<<"[get_xml_Hypothese Err!!xmlImageID:]"<<xmlImageID;
			continue;
		}

		/************************Get_Bing_Hypothese*****************************/	
		boxHypothese.clear();
		run.start();
		//BinTraining:2-NO Remove Rectfor Training;1-Remove small Rect for Training;0-Remove small Rect for Test
		nRet = api_mainboby.Get_Bing_Hypothese( img, boxHypothese, 2 );
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
		nRet = api_mainboby.Get_iou_cover( boxHypothese, vecOutXmlRect, vectorIouCover );
		if ( (nRet!=0) || (vectorIouCover.size()<1) )
		{
			LOOGE<<"[Get_iou_cover Err!!loadImgPath:]"<<loadImgPath;
			continue;
		}

		/************************save pos roi data*****************************/
		for( i=0;i<vecOutXmlRect.size();i++)
		{
			cvSetImageROI( img,cvRect(vecOutXmlRect[i].second[0],vecOutXmlRect[i].second[1], 
				(vecOutXmlRect[i].second[2]-vecOutXmlRect[i].second[0]),
				(vecOutXmlRect[i].second[3]-vecOutXmlRect[i].second[1]) ) );	//for imagequality
			IplImage* MutiROI = cvCreateImage(cvGetSize(img),img->depth,img->nChannels);
			cvCopy( img, MutiROI, NULL );
			cvResetImageROI(img);	
			
			/*****************************Resize Img*****************************/
			IplImage *MutiROIResize = cvCreateImage(cvSize(256, 256), img->depth, img->nChannels);
			cvResize( MutiROI, MutiROIResize );

			sprintf( svImgFile, "%s/%s/%s_%d.jpg", svImagePath, vecOutXmlRect[i].first.c_str(), xmlImageID.c_str(), i );
			cvSaveImage( svImgFile, MutiROIResize );

			cvReleaseImage(&MutiROIResize);MutiROIResize = 0;
			cvReleaseImage(&MutiROI);MutiROI = 0;
		}

		/************************save neg roi data*****************************/
		for( i=0;i<vectorIouCover.size();i++)
		{
			if ( ( vectorIouCover[i].second.first<=0.2 ) && (vectorIouCover[i].second.second!=1) &&
				 ( vectorIouCover[i].first[2]-vectorIouCover[i].first[0]>=128) && 
				 ( vectorIouCover[i].first[3]-vectorIouCover[i].first[1]>=128) )
			{
				cvSetImageROI( img,cvRect(vectorIouCover[i].first[0],vectorIouCover[i].first[1], 
					(vectorIouCover[i].first[2]-vectorIouCover[i].first[0]),
					(vectorIouCover[i].first[3]-vectorIouCover[i].first[1]) ) );	//for imagequality
				IplImage* MutiROI = cvCreateImage(cvGetSize(img),img->depth,img->nChannels);
				cvCopy( img, MutiROI, NULL );
				cvResetImageROI(img);	
				
				/*****************************Resize Img*****************************/
				IplImage *MutiROIResize = cvCreateImage(cvSize(256, 256), img->depth, img->nChannels);
				cvResize( MutiROI, MutiROIResize );

				sprintf( svImgFile, "%s/neg/%s_%d.jpg", svImagePath, xmlImageID.c_str(), i );
				cvSaveImage( svImgFile, MutiROIResize );

				cvReleaseImage(&MutiROIResize);MutiROIResize = 0;
				cvReleaseImage(&MutiROI);MutiROI = 0;
			}
		}

		nCount++;
		if( nCount%50 == 0 )
		{
			printf("Loaded %ld img...\n",nCount);
		}

		cvReleaseImage(&img);img = 0;
	}

	/*********************************close file*************************************/
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	

	/*********************************Release*************************************/
	api_mainboby.Release();

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
	double allPredictTime;
	FILE *fpListFile = 0;

	RunTimer<double> run;
	API_COMMEN api_commen;
	API_MAINBOBY api_mainboby;

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
	nRet = api_mainboby.Init( KeyFilePath, layerName, binGPU, deviceID ); 
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
		if(!img || (img->width<128) || (img->height<128) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
		{	
			cout<<"Can't open " << loadImgPath << endl;
			continue;
		}	

		/************************getRandomID*****************************/
		api_commen.getRandomID( ImageID );
		//ImageID = api_commen.GetIDFromFilePath( loadImgPath );

		/************************ResizeImg*****************************/
		IplImage* imgResize = api_mainboby.ResizeImg( img );
		
		/************************Predict*****************************/	
		Res.clear();
		run.start();
		nRet = api_mainboby.Predict( imgResize, ImageID, layerName, Res );
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
		sprintf( tPath, "res_predict/%s/", Res[0].first.first.c_str() );
		for(i=0;i<Res.size();i++)  
		{				
			Scalar color = colors[i%8];
			rectangle( matImg, cvPoint(Res[i].first.second[0], Res[i].first.second[1]),
	                   cvPoint(Res[i].first.second[2], Res[i].first.second[3]), color, 2, 8, 0);

			sprintf(szImgPath, "%.2f %s", Res[i].second, Res[i].first.first.c_str() );
			text = szImgPath;
			//putText(matImg, text, cvPoint(Res[i].first.second[0]+1, Res[i].first.second[1]+20), 
			//	FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
			putText(matImg, text, cvPoint(1, i*20+20), FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
				
			sprintf(tPath, "%s%s-%.2f_", tPath, Res[i].first.first.c_str(), Res[i].second );
		}
		if ( saveImg == 0 )
		{
			sprintf( savePath, "%s%lld.jpg", tPath, ImageID );
			imwrite( savePath, matImg );
		}
		else if ( saveImg == 1 )
		{
			sprintf( savePath, "%s%lld.jpg", tPath, ImageID );
			cvSaveImage( savePath, imgResize );

			sprintf( savePath, "%s%lld_predict.jpg", tPath, ImageID );
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
	api_mainboby.Release();

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

	if (argc == 5 && strcmp(argv[1],"-get_img_num") == 0) {
		ret = Get_image_Num( argv[2], argv[3], atol(argv[4]) );
	}
	else if (argc == 6 && strcmp(argv[1],"-get_big_img") == 0) {
		ret = Get_Big_image( argv[2], argv[3], argv[4], atol(argv[5]) );
	}
	else if (argc == 7 && strcmp(argv[1],"-get_vocdata") == 0) {
		ret = Get_VocData( argv[2], argv[3], argv[4], argv[5], argv[6] );
	}
	else if (argc == 6 && strcmp(argv[1],"-ch_xml_name") == 0) {
		ret = Change_XML_Name( argv[2], argv[3], argv[4], argv[5] );
	}
	else if (argc == 7 && strcmp(argv[1],"-get_voc2xml") == 0) {
		ret = Get_Voc2Xml( argv[2], argv[3], argv[4], argv[5], atoi(argv[6]) );
	}
	else if (argc == 4 && strcmp(argv[1],"-get_imagenetxml2xml") == 0) {
		ret = Get_ImageNetXml2Xml( argv[2], argv[3] );
	}
	else if (argc == 6 && strcmp(argv[1],"-get_img2xml") == 0) {
		ret = Get_Img2Xml( argv[2], argv[3], argv[4], argv[5] );
	}
	else if (argc == 6 && strcmp(argv[1],"-check_xml_roi_img") == 0) {
		ret = Check_XML_Roi_Img( argv[2], argv[3], argv[4], atol(argv[5]) );
	}
	else if (argc == 7 && strcmp(argv[1],"-get_bing_roi") == 0) {
		ret = Get_Bing_ROI( argv[2], argv[3], argv[4], atol(argv[5]), atol(argv[6]) );
	}
	else if (argc == 8 && strcmp(argv[1],"-get_xml_bing_roi") == 0) {
		ret = Get_Xml_Bing_ROI( argv[2], argv[3], argv[4], argv[5], atol(argv[6]), atol(argv[7]) );
	}
	else if (argc == 9 && strcmp(argv[1],"-get_xml_bing_roi_for_1_0_0") == 0) {
		ret = Get_Xml_Bing_ROI_for_1_0_0( argv[2], argv[3], argv[4], argv[5], argv[6], atol(argv[7]), atol(argv[8]) );
	}
	else if (argc == 8 && strcmp(argv[1],"-frcnn") == 0) {
		ret = frcnn_test( argv[2], argv[3], argv[4], atol(argv[5]), atol(argv[6]), atol(argv[7]) );
	}
	else
	{
		cout << "usage:\n" << endl;
		cout << "\tDemo_mainboby_frcnn -get_img_num queryList.txt savepath MaxSingleClassNum\n" << endl;
		cout << "\tDemo_mainboby_frcnn -get_big_img queryList.txt savepath ClassName MaxSingleClassNum\n" << endl;
		cout << "\tDemo_mainboby_frcnn -get_vocdata queryList.txt inImgPath inXmlPath imgSavePath xmlSavePath\n" << endl;
		cout << "\tDemo_mainboby_frcnn -ch_xml_name queryList.txt imgPath xmlSavePath imgSavePath\n" << endl;
		cout << "\tDemo_mainboby_frcnn -get_voc2xml queryList.txt imgPath xmlSavePath imgSavePath maxNum\n" << endl;
		cout << "\tDemo_mainboby_frcnn -get_imagenetxml2xml queryList.txt xmlSavePath\n" << endl;
		cout << "\tDemo_mainboby_frcnn -get_img2xml queryList.txt xmlSavePath imgSavePath labelname\n" << endl;
		cout << "\tDemo_mainboby_frcnn -check_xml_roi_img queryList.txt imgPath imgSavePath binVOC\n" << endl;
		cout << "\tDemo_mainboby_frcnn -get_bing_roi queryList.txt keyFilePath layerName binGPU deviceID\n" << endl;
		cout << "\tDemo_mainboby_frcnn -get_xml_bing_roi queryList.txt loadXmlPath keyFilePath layerName binGPU deviceID\n" << endl;
		cout << "\tDemo_mainboby_frcnn -get_xml_bing_roi_for_1_0_0 queryList.txt loadXmlPath svImagePath keyFilePath layerName binGPU deviceID\n" << endl;
		cout << "\tDemo_mainboby_frcnn -frcnn queryList.txt keyFilePath layerName binGPU deviceID saveImg\n" << endl;
		return ret;
	}
	return ret;
}
