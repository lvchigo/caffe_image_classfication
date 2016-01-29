#pragma once
//#include <cuda_runtime.h>
//#include <google/protobuf/text_format.h>
#include <queue>  // for std::priority_queue
#include <utility>  // for pair

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <dirent.h>
#include <unistd.h>

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>
#include <opencv/ml.h>
#include <opencv/cvaux.h>

#include <time.h>
#include <sys/mman.h> /* for mmap and munmap */
#include <sys/types.h> /* for open */
#include <sys/stat.h> /* for open */
#include <fcntl.h>     /* for open */
#include <pthread.h>

#include <vector>
#include <list>
#include <map>
#include <algorithm>

#include "TErrorCode.h"

#include "tinyxml2.hpp"	//read xml
#include "API_xml.h"	//read xml

using namespace cv;
using namespace std;
using namespace tinyxml2;

/***********************************Init*************************************/
/// construct function 
API_XML::API_XML()
{
}

/// destruct function 
API_XML::~API_XML(void)
{
}

int API_XML::load_xml( string loadXml, string &ImageID, vector< pair< string, Vec4i > > &vecLabelRect )
{
	char tPath[256];
	char szImgPath[256];
	int i, j, svImg, nRet = 0;
	unsigned long long nCount, labelCount;

	/********************************Init*****************************/
	string strLabel;
	string tmpPath,iid;
	long end = 0;
	vecLabelRect.clear();

	/********************************reading the xml files*****************************/
	XMLDocument doc;
	XMLError eResult = doc.LoadFile(const_cast<char*>(loadXml.c_str()));
	if (eResult != XML_SUCCESS) {
		cout<<"Error opening the xml files! "<<loadXml<<endl;
		return TEC_INVALID_PARAM;
	}

	XMLNode* pRoot = doc.FirstChildElement("annotation");
	if (pRoot == nullptr) {
		cout<<"No annotation available!"<<endl;
		return TEC_INVALID_PARAM;
	}
	
	XMLElement* pNameElement = pRoot->FirstChildElement("filename");
	if (pNameElement == nullptr) {
		cout<<"No filename here!"<<endl;
		return TEC_INVALID_PARAM;
	}

	//read name 
	const char * Name = pNameElement->GetText();
	//printf("Name:%s\n",Name);

	ImageID = Name;
	//printf("before ImageID:%s\n",ImageID.c_str());
	std::size_t found = ImageID.find(".jpg");
	if ( (found!=std::string::npos) && (found>0) )
	{
		end = 0;
		tmpPath = ImageID;
		end = tmpPath.find_last_of(".");
		if (end>0)
		{
			iid = tmpPath.substr(0,end);
			ImageID = iid;
		}
	}
	//printf("after ImageID:%s,found:%d\n",ImageID.c_str(),found);
	//printf("change Name:%s,end:%d\n",iid.c_str(),end);

	//read name
	XMLElement* pObjectElement = pRoot->FirstChildElement("object");
	if (pObjectElement == nullptr) {
		cout<<"No filename here!"<<endl;
		return TEC_INVALID_PARAM;
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
		strLabel = string(label);
		//printf("%s ",label);

		//change label
		if ( strLabel == "Copy of goods.goods.cosmetics")
		{ 
			 printf("filename:%s %s\n", ImageID.c_str(), strLabel.c_str() );
			 strLabel = "goods.goods.cosmetics";
		}
		else if ( ( strLabel == "Copy of Copy of goods.shoe.shoe") || 
			 ( strLabel == "Copy of goods.shoe.shoe") )
		{
			 printf("filename:%s %s\n", ImageID.c_str(), strLabel.c_str() );
			 strLabel = "goods.shoe.shoe";
		}
		else if ( strLabel == "goods.train.train" )
		{
			printf("filename:%s %s\n", ImageID.c_str(), strLabel.c_str() );
			strLabel = "goods.car.train";
		}

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
		
		vecLabelRect.push_back( std::make_pair( strLabel, rect ) );

		pObjectElement = pObjectElement->NextSiblingElement("object");  
	} 
	//printf("\n");
	
	return 0;
}

int API_XML::write_xml( string ImageID, string xmlSavePath, vector< pair< string, Vec4i > > vecLabelRect )
{
	if (vecLabelRect.size()<1) 
	{	
		cout<<"Err to write_xml"<<endl;
		return TEC_INVALID_PARAM;
	}	
	
	char tPath[256];
	int i,x1,y1,x2,y2;
	
	//write xml
	XMLDocument *pDoc = new XMLDocument();  
	if (NULL == pDoc) { return TEC_INVALID_PARAM;  }  
	XMLElement *pRootEle = pDoc->NewElement("annotation");  //FDTinyXML2_ROOT_NAME
	if (NULL == pRootEle) { return TEC_INVALID_PARAM; }  
	pDoc->LinkEndChild(pRootEle); 
	
	XMLElement *filenameXMLElement = pDoc->NewElement("filename");  
	sprintf(tPath, "%s", ImageID.c_str() );
	XMLText *filenameXMLText = pDoc->NewText(tPath);  
	filenameXMLElement->LinkEndChild(filenameXMLText);  
	pRootEle->LinkEndChild(filenameXMLElement);  

	for(i=0;i<vecLabelRect.size();i++)
	{
		x1 = (vecLabelRect[i].second[0]<1)?1:vecLabelRect[i].second[0];
		y1 = (vecLabelRect[i].second[1]<1)?1:vecLabelRect[i].second[1];
		x2 = (vecLabelRect[i].second[2]<1)?1:vecLabelRect[i].second[2];
		y2 = (vecLabelRect[i].second[3]<1)?1:vecLabelRect[i].second[3];

		XMLElement *objXMLElement = pDoc->NewElement("object");  
		pRootEle->LinkEndChild(objXMLElement);  

		XMLElement *nameXMLElement = pDoc->NewElement("name");  
		sprintf( tPath, "%s", vecLabelRect[i].first.c_str() );
		XMLText *nameXMLText = pDoc->NewText(tPath);  
		nameXMLElement->LinkEndChild(nameXMLText);  
		objXMLElement->LinkEndChild(nameXMLElement);  
	
		XMLElement *bndboxXMLElement = pDoc->NewElement("bndbox");  
		objXMLElement->LinkEndChild(bndboxXMLElement);  

		XMLElement *xminXMLElement = pDoc->NewElement("xmin");  
		sprintf(tPath, "%d", x1 );
		XMLText *xminXMLText = pDoc->NewText(tPath);  
		xminXMLElement->LinkEndChild(xminXMLText);  
		bndboxXMLElement->LinkEndChild(xminXMLElement);  

		XMLElement *yminXMLElement = pDoc->NewElement("ymin");  
		sprintf(tPath, "%d", y1 );
		XMLText *yminXMLText = pDoc->NewText(tPath);  
		yminXMLElement->LinkEndChild(yminXMLText);  
		bndboxXMLElement->LinkEndChild(yminXMLElement);  

		XMLElement *xmaxXMLElement = pDoc->NewElement("xmax");  
		sprintf(tPath, "%d", x2 );
		XMLText *xmaxXMLText = pDoc->NewText(tPath);  
		xmaxXMLElement->LinkEndChild(xmaxXMLText);  
		bndboxXMLElement->LinkEndChild(xmaxXMLElement);  

		XMLElement *ymaxXMLElement = pDoc->NewElement("ymax");  
		sprintf(tPath, "%d", y2 );
		XMLText *ymaxXMLText = pDoc->NewText(tPath);  
		ymaxXMLElement->LinkEndChild(ymaxXMLText);  
		bndboxXMLElement->LinkEndChild(ymaxXMLElement);  
	}

	//write xml
	sprintf(tPath, "%s/%s.xml", xmlSavePath.c_str(), ImageID.c_str() );
	pDoc->SaveFile( tPath );  
	if (pDoc) {  delete pDoc;  }

	return 0;
}

int API_XML::Get_Xml_Hypothese( 
	string ImageID, 
	int width, 
	int height, 
	vector< pair< string, Vec4i > > vecLabelRect, 
	vector< pair< string, Vec4i > > &vecOutRect,
	int binHypothese)
{
	if( vecLabelRect.size() < 1 ) 
	{	
		cout<<"[input err]"<<endl;
		return TEC_INVALID_PARAM;
	}

	/***********************************init*************************************/
	int i,j,rWidth, rHeight, biasWidth, biasHeight, roiSizeNum;
	float ratio_wh = 0;
	string label;

	float roiSizeMultiple, roiSize = 0;
	if ( binHypothese == 1 )
	{
		roiSizeMultiple = 0.075;
		roiSizeNum = 2;
	}
	else
	{
		roiSizeMultiple = 0.025;
		roiSizeNum = 12;
	}

	/************************get_xml_Hypothese*****************************/
	vecOutRect.clear();
	for (i=0;i<vecLabelRect.size();i++)
	{
		label = vecLabelRect[i].first;
		rWidth = vecLabelRect[i].second[2]-vecLabelRect[i].second[0];
		rHeight = vecLabelRect[i].second[3]-vecLabelRect[i].second[1];

		//remove err roi
		if ( (vecLabelRect[i].second[0]<0) || (vecLabelRect[i].second[1]<0) || 
			 (vecLabelRect[i].second[2]>width) || (vecLabelRect[i].second[3]>height) || 
			 (rWidth<16) || (rHeight<16) ||
			 (rWidth<0.2*rHeight) || (rHeight<0.2*rWidth) )
			continue;

		roiSize = 0;
		for (j=0;j<roiSizeNum;j++)
		{
			roiSize += roiSizeMultiple;
			biasWidth = int(rWidth*roiSize+0.5);
			biasHeight = int(rHeight*roiSize+0.5);

			//for normal
			if ( binHypothese == 0 )
			{
				//ADD-SIZE
				if (vecLabelRect[i].second[0]-biasWidth>1)
				{
					Vec4i Rect;
					Rect[0] = vecLabelRect[i].second[0]-biasWidth;
					Rect[1] = vecLabelRect[i].second[1];
					Rect[2] = vecLabelRect[i].second[2];
					Rect[3] = vecLabelRect[i].second[3];
					vecOutRect.push_back( std::make_pair( label, Rect ) );
				}

				if (vecLabelRect[i].second[1]-biasHeight>1)
				{
					Vec4i Rect;
					Rect[0] = vecLabelRect[i].second[0];
					Rect[1] = vecLabelRect[i].second[1]-biasHeight;
					Rect[2] = vecLabelRect[i].second[2];
					Rect[3] = vecLabelRect[i].second[3];
					vecOutRect.push_back( std::make_pair( label, Rect ) );
				}
				
				if (vecLabelRect[i].second[2]+biasWidth<(width-1))
				{
					Vec4i Rect;
					Rect[0] = vecLabelRect[i].second[0];
					Rect[1] = vecLabelRect[i].second[1];
					Rect[2] = vecLabelRect[i].second[2]+biasWidth;
					Rect[3] = vecLabelRect[i].second[3];
					vecOutRect.push_back( std::make_pair( label, Rect ) );
				}

				if (vecLabelRect[i].second[3]+biasHeight<(height-1))
				{
					Vec4i Rect;
					Rect[0] = vecLabelRect[i].second[0];
					Rect[1] = vecLabelRect[i].second[1];
					Rect[2] = vecLabelRect[i].second[2];
					Rect[3] = vecLabelRect[i].second[3]+biasHeight;
					vecOutRect.push_back( std::make_pair( label, Rect ) );
				}
			}
			else if ( binHypothese == 1 )	//only for mainbody v1.0.0
			{
				//ADD-SIZE
				if ( (vecLabelRect[i].second[0]-biasWidth>1) && (vecLabelRect[i].second[1]-biasHeight>1) && 
					 (vecLabelRect[i].second[2]+biasWidth<(width-1)) && vecLabelRect[i].second[3]+biasHeight<(height-1) )
				{
					Vec4i Rect;
					Rect[0] = vecLabelRect[i].second[0]-biasWidth;
					Rect[1] = vecLabelRect[i].second[1]-biasHeight;
					Rect[2] = vecLabelRect[i].second[2]+biasWidth;
					Rect[3] = vecLabelRect[i].second[3]+biasHeight;
					vecOutRect.push_back( std::make_pair( label, Rect ) );
				}
			}
		}

		//add self
		if ( vecOutRect.size()<1 )
			vecOutRect.push_back( std::make_pair( label, vecLabelRect[i].second ) );
	}

	return TOK;
}

