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

#include "API_commen.h"
#include "API_caffe.h"
#include "API_mainboby.h"
#include "TErrorCode.h"
#include "plog/Log.h"

#include "tinyxml2.hpp"	//read xml

using namespace cv;
using namespace std;
using namespace tinyxml2;


static bool PredictResSortComp(
	const pair< pair< int, Vec4i >, float > elem1, 
	const pair< pair< int, Vec4i >, float > elem2)
{
	return (elem1.second > elem2.second);
}

static bool IOUSortComp(
	const pair< int, float > elem1, 
	const pair< int, float > elem2)
{
	return (elem1.second > elem2.second);
}


/***********************************Init*************************************/
/// construct function 
API_MAINBOBY::API_MAINBOBY()
{
}

/// destruct function 
API_MAINBOBY::~API_MAINBOBY(void)
{
}

/***********************************Init*************************************/
int API_MAINBOBY::Init( 
	const char* 	KeyFilePath,						//[In]:KeyFilePath
	const char* 	layerName,							//[In]:layerName:"fc7"
	const int		binGPU, 							//[In]:USE GPU(1) or not(0)
	const int		deviceID )	 						//[In]:GPU ID
{
	char tPath[1024] = {0};
	char tPath2[1024] = {0};
	char tPath3[1024] = {0};

	/***********************************Init Bing*************************************/
	objNess = new Objectness(2, 8, 2);
	sprintf(tPath, "%s/mainbody_frcnn/BING_voc2007/ObjNessB2W8MAXBGR",KeyFilePath);
	objNess->loadTrainedModelOnly(tPath);

	/***********************************Init FastRCNNCls**********************************/
	fastRCNN = new FastRCNNCls;
	//sprintf(tPath, "%s/mainbody_frcnn/frcnn_model/CaffeNet_in_56class_20151225/test_caffenet.prototxt",KeyFilePath);	//voc2007+2012+in20151211
	//sprintf(tPath2, "%s/mainbody_frcnn/frcnn_model/CaffeNet_in_56class_20151225/caffenet_fast_rcnn_iter_40000.caffemodel",KeyFilePath);	//voc2007+2012+in20151211
	sprintf(tPath, "%s/mainbody_frcnn/frcnn_model/CaffeNet_in_90class_20151226/test_caffenet.prototxt",KeyFilePath);	//voc2007+2012+in20151211+svd
	sprintf(tPath2, "%s/mainbody_frcnn/frcnn_model/CaffeNet_in_90class_20151226/caffenet_fast_rcnn_iter_40000.caffemodel",KeyFilePath);	//voc2007+2012+in20151211+svd
	//sprintf(tPath, "%s/mainbody_frcnn/frcnn_model/CaffeNet/test.prototxt",KeyFilePath);	//voc2007+2012
	//sprintf(tPath2, "%s/mainbody_frcnn/frcnn_model/CaffeNet/caffenet_fast_rcnn_iter_40000.caffemodel",KeyFilePath);	//voc2007+2012
#ifdef USE_MODEL_VGG16
	sprintf(tPath, "%s/mainbody_frcnn/frcnn_model/CaffeNet_in_90class_20151226/test_vgg16_svd.prototxt",KeyFilePath);
	sprintf(tPath2, "%s/mainbody_frcnn/frcnn_model/CaffeNet_in_90class_20151226/vgg16_fast_rcnn_iter_40000_svd_fc6_1024_fc7_256.caffemodel",KeyFilePath);
#endif
	fastRCNN->set_model( tPath, tPath2, binGPU, deviceID );

	tgt_cls.clear();
	for (i=0; i<numObjClass; i++) {
		tgt_cls.push_back(i+1);
	}
	
	/***********************************Load dic_voc20Class File**********************************/
	dic_voc20Class.clear();
	//sprintf(tPath, "%s/mainbody_frcnn/frcnn_model/CaffeNet_in_56class_20151225/Dict_FRCNN_56label.txt",KeyFilePath);
	sprintf(tPath, "%s/mainbody_frcnn/frcnn_model/CaffeNet_in_90class_20151226/Dict_FRCNN_90label.txt",KeyFilePath);
	//sprintf(tPath, "%s/mainbody_frcnn/frcnn_model/CaffeNet/dic_voc20Class",KeyFilePath);
	printf("load dic_voc20Class:%s\n",tPath);
	api_commen.loadWordDict(tPath,dic_voc20Class);
	printf( "dict:size-%d,tag:", int(dic_voc20Class.size()) );
	for ( i=0;i<dic_voc20Class.size();i++ )
	{
		printf( "%d-%s ",i,dic_voc20Class[i].c_str() );
	}
	printf( "\n" );

	return nRet;
}

IplImage* API_MAINBOBY::ResizeImg( IplImage *img, int MaxLen )
{
	int rWidth, rHeight;

	IplImage *imgResize;
	if( ( img->width>MaxLen ) || ( img->height>MaxLen ) )
	{
		nRet = api_commen.GetReWH( img->width, img->height, MaxLen, rWidth, rHeight );	
		if (nRet != 0)
		{
			LOOGE<<"[GetReWH]";
			return NULL;
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

	return imgResize;
}

int API_MAINBOBY::GetTwoRect_Intersection_Union( vector<Vec4i> inBox, double &Intersection, double &Union )
{
	if ( inBox.size()<1 )
	{	
		LOOGE<<"[GetTwoRect_Union_Intersection init err]";
		return TEC_INVALID_PARAM;
	}
	
	double x[5]={0};
	double y[5]={0};
	int xy[4][4] = {0};
	int m,i,j,i1,i2,j1,j2,k,kk,n=inBox.size();

    k=0;
    memset(xy,0,sizeof(xy));
    for(i=0;i<n;i++)
    {
        x[k]=inBox[i][0];
        y[k]=inBox[i][1];
        k++;
        x[k]=inBox[i][2];
        y[k]=inBox[i][3];
        k++;
    }
    sort(x,x+2*n);
    sort(y,y+2*n);
	kk = 0;
    for(k=0;k<n;k++)
    {
        for(i1=0;i1<2*n;i1++)
        {
            if(x[i1]==inBox[k][0])
                break;
        }
        for(i2=0;i2<2*n;i2++)
        {
            if(x[i2]==inBox[k][2])
                break;
        }
        for(j1=0;j1<2*n;j1++)
        {
            if(y[j1]==inBox[k][1])
                break;
        }
        for(j2=0;j2<2*n;j2++)
        {
            if(y[j2]==inBox[k][3])
                break;
        }
        for(i=i1;i<i2;i++)
        {
            for(j=j1;j<j2;j++)
            {
                xy[i][j] |= 1<<k;
            }
        }
		kk |= 1<<k;
    }

	Union = 0.0;
	Intersection = 0.0;
    for(i=0;i<2*n;i++)
    {
        for(j=0;j<2*n;j++)
        {
			Union += ((xy[i][j] != 0 ? 1:0)*(x[i+1]-x[i])*(y[j+1]-y[j]));
            Intersection += ((xy[i][j] == kk ? 1:0)*(x[i+1]-x[i])*(y[j+1]-y[j])); 
        }
    }
	
    //printf("Rect-1:%d-%d-%d-%d\n",inBox[0][0],inBox[0][1],inBox[0][2],inBox[0][3]);
	//printf("Rect-2:%d-%d-%d-%d\n",inBox[1][0],inBox[1][1],inBox[1][2],inBox[1][3]);
    //printf("Union: %.2f\n",Union);
	//printf("Intersection: %.2f\n",Intersection);
    //printf("\n");

    return 0;
}

//vectorIouCover: vector< pair<Vec4i, pair< iou, cover > > > 
int API_MAINBOBY::Get_iou_cover( 
	vector< pair< float, Vec4i > > 				bingBox, 
	vector< pair< string, Vec4i > > 			gtBox, 
	vector< pair<Vec4i, pair< double, int > > > &vectorIouCover )
{
	if ( ( bingBox.size()<1 ) || ( gtBox.size()<1 ) )
	{	
		LOOGE<<"[GetTwoRect_Union_Intersection init err]";
		return TEC_INVALID_PARAM;
	}

	double Intersection, Union, tmpIOU, maxIOU;
	int label, coverGT;
	vector<Vec4i> twoBox;
	vector< pair< int, float > > vectorIOU;

	vectorIouCover.clear();
	for(i=0;i<bingBox.size();i++)
	{
		vectorIOU.clear();
		for(j=0;j<gtBox.size();j++)
		{
			twoBox.clear();
			twoBox.push_back(bingBox[i].second); 
			twoBox.push_back(gtBox[j].second); 

			Intersection = 0;
			Union = 0;
			nRet = GetTwoRect_Intersection_Union( twoBox, Intersection, Union );
			if ( ( nRet!= 0 ) || ( Union == 0 ) || ( Intersection > Union ) )
			{
				tmpIOU = 0;
				LOOGE<<"[Get Err Intersection && Union ]";
				continue;
			}
			
			tmpIOU = Intersection*1.0/Union ;
			vectorIOU.push_back( std::make_pair( j, tmpIOU ) );
		}

		//sort
		sort(vectorIOU.begin(), vectorIOU.end(),IOUSortComp);

		//count cover ground truth
		coverGT = 0;
		label = vectorIOU[0].first;
		maxIOU = vectorIOU[0].second;
		if ( (bingBox[i].second[0]<=gtBox[label].second[0]) && (bingBox[i].second[1]<=gtBox[label].second[1]) &&
			 (bingBox[i].second[2]>=gtBox[label].second[2]) && (bingBox[i].second[3]>=gtBox[label].second[3]) )
		{
			coverGT = 1;
			//printf("maxIOU:%.4f,coverGT:bing-%d-%d-%d-%d,gt-%d-%d-%d-%d\n",maxIOU,
			//	bingBox[i].second[0],bingBox[i].second[1],bingBox[i].second[2],bingBox[i].second[3],
			//	gtBox[label].second[0],gtBox[label].second[1],gtBox[label].second[2],gtBox[label].second[3]);
		}

		if ( (maxIOU>=0.7) && (coverGT==1) || (maxIOU<=0.3) && (coverGT!=1) )
			vectorIouCover.push_back( std::make_pair( bingBox[i].second, std::make_pair( maxIOU, coverGT ) ) );
	}
	
	return 0;
}

int API_MAINBOBY::load_xml( string loadXml, string &ImageID, vector< pair< string, Vec4i > > &vecLabelRect )
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


int API_MAINBOBY::write_xml( string ImageID, string xmlSavePath, vector< pair< string, Vec4i > > vecLabelRect )
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

int API_MAINBOBY::Get_Xml_Hypothese( 
	string ImageID, 
	int width, 
	int height, 
	vector< pair< string, Vec4i > > vecLabelRect, 
	vector< pair< string, Vec4i > > &vecOutRect,
	int binHypothese)
{
	if( vecLabelRect.size() < 1 ) 
	{	
		LOOGE<<"[input err]";
		return TEC_INVALID_PARAM;
	}

	/***********************************init*************************************/
	int rWidth, rHeight, biasWidth, biasHeight, roiSizeNum;
	float ratio_wh = 0;
	string label;

	float roiSizeMultiple, roiSize = 0;
	if ( binHypothese == 1 )
	{
		roiSizeMultiple = 0.05;
		roiSizeNum = 6;
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
			 (rWidth<TImgSize) || (rHeight<TImgSize) ||
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
		vecOutRect.push_back( std::make_pair( label, vecLabelRect[i].second ) );
	}

	return TOK;
}

//BinTraining:2-NO Remove Rectfor Training;1-Remove small Rect for Training;0-Remove small Rect for Test
int API_MAINBOBY::Get_Bing_Hypothese( IplImage *img, vector< pair<float, Vec4i> > &outBox, int BinTraining )
{
	if( !img || (img->width<128) || (img->height<128) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U ) 
	{	
		LOOGE<<"[input err]";
		return TEC_INVALID_PARAM;
	}

	/***********************************init*************************************/
	int width, height, rWidth, rHeight, T_WH, topN = 1000;
	float ratio_wh = 0;

	/************************getObjBndBoxes*****************************/
	ValStructVec<float, Vec4i> inBox;
	Mat matImg(img);
	inBox.clear();
	inBox.reserve(10000);
	
	run.start();
	objNess->getObjBndBoxes(matImg, inBox, 130);
	run.end();
	LOOGI<<"[getObjBndBoxes] time:"<<run.time();

	if (inBox.size()<1)
	{	
		LOOGE<<"[getObjBndBoxes err]";
		return TEC_INVALID_PARAM;
	}

	width = img->width;
	height = img->height;
	if ( BinTraining != 0 )
	{
		T_WH = (width>height)?((int)(height*0.1)):((int)(width*0.1));
		T_WH = (T_WH>TRectSize)?T_WH:TRectSize;
	}
	else if ( BinTraining == 0 )
	{
		T_WH = (width>height)?((int)(height*0.2)):((int)(width*0.2));
		T_WH = (T_WH>TImgSize)?T_WH:TImgSize;
	}
		
	/***********************************remove small roi*************************************/
	outBox.clear();
	for(i=0;i<inBox.size();i++)
	{
		inBox[i][0] = (int)(inBox[i][0]>0.0)?inBox[i][0]:0;
		inBox[i][1] = (int)(inBox[i][1]>0.0)?inBox[i][1]:0;
		inBox[i][2] = (int)(inBox[i][2]>0.0)?inBox[i][2]:0;
		inBox[i][3] = (int)(inBox[i][3]>0.0)?inBox[i][3]:0;	

		inBox[i][0] = (int)(inBox[i][0]<width)?inBox[i][0]:(width-1);
		inBox[i][1] = (int)(inBox[i][1]<height)?inBox[i][1]:(height-1);
		inBox[i][2] = (int)(inBox[i][2]<width)?inBox[i][2]:(width-1);
		inBox[i][3] = (int)(inBox[i][3]<height)?inBox[i][3]:(height-1);
		
		rWidth = inBox[i][2]-inBox[i][0];
		rHeight = inBox[i][3]-inBox[i][1];	

		//remove err roi
		if ( (rWidth<1) || (rHeight<1) )
			continue;

		if ( (inBox[i][0]<0) || (inBox[i][1]<0) ||(inBox[i][2]<0) ||(inBox[i][3]<0) )
		{
			printf("out of rect :%d-%d-%d-%d\n",inBox[i][0],inBox[i][1],inBox[i][2],inBox[i][3]);
		}

		if ( ( BinTraining == 0 ) || ( BinTraining == 1 ) )
		{
			ratio_wh = rWidth*1.0/rHeight;
			
			//remove small roi
			if ( (rWidth<T_WH) || (rHeight<T_WH) || (ratio_wh<0.2) || (ratio_wh>5) )
				continue;
		}

		outBox.push_back( std::make_pair(inBox(i), inBox[i]) ); 
		if ( outBox.size() == topN )
			break;
	}

	/***********************************Release**********************************/
	//ValStructVec<float, Vec4i>().swap(inBox);
	if (outBox.size()<1)
	{	
		LOOGE<<"[remove small roi err]";
		return TEC_INVALID_PARAM;
	}
	
	return TOK;
}

int API_MAINBOBY::Predict(
	IplImage										*image, 			//[In]:image
	UInt64											ImageID,			//[In]:ImageID
	const char* 									layerName,
	vector< pair< pair< string, Vec4i >, float > >	&Res)			//[In]:Layer Name by Extract
{	
	if(!image || (image->width<16) || (image->height<16) || image->nChannels != 3 || image->depth != IPL_DEPTH_8U ) 
	{	
		LOOGE<<"input err!!";
		return TEC_INVALID_PARAM;
	}

	int label, T_WH, width, height, rWidth, rHeight, x1, x2, y1, y2, topN = 5;
	nRet = 0;
	float ratio_wh = 0;
	vector< pair< pair< int, Vec4i >, float > > 	vecLabel;
	vector< pair< pair< string, Vec4i >, float > >	MergeRes;
	
	/************************Get_Bing_Hypothese*****************************/
	vector< pair<float, Vec4i> > HypotheseBox;
	HypotheseBox.clear();
	nRet = Get_Bing_Hypothese( image, HypotheseBox );
	if ( (nRet!=0) || (HypotheseBox.size()<1) )
	{
		LOOGE<<"[Get_Bing_Hypothese Err!!ImageID:]"<<ImageID;
		return TEC_INVALID_PARAM;
	}
	//printf("HypotheseBox.size():%d\n",HypotheseBox.size());

	fastRCNN->set_boxes( HypotheseBox );
	
	/************************fastRCNN->detect*****************************/
	Mat imgMat(image);
	vector<PropBox> randPboxes = fastRCNN->detect(imgMat, tgt_cls);
	if ( randPboxes.size()<1 )
	{
		LOOGE<<"[randPboxes.size()<1!!ImageID:]"<<ImageID<<"HypotheseBox.size():"<<HypotheseBox.size();	
		return TEC_INVALID_PARAM;
	}
	//printf("randPboxes.size():%d\n",randPboxes.size());

	/************************data ch*****************************/
	width = image->width;
	height = image->height;
	T_WH = (width>height)?((int)(height*0.1)):((int)(width*0.1));
	T_WH = (T_WH>TRectSize)?T_WH:TRectSize;
		
	vecLabel.clear();
	//printf("ImageID_%lld:",ImageID);
	for ( i=0;i<randPboxes.size();i++ )
	{
		label = randPboxes[i].cls_id-1;
		if ( randPboxes[i].confidence < 0.2 )
		{
			//printf("out of size label:%d\n",label);
			continue;
		}

		Vec4i rect;
		if ( randPboxes[i].x1 < 0 )
			rect[0] = 0;
		else if ( randPboxes[i].x1 < (width-1) )
			rect[0] = randPboxes[i].x1;
		else
			rect[0] = (width-1);

		if ( randPboxes[i].y1 < 0 )
			rect[1] = 0;
		else if ( randPboxes[i].y1 < (height-1) )
			rect[1] = randPboxes[i].y1;
		else
			rect[1] = (height-1);

		if ( randPboxes[i].x2 < 0 )
			rect[2] = 0;
		else if ( randPboxes[i].x2 < (width-1) )
			rect[2] = randPboxes[i].x2;
		else
			rect[2] = (width-1);

		if ( randPboxes[i].y2 < 0 )
			rect[3] = 0;
		else if ( randPboxes[i].y2 < (height-1) )
			rect[3] = randPboxes[i].y2;
		else
			rect[3] = (height-1);

		//remove small roi
		rWidth = rect[2]-rect[0];
		rHeight = rect[3]-rect[1];
		ratio_wh = rWidth*1.0/rHeight;
		if ( (rWidth<T_WH) || (rHeight<T_WH) || (ratio_wh<0.125) || (ratio_wh>8) )
			continue;

		vecLabel.push_back( std::make_pair( std::make_pair( label, rect ), randPboxes[i].confidence ) );
		//printf("%d_%d_%d_%d_%d_%.4f ",label, rect[0], rect[1], rect[2], rect[3], randPboxes[i].confidence );
	}
	//printf("\n");
	if ( vecLabel.size()<1 )
	{
		LOOGE<<"[vecLabel.size()<1!!ImageID:]"<<ImageID;
		return TEC_INVALID_PARAM;
	}
	//printf("vecLabel.size():%d\n",vecLabel.size());

	//sort AllRes
	sort(vecLabel.begin(), vecLabel.end(),PredictResSortComp);

	/************************MergeVOC20classLabel*****************************/
	MergeRes.clear();
 	nRet = MergeVOC20classLabel( vecLabel, MergeRes );
	if ( (nRet!=0) || (MergeRes.size()<1) )
	{
		LOOGE<<"[MergeVOC20classLabel Err!!ImageID:]"<<ImageID;
		return TEC_INVALID_PARAM;
	}

	/************************Merge Res*****************************/
	Res.clear();
/*	nRet = Merge_Predict( MergeRes, Res);
	if (nRet != 0) 
	{
		LOOGE<<"Fail to Merge_Predict!!";
		return TEC_BAD_STATE;
	}*/
	topN = (MergeRes.size()>topN)?topN:MergeRes.size();
	Res.assign( MergeRes.begin(), MergeRes.begin()+topN );
	//printf("ImageID:%lld,HypotheseBox:%d,randPboxes:%d,vecLabel:%d,Res.size():%d\n",
	//	ImageID, HypotheseBox.size(), randPboxes.size(), vecLabel.size(), Res.size());

	return nRet;
}

int API_MAINBOBY::MergeVOC20classLabel(
	vector< pair< pair< int, Vec4i >, float > >			inImgLabel, 		//[In]:ImgDetail from GetLabel
	vector< pair< pair< string, Vec4i >, float > > 		&LabelInfo)			//[Out]:LabelInfo
{
	if ( inImgLabel.size() < 1 ) 
	{ 
		LOOGE<<"MergeLabel[err]:inImgLabel.size()<1!!";
		return TEC_INVALID_PARAM;
	}
	
	int label;
	float score = 0.0;
	LabelInfo.clear();

	for ( i=0;i<inImgLabel.size();i++ )
	{
		label = inImgLabel[i].first.first;
		Vec4i rect = inImgLabel[i].first.second;
		score = inImgLabel[i].second;		

		if (label < dic_voc20Class.size() )
		{
			LabelInfo.push_back( std::make_pair( std::make_pair( dic_voc20Class[label], rect ), score ) );
		}
		else
		{ 
			LOOGE<<"MergeVOC20classLabel[err]!!";
			return TEC_INVALID_PARAM;
		}
	}
	
	return TOK;
}

/***********************************Release**********************************/
void API_MAINBOBY::Release()
{
	/***********************************net Model**********************************/
	//api_caffe.Release();
	delete objNess;
	delete fastRCNN;
}





