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
#include "TErrorCode.h"
#include "plog/Log.h"
#include "API_xml.h"	//read xml
#include "json/json.h"

using namespace cv;
using namespace std;

typedef struct CategoriesInfo
{
	string		_big_class;
	int			_id;
	string		_small_class;
};

typedef struct AnnotationInfo
{
	int			_category_id;
	int			_rect[4];
};

int COCO_XML_DataCH(char *loadImg, char *loadAnnotation, char *svImg, char *svXml, char *CheckImg)
{	
	char tPath[256];
	char loadImgPath[256];
	char szImgPath[256];
	char savePath[256];
	vector<string> vecText;
	int i, j, label, nRet = 0;
	long inputLabel, nCount, nCountFace, labelCount;
	string strImageID,file_name,text;
	API_COMMEN api_commen;
	API_XML api_xml;

	map< string, int > 				mapLabel;
	map< string, int >::iterator 	itLabel;
	vector< pair< string, Vec4i > > 	vecLabelRect;

	/***********************************Init**********************************/
	plog::init(plog::info, "plog.txt"); 

	const static Scalar colors[] =  { 	CV_RGB(0,0,255),	CV_RGB(0,255,255),	CV_RGB(0,255,0),	
										CV_RGB(255,255,0),	CV_RGB(255,0,0),	CV_RGB(255,0,255),
										CV_RGB(0,128,255), 	CV_RGB(255,128,0)} ;

	CvFont font;
 	cvInitFont(&font,CV_FONT_HERSHEY_TRIPLEX,0.35f,0.7f,0,1,CV_AA);

	/********************************Open Query List*****************************/
	FILE *fpListFile = fopen(loadImg,"r");
	if (!fpListFile) 
	{
		cout << "0.can't open " << loadImg << endl;
		return TEC_INVALID_PARAM;
	}

	/***********************************Save File**********************************/
	FILE *fpListFile_categories = fopen("res/categories.txt","wt+");
	if (!fpListFile_categories) 
	{
		cout << "0.can't open " << "res/categories.txt" << endl;
		return TEC_INVALID_PARAM;
	}

	FILE *fpListFile_trainval = fopen("res/trainval.txt","wt+");
	if (!fpListFile_trainval) 
	{
		cout << "0.can't open " << "res/trainval.txt" << endl;
		return TEC_INVALID_PARAM;
	}

	/********************************Open json*****************************/
	Json::Reader reader;
	Json::Value root;

	map< int, CategoriesInfo > 			mapCategoriesInfo;	//<_id,CategoriesInfo>
	map< int, CategoriesInfo >::iterator itCategoriesInfo;
	map< string, int > 			mapImInfo;	//<_file_name,_id>
	map< string, int >::iterator itImInfo;
	map< int, vector< AnnotationInfo > > mapAnnotationInfo;	//<_image_id,AnnotationInfo>
	map< int, vector< AnnotationInfo > >::iterator itAnnotationInfo;
	
	ifstream fpJson;
    fpJson.open(loadAnnotation, ios::binary);

	/*****************************json parse*****************************/
	if(reader.parse(fpJson,root))
	{
		//label
		const Json::Value iters_categories = root["categories"];
		for( i = 0; i<iters_categories.size(); ++i)
		{
			CategoriesInfo categoriesInfo;
			categoriesInfo._big_class = iters_categories[i]["supercategory"].asString();	//12 big class
			categoriesInfo._id = iters_categories[i]["id"].asInt();
			categoriesInfo._small_class = iters_categories[i]["name"].asString();			//91 small class
			
			mapCategoriesInfo[categoriesInfo._id] = categoriesInfo;

			//write categories.txt
			fprintf(fpListFile_categories, "%s.%s\n", categoriesInfo._big_class.c_str(), 
				categoriesInfo._small_class.c_str() );
			//fprintf(fpListFile_categories, "%s		%d		%s\n", categoriesInfo._big_class.c_str(), 
			//	categoriesInfo._id, categoriesInfo._small_class.c_str() );
		}

		//file_name-->id
		const Json::Value iters_images = root["images"];
		for( i = 0; i<iters_images.size(); ++i)
		{
			string _file_name = iters_images[i]["file_name"].asString();
			int	_id = iters_images[i]["id"].asInt();
			
			mapImInfo[_file_name] = _id;
		}

		//id-->label/bbox
		const Json::Value iters_annotations = root["annotations"];
		for( i = 0; i<iters_annotations.size(); ++i)
		{
		    AnnotationInfo annotationInfo;
			int _image_id = iters_annotations[i]["image_id"].asInt();
			annotationInfo._category_id = iters_annotations[i]["category_id"].asInt();
			const Json::Value iters_annotations_rect = iters_annotations[i]["bbox"];
			for( j = 0; j<iters_annotations_rect.size(); ++j)
			{
				annotationInfo._rect[j] = int(atof(iters_annotations_rect[j].asString().c_str())+0.5);
			}
			
			mapAnnotationInfo[_image_id].push_back(annotationInfo);
		}
	}
	else
	{
		printf("json parse error!\n");
		return -1;
	}
	/*********************************close json*************************************/
	fpJson.close();	
	if (fpListFile_categories) {fclose(fpListFile_categories);fpListFile_categories = 0;}	

	/*****************************Process one by one*****************************/
	nCount = 0;
	while(EOF != fscanf(fpListFile, "%s", loadImgPath))
	{
		vecLabelRect.clear();
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
		file_name = strImageID + ".jpg";

		/************************find name*****************************/
		itImInfo = mapImInfo.find(file_name);
		if (itImInfo == mapImInfo.end())
		{	
			cvReleaseImage(&img);img = 0;
			continue;
		}

		/************************find id*****************************/
		itAnnotationInfo = mapAnnotationInfo.find(itImInfo->second);
		if (itAnnotationInfo == mapAnnotationInfo.end())
		{	
			cvReleaseImage(&img);img = 0;
			continue;
		}

		/************************sv xml && img*****************************/
		for( i=0;i<itAnnotationInfo->second.size();i++)
		{
			/************************find label*****************************/
			itCategoriesInfo = mapCategoriesInfo.find(itAnnotationInfo->second[i]._category_id);
			if (itCategoriesInfo == mapCategoriesInfo.end())
			{	
				continue;
			}
			
			/************************change vecLabelRect*****************************/
			sprintf(szImgPath, "%s.%s", itCategoriesInfo->second._big_class.c_str(), 
				itCategoriesInfo->second._small_class.c_str() );
			text = szImgPath;
			
			Vec4i rect( itAnnotationInfo->second[i]._rect[0], itAnnotationInfo->second[i]._rect[1],
					   (itAnnotationInfo->second[i]._rect[2]+itAnnotationInfo->second[i]._rect[0]),
					   (itAnnotationInfo->second[i]._rect[3]+itAnnotationInfo->second[i]._rect[1]));
			vecLabelRect.push_back( make_pair( text, rect ) );

			//write map
			itLabel = mapLabel.find(text);		
			if (itLabel != mapLabel.end()) // find it
				mapLabel[itLabel->first] = itLabel->second+1;		//[In]dic code-words
			else
				mapLabel[text] = 1;
		}

		if (vecLabelRect.size()<1)
		{	
			cvReleaseImage(&img);img = 0;
			continue;
		}

		//cp xml file
		nRet = api_xml.write_xml( strImageID, svXml, vecLabelRect );
		if (nRet!=0)
		{
			cout << "Err to write_xml!" << endl;
			cvReleaseImage(&img);img = 0;
			continue;
		}

		//cp img file
		//sprintf( tPath, "cp %s %s/%s.jpg", loadImgPath, svImg, strImageID.c_str() );
		//system( tPath );

		//write trainval.txt
		fprintf(fpListFile_trainval, "%s\n", strImageID.c_str() );

		/************************check img*****************************/
		for( i=0;i<vecLabelRect.size();i++)
		{
			Scalar color = colors[i%8];
			cvRectangle( img, cvPoint(vecLabelRect[i].second[0], vecLabelRect[i].second[1]),
	                   cvPoint(vecLabelRect[i].second[2], vecLabelRect[i].second[3]), color, 2, 8, 0);

			cvPutText( img, vecLabelRect[i].first.c_str(), cvPoint(vecLabelRect[i].second[0]+1, 
				vecLabelRect[i].second[1]+20), &font, color );
		}
		sprintf( savePath, "%s/%s.jpg", CheckImg, strImageID.c_str() );
		cvSaveImage( savePath, img );

		//count
		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld img...\n",nCount);

		cvReleaseImage(&img);img = 0;
	}
	/*********************************close file*************************************/
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	
	if (fpListFile_trainval) {fclose(fpListFile_trainval);fpListFile_trainval = 0;}	

	/*********************************write label file*********************************/
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
		printf("%s %lld\n", itLabel->first.c_str(), itLabel->second);

		labelCount += itLabel->second;
	}

	fprintf(fpListFile_training, "\n\n", itLabel->first.c_str());
	for(itLabel = mapLabel.begin(); itLabel != mapLabel.end(); itLabel++)
	{
		fprintf(fpListFile_training, "'%s', ", itLabel->first.c_str());
	}
	fprintf(fpListFile_training, "\n\n", itLabel->first.c_str());
	if (fpListFile_training) {fclose(fpListFile_training);fpListFile_training = 0;}	

	/*********************************Print Info*********************************/
	if ( nCount != 0 ) 
	{
		printf("Load img num:%lld, all label count:%lld\n", nCount, labelCount );
		printf("Total Class num:%d\n", mapLabel.size() );
	}
	
	cout<<"Done!! "<<endl;
	
	return 0;

}

#define VOC_LABEL_NUM 20
int Get_Voc2COCO( char *loadXMLPath, char *loadImagePath, char* svXmlPath )
{
	char tPath[256];
	char loadImgPath[256];
	char loadXmlPath[256];
	char szImgPath[256];
	int i, j, svImg, bin_write, bin_write_full, nRet = 0;
	unsigned long long nCount, labelCount;
	int binSameLabel = 1;

	const string vocName[VOC_LABEL_NUM] = {
		"aeroplane","bicycle","bird","boat","bottle",
		"bus","car","cat","chair","cow",
		"diningtable","dog","horse","motorbike","person",
		"pottedplant","sheep","sofa","train","tvmonitor"};
	const string frcnnName[VOC_LABEL_NUM] = {
		"vehicle.airplane","vehicle.bicycle","animal.bird","vehicle.boat","kitchen.bottle",
		"vehicle.bus","vehicle.car","animal.cat","furniture.chair","animal.cow",
		"furniture.dining table","animal.dog","animal.horse","vehicle.motorcycle","person.person",
		"furniture.potted plant","animal.sheep","furniture.chair","vehicle.train","electronic.tv"};

	//Init
	string label;
	string tmpPath,iid,strID, ImageID;
	long end = 0;

	API_COMMEN api_commen;
	API_XML api_xml;

	map< string, string > 				mapVocInName;
	map< string, string >::iterator 	itVocInName;
	vector < pair < string,Vec4i > > 	vecInLabelRect;
	vector < pair < string,Vec4i > > 	vecOutLabelRect;

	//write voc_in_name
	for(i=0;i<VOC_LABEL_NUM;i++)
	{
		mapVocInName[vocName[i]] = frcnnName[i];
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
		vecOutLabelRect.clear();
		for(i=0;i<vecInLabelRect.size();i++)
		{
			/************************find label*****************************/
			itVocInName = mapVocInName.find( vecInLabelRect[i].first );
			if (itVocInName == mapVocInName.end())
			{	
				continue;
			}

			vecOutLabelRect.push_back( std::make_pair( itVocInName->second, vecInLabelRect[i].second ) );
		}
		
		//write_xml
		nRet = api_xml.write_xml( ImageID, svXmlPath, vecOutLabelRect );
		if (nRet!=0)
		{
			cout << "Err to write_xml!" << endl;
			continue;
		}

		//cp img file
		//sprintf( tPath, "cp %s/%s.jpg %s/%s.jpg", imgPath, filename.c_str(), imgSavePath, filename.c_str() );
		//system( tPath );

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

int main(int argc, char* argv[])
{
	int  ret = 0;
	char szKeyFiles[256],szSavePath[256];
	API_COMMEN api_commen;

	if (argc == 7 && strcmp(argv[1],"-coco") == 0) {
		ret = COCO_XML_DataCH( argv[2], argv[3], argv[4], argv[5], argv[6] );
	}
	else if (argc == 5 && strcmp(argv[1],"-voc2coco") == 0) {
		ret = Get_Voc2COCO( argv[2], argv[3], argv[4] );
	}
	else
	{
		cout << "usage:\n" << endl;
		cout << "\tDemo_json -coco loadImg loadAnnotation svImg svXml CheckImg\n" << endl;
		cout << "\tDemo_json -voc2coco loadXMLPath loadImagePath svXml\n" << endl;
		return ret;
	}
	return ret;
}




