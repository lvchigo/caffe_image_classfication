/*
 * =====================================================================================
 *
 *       filename:  API_xml.h
 *
 *    description:  xml interface
 *
 *        version:  1.0
 *        created:  2016-03-08
 *       revision:  none
 *       compiler:  g++
 *
 *         author:  xiaogao
 *        company:  in66.com
 *
 *      copyright:  2016 itugo Inc. All Rights Reserved.
 *      
 * =====================================================================================
 */


#ifndef _API_XML_H_
#define _API_XML_H_

#include <string>
#include <vector>
#include <opencv/cv.h>

using namespace cv;
using namespace std;

class API_XML
{

/***********************************Common***********************************/
typedef unsigned char uchar;
typedef unsigned long long UInt64;

/***********************************public***********************************/
public:

	/// construct function 
    API_XML();
    
	/// distruct function
	~API_XML(void);

	/***********************************xml*************************************/
	int load_xml( string loadXml, string &ImageID, vector< pair< string, Vec4i > > &vecLabelRect );
	
	int write_xml( string ImageID, string xmlSavePath, vector< pair< string, Vec4i > > vecLabelRect );
	
	int Get_Xml_Hypothese( 
		string ImageID, int width, int height, 
		vector< pair< string, Vec4i > > vecLabelRect, 
		vector< pair< string, Vec4i > > &vecOutRect,
		int binHypothese);

/***********************************private***********************************/
private:
	

};

#endif

