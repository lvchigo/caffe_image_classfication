#pragma once
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

