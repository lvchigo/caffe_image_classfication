#pragma once
#include <vector>
#include <iostream>
#include <string>
#include <stdio.h>

using namespace std;

class API_WGET_INIM
{

/***********************************public***********************************/
public:

	/// construct function 
    API_WGET_INIM();
    
	/// distruct function
	~API_WGET_INIM(void);

	/***********************************wget_image_file*************************************/
	int wget_image_file(const std::string &url, const std::string &image_file);
	int wget_image_file_sys(const std::string &url, const std::string &image_file);

	/***********************************id2url**********************************/
	string id2url(const std::string& id);

/***********************************private***********************************/
private:

	

};

