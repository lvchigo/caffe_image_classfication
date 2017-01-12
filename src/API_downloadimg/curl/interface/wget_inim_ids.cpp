#pragma once
#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#include "string_operator.h"
#include "json/json.h"
#include "downloader.h"
#include "wget_inim_ids.h"

using namespace std;

/***********************************Init*************************************/
/// construct function 
API_WGET_INIM::API_WGET_INIM()
{
}

/// destruct function 
API_WGET_INIM::~API_WGET_INIM(void)
{
}

/***********************************wget_image_file*************************************/
int API_WGET_INIM::wget_image_file(const std::string &url, const std::string &image_file)
{
  std::string cmd = "wget -q -T 1 -t 1 " + url + " -O " + image_file;
  std::cout<<cmd<<std::endl;
  FILE* fptr = popen(cmd.c_str(), "r");
  pclose(fptr);
  return 1;
}

int API_WGET_INIM::wget_image_file_sys(const std::string &url, const std::string &image_file)
{
	std::string wgetFile = "wget -q -T 10 -t 1 " + url + " -O " + image_file;
	int nRet = system( wgetFile.c_str() );
	//printf("wgetFile:%s\n",wgetFile.c_str());

/*	char wgetFile[1024]={0};
	sprintf( wgetFile, "wget -q -T 3 -t 1 %s -O %s", url.c_str(), image_file.c_str() );
	printf("wgetFile:%s\n",wgetFile);
	int nRet = system( wgetFile );
	printf("wgetFile:%s\n",wgetFile);
	printf("url:%s\n",url.c_str());
	printf("image_file:%s\n",image_file.c_str());
	printf("\n");*/

	return 0;
}

/***********************************id2url*************************************/
string API_WGET_INIM::id2url(const std::string& id)
{
	Downloader dd;
	std::string imurl("");
	std::string url = "http://in.itugo.com/api/getphotourl?id=" + id; 
	std::cout<<url<<std::endl;
	std::string str = dd.get_content(url);
	std::cout<<str<<std::endl;
	Json::Reader reader;
	Json::Value root;
	if (reader.parse(str.c_str(), root))
    {
		if(root["data"].isString()){
			std::string sdata = root["data"].asString();
			imurl = sdata;
      }//if data
    }//if json
	std::cout<<"curl: "<<imurl<<std::endl;
	return imurl;
}

