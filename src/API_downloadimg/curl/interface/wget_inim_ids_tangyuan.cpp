#include <iostream>
#include <string>
#include <fstream>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#include "string_operator.h"
#include "json/json.h"
#include "downloader.h"
#include "wget_inim_ids.hpp"

#include "boost/threadpool.hpp"
using boost::threadpool::pool;

//Downloader dd;
RunTimer<double> rt;
const int THREADCOUNT = 16;
const int MAXITERS = 200;
static pthread_mutex_t g_mutex = PTHREAD_MUTEX_INITIALIZER;

typedef struct imnode
{
  std::string _id;
  std::string _url;
  std::string _imf;
  std::string _res;
  void print()
  {
    std::cout<<std::endl<<"================="<<std::endl;
    std::cout<<"id:"<<_id<<std::endl;
    std::cout<<"url:"<<_url<<std::endl;
    std::cout<<"imf:"<<_imf<<std::endl;
    std::cout<<"res:"<<_res<<std::endl;
    std::cout<<"================="<<std::endl;
  }
}imnode;

typedef struct inim
{
	std::string _user;
	std::string _id;
	//std::string _tag;
	std::string _url;

	inim(std::string id, std::string url)
	{
		_id = id;
		//_tag = tag;
		_url = url;
		//_user = user;
	}

  void print()
  {
    std::cout<<std::endl;
    std::cout<<_id<<std::endl;
    //std::cout<<_tag<<std::endl;
    std::cout<<_url<<std::endl;
    std::cout<<std::endl;

  }
}inim;

int wget_image_file(const std::string &url, const std::string &image_file)
{
  std::string cmd = "wget -q -T 1 -t 1 " + url + " -O " + image_file;
  std::cout<<cmd<<std::endl;
  FILE* fptr = popen(cmd.c_str(), "r");
  pclose(fptr);
  return 1;
}

std::string id2url(const std::string& id)
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

std::vector<inim> parase(const std::string& list_file)
{
  std::ifstream stream_reader(list_file.c_str());
  std::string line;

  int count = 0;
  std::vector< inim > list;
  list.reserve(100000);

  while( std::getline(stream_reader, line) )
    {
      if( std::strlen(line.c_str()) == 0 )
	{
	  continue;
	}
      //std::cout<<line<<std::endl;
	  // std::vector<std::string> cell = split(line, ",");
      inim im(line, "");
      //im.print();
      list.push_back(im);

      ++count;
    }/*loop*/
  stream_reader.close();
  return list;
}

void task_with_parameter(imnode* nd)
{
  RunTimer<double> rtm;
  rtm.start();
  //nd->_url = id2url(nd->_id);
  wget_image_file(nd->_url, nd->_imf);
  nd->print();
  rtm.end();
  std::cout<<"-->"<<nd->_id<<": "<<rtm.time()<<std::endl;
}

int batch(int argc, char** argv)
{
	std::string out_dir(argv[2]);//("/home/tangyuan/ssd/Data/test/ads/text0508");
	std::string list_file(argv[1]);//("/home/tangyuan/Downloads/text0508-urls.csv");
	std::vector<inim> imlist = parase(list_file);
	int imNum = imlist.size();
  
	pool* mtp = new pool(THREADCOUNT);
	int threadNum = THREADCOUNT;
	//imNum = MAXITERS;
	RunTimer<double> rts;
	rts.start();
	for(int i = 0; i<imNum; i ++)
    {
      	imnode* nd = new imnode();
        
        nd->_id = imlist[i]._id;
		//nd->_url = imlist[i]._url;
        nd->_imf = out_dir + "/" + imlist[i]._id + ".jpg";
        //nd->print();
		nd->_url = id2url(nd->_id);
		getchar();
		mtp->schedule(boost::bind(task_with_parameter, nd));
		
		//std::cout<<ge->recoWeb(nd->_url)<<std::endl;
    }
	mtp->wait();
	rts.end();
  
	std::cout<<"run time: "<<rts.time()<<" avg: "<<rts.time()/imNum<<std::endl;
	return 0;
   
}

int main(int argc, char** argv)
{
 
  batch(argc, argv);
  return 0;
}
