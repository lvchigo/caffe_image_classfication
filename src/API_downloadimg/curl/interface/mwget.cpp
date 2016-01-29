#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <cstdio>
#include <cmath>
#include <sys/stat.h>

#include <pthread.h>
#include <curl/curl.h>
#include "string_operator.h"

RunTimer<double> rt;
const int THREADCOUNT = 20;
static pthread_mutex_t g_mutex = PTHREAD_MUTEX_INITIALIZER;
typedef struct inim
{
	std::string _id;
	//std::string _tag;
	std::string _url;

	inim(std::string id, std::string url)
	{
		_id = id;
		//_tag = tag;
		_url = url;
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

std::vector<inim> parase2(const std::string& list_file)
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
			std::cout<<line<<std::endl;
			std::vector<std::string> cell = split(line, ",");
			inim im(cell[0], cell[1]);
			im.print();
	        list.push_back(im);
	
			++count;
		}/*loop*/
	
	
	//
	stream_reader.close();

	return list;
}

int wget_image_file(const std::string &url, const std::string &image_file)
{
	std::string cmd = "wget -q -T 1 -t 1 " + url + " -O " + image_file;
	FILE* fptr = popen(cmd.c_str(), "r");
	pclose(fptr);
	return 1;
}

void set_share_handle(CURL* curl_handle)
{
    static CURLSH* share_handle = NULL;
    if (!share_handle)
	{
		share_handle = curl_share_init();
		curl_share_setopt(share_handle, CURLSHOPT_SHARE, CURL_LOCK_DATA_DNS);
	}
	curl_easy_setopt(curl_handle, CURLOPT_SHARE, share_handle);
	curl_easy_setopt(curl_handle, CURLOPT_DNS_CACHE_TIMEOUT, 60 * 5);
}

int mcurl(int argc, char** argv)
{
	std::string list_file("mwget1k.list");
    std::vector<inim> imlist = parase2(list_file);

    curl_global_cleanup();
	
	curl_global_init( CURL_GLOBAL_ALL );
	return 0;
}



void *wgetThread (void *pData)
{
	inim *p = (inim *) pData;
	
	//pthread_mutex_lock (&g_mutex);
    wget_image_file(p->_url, p->_id+".jpg");
	//pthread_mutex_unlock (&g_mutex);
    
	std::cout<<"####Thread exit id="<<pthread_self()<<"####"<<std::endl;
	pthread_exit (0);

	return NULL;
}

int mwget(int argc, char** argv)
{
	std::string list_file("mwget1k.list");
    std::vector<inim> imlist = parase2(list_file);
    int imNum = imlist.size();

    int threadNum = THREADCOUNT;
	pthread_t ptid[THREADCOUNT];
	rt.start();
    for(int i = 0; i<imNum; i += threadNum)
	{
		int t = 0;
		for(t = 0; t < threadNum; ++t){
			int idx = i+t;
			if (idx == imNum) break;

			int rc = pthread_create (&ptid[t], NULL, wgetThread, &(imlist[idx]) );
		}

		for(int j = 0; j<t; ++j)
			pthread_join(ptid[j], NULL);
    }
	rt.end();
	std::cout<<"run time: "<<rt.time()<<" avg: "<<rt.time()/imNum<<std::endl;
	return 0;
}
int swget(int argc, char** argv)
{
	std::string list_file("mwget1k.list");
    std::vector<inim> imlist = parase2(list_file);
    int imNum = imlist.size();

    
	rt.start();
    for(int i = 0; i<imNum; ++i)
	{
		wget_image_file(imlist[i]._url, imlist[i]._id+".jpg");
		
    }
	rt.end();
	std::cout<<"run time: "<<rt.time()<<" avg: "<<rt.time()/imNum<<std::endl;
	return 0;
}
int main(int argc, char** argv)
{

	//mwget(argc, argv);
	swget(argc, argv);;
	return 0;
}
