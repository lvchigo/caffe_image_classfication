#include <iostream>
#include <string>
#include <fstream>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <tr1/memory> 

#include "string_operator.h"
//#include "geekeye.h"
#include "json/json.h"
#include "downloader.h"
#include <curl/curl.h>
#include <time.h>

//add by xiaogao-20150414
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include "API_in36class_classification.h"
#include "threadpool.hpp"
#include <boost/bind.hpp>
#include <boost/thread/mutex.hpp>

using namespace cv;
using namespace std;
using boost::threadpool::pool;


//=================================================================================//
const std::string URLTEMP = "http://in.itugo.com/api/getphotogeekeye?limit=";
const std::string URLPOST = "http://in.itugo.com/api/revphotogeekeyeres";
const std::string POSTHEAD = "res=";
RunTimer<double> rt;
static pthread_mutex_t g_mutex = PTHREAD_MUTEX_INITIALIZER;
const int THREADCOUNT = 10;
const int NLIMIT = 100;

typedef struct tagThreadParam
{
	int para;
	int nThreads;
}ThreadParam;

typedef struct inim
{
    std::string _id;
    //std::string _tag;
	std::string _url;
	
	inim(){
		
	}
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

typedef struct imnode
{
	std::string _id;
	std::string _url;
	std::string _imf;
}imnode;

//=================================================================================//
vector<string> vecRes;
vector<imnode> vecImNode;
API_IN36CLASS_CLASSIFICATION api_in36class_classification;

//=================================================================================//
int wget_image_file(const std::string url, const std::string image_file);
int wget_image_file_wget(const std::string &url, const std::string &image_file);
int get_im_list(int im_count, std::vector<inim> &imlist);
void *workThread (void *para);
int batch(const std::vector<inim>& imlist, std::string &results, int &numRes);
size_t write_data(void *ptr, size_t size, size_t nmemb, void *data);
int curl_get(const std::string &Url, std::string &UrlCallback);
int curl_send(const std::string& Res, std::string &ResCallback);
int get_status(const std::string& str);
int IN_ImgLabel( const char *szKeyFiles, const int binGPU );

/*********************************threadpool*************************************/
boost::mutex m_io_monitor;
void task_threadpool( const std::vector<inim> imlist, const int TaskID);
int test_threadpool( const char *szKeyFiles, const int binGPU );

//=================================================================================//
int main(int argc, char** argv)
{
	int  ret = 0;

	if (argc == 4 && strcmp(argv[1],"-label") == 0) {
		ret = IN_ImgLabel( argv[2], atol(argv[3]));
	}
	else if (argc == 4 && strcmp(argv[1],"-threadpool") == 0) {
		ret = test_threadpool( argv[2], atol(argv[3]));
	}
	else
	{
		cout << "usage:\n" << endl;
		cout << "\tDemo -label keyFilePath binGPU\n" << endl;
		cout << "\tDemo -threadpool keyFilePath binGPU\n" << endl;
		return ret;
	}	
	return ret;
}


//=================================================================================//
int curl_get(const std::string &Url, std::string &UrlCallback)
{
	UrlCallback = "";
	if( ( Url == "" ) || ( Url == "null" ) )
	{
		printf( "curl get Input:%s err!!\n", Url.c_str() );
		return -1;
	}
	else
		printf( "Curl get Input:%s\n",Url.c_str() );

	/* init the curl session */
	CURL *curl = curl_easy_init();
	if(NULL == curl)
	{
		printf( "curl get:curl_easy_init error.\n" );
		curl_easy_cleanup(curl);
		return -1;
	}

	curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);
	curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 5L);
	curl_easy_setopt(curl, CURLOPT_LOW_SPEED_LIMIT, 1L);
	curl_easy_setopt(curl, CURLOPT_LOW_SPEED_TIME, 5L);
	curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
	curl_easy_setopt(curl, CURLOPT_URL, Url.c_str());
	curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
	curl_easy_setopt(curl, CURLOPT_WRITEDATA, static_cast<void *>(&UrlCallback) );
	if (curl_easy_perform(curl) != CURLE_OK)
	{
		printf( "curl get %s error.\n",Url.c_str() );
		curl_easy_cleanup(curl);
		return -1;
	}	
	curl_easy_cleanup(curl);

	if( ( UrlCallback == "" ) || ( UrlCallback == "null" ) )
	{
		printf( "curl get Callback:%s\n", UrlCallback.c_str() );
		return -1;
	}
	else
		printf( "Curl get Callback:%s\n",UrlCallback.c_str() );
	
	return 0;
}

int curl_send(const std::string& Res, std::string &ResCallback)
{
	ResCallback = "";
	if( ( Res == "" ) || ( Res == "null" ) )
	{
		printf( "curl send Input:%s err!!\n", Res.c_str() );
		return -1;
	}
	else
		printf( "curl send Input:%s\n",Res.c_str() );

	//CURLcode res;
        CURL *curl = curl_easy_init();
	if(NULL == curl)
	{
		printf( "curl send:curl_easy_init error.\n" );
		curl_easy_cleanup(curl);
		return -1;
	}

	curl_easy_setopt(curl, CURLOPT_URL, URLPOST.c_str());
	curl_easy_setopt(curl, CURLOPT_POSTFIELDS, Res.c_str());
	curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
	curl_easy_setopt(curl, CURLOPT_WRITEDATA, static_cast<void *>(&ResCallback) );
	curl_easy_setopt(curl, CURLOPT_POST, 1);
	//curl_easy_setopt(curl, CURLOPT_VERBOSE, 1);
	//curl_easy_setopt(curl, CURLOPT_HEADER, 1);
	curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1);
	curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
	//curl_easy_setopt(curl, CURLOPT_COOKIEFILE, "curlposttest.cookie");
	//res = curl_easy_perform(curl);
	if (curl_easy_perform(curl) != CURLE_OK)
	{
		printf( "curl send %s error.\n",Res.c_str() );
		curl_easy_cleanup(curl);
		return -1;
	}
	curl_easy_cleanup(curl);

	if( ( ResCallback == "" ) || ( ResCallback == "null" ) )
	{
		printf( "curl send Callback:%s err!!\n", ResCallback.c_str() );
		return -1;
	}
	else
		printf( "curl send Callback:%s\n",ResCallback.c_str() );

	return 0;
}


int get_status(const std::string& str)
{
	Json::Reader reader;
	Json::Value root;

	
	if (reader.parse(str.c_str(), root))
	{
		// 0--true
		return (0 == str_compare(root["succ"].asString().c_str(), "true" ) );
				
    	}//if json

	return 0;
}

int get_im_list(int im_count, std::vector<inim>& imlist)
{
	int nRet = 0;
	imlist.clear();
	imlist.reserve(1000);

	std::string url_list = URLTEMP + number_to_string<int>(im_count);
	
	//printf( "get_im_list[1]:get_content...\n" );
	std::string str = "";
	nRet = curl_get( url_list, str);
	if( nRet != 0 )
	{
		return nRet;
	}

	Json::Reader reader;
	Json::Value root;

	//std::cout<<"get_im_list[2]:reader.parse"<<std::endl;
	if (reader.parse(str.c_str(), root))
	{
		// 0--true
		//std::cout<<"get_im_list[3]:str_compare[1.0]"<<std::endl;
		//printf( "root[succ]:%s!!\n",root["succ"].asString().c_str() );
		if( 0 == strcmp( root["succ"].asString().c_str(), "true" ) )
		{
			//std::cout<<"get_im_list[4]:str_compare[1.1]"<<std::endl;
			const Json::Value iters = root["data"];
			//std::cout<<iters.size()<<std::endl;
			//std::cout<<"get_im_list[5]:str_compare[1.2]"<<std::endl;
			for(int i = 0; i<iters.size(); ++i){
			    inim im(iters[i]["id"].asString(), iters[i]["url"].asString() );
				imlist.push_back(im);
				//im.print();
			}//for-data
			
		}//if-succ
		else
		{
			printf( "get_im_list[6]:(root !=succ);str=%s\n", str.c_str() );
			nRet = -1;
		}
	
    	}//if json
	else
	{
		printf( "get_im_list[7]:reader.parse err;str=%s\n", str.c_str() );
		nRet = -1;
	}

	if(imlist.size()<1) 
	{
		printf( "get_im_list[8]:(imlist.size()<1);str=%s\n", str.c_str() );
		nRet = -1;
	}

	return nRet;	
}

size_t write_data(void *ptr, size_t size, size_t nmemb, void *data)
{
	size_t written = size*nmemb;
	std::string *str = static_cast<std::string *>(data);
	str->append( (char *)ptr, written);
	
	return written;
}

int wget_image_file(const std::string url, const std::string image_file)
{
	std::string wgetFile = "wget -q -T 3 -t 1 " + url + " -O " + image_file;
	int nRet = system( wgetFile.c_str() );

	return 0;
}

int wget_image_file_wget(const std::string &url, const std::string &image_file)
{
	std::string cmd = "wget -q -T 3 -t 1 " + url + " -O " + image_file;

	FILE* fptr;
	if( ( fptr = popen(cmd.c_str(), "r") ) == NULL )
	{
		printf( "wget_image_file_wget err:%s not find!!\n", cmd.c_str() );
		return -1;
	}
	pclose(fptr);
	return 0;
}

//add by xiaogao-20150414
void *workThread (void *para)
{
	int i,idx,saveResNum,nRet = 0;
	const long ImageID = 1;
	char szImgPath[256] = {0};
	char rmImgPath[256] = {0};
	std::string res = "";
	ThreadParam *pParam = (ThreadParam*)para;
	
	for (idx = pParam->para; idx < vecImNode.size(); idx += pParam->nThreads) 
	{
		//printf("Start workThread[0]:idx-%d,size-%d\n",idx,vecImNode.size());
		vector< pair< string, float > > LabelInfo;
		LabelInfo.clear();
		//float tWget,tGetLabel,tMergeLabel,tWriteData;
	
		/*****************************Get Image File*****************************/
		//printf("Start workThread[1]:Get Image File...\n");
		//rt.start();
		nRet = wget_image_file(vecImNode[idx]._url, vecImNode[idx]._imf);
		//nRet = wget_image_file_wget(vecImNode[idx]._url, vecImNode[idx]._imf);
		//rt.end();
		//tWget = rt.time();	
		if (nRet != 0)
		{
			printf("wget_image_file err!!\n");
			//continue;
		}

		/*****************************Load Image*****************************/
		//printf("Start workThread[2]:Load Image:%s\n",vecImNode[idx]._imf.c_str());
		IplImage *img = cvLoadImage( vecImNode[idx]._imf.c_str() );
		if(!img || (img->width<16) || (img->height<16) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
		{	
			cout<<"Can't open " << vecImNode[idx]._imf.c_str() << endl;
			nRet = -1;
			//continue;
		}	
	
		/*****************************Get Label*****************************/
		if( 0 == nRet )
		{
			//printf("Start workThread[3]:Get Label...\n");			
			pthread_mutex_lock (&g_mutex);
			//rt.start();
			nRet = api_in36class_classification.Predict(img, ImageID, "fc7", 3, LabelInfo);
			//rt.end();
			//tGetLabel = rt.time();
			pthread_mutex_unlock (&g_mutex);		
			if (nRet != 0)
			{
			   cout<<"Fail to Predict!! "<<endl;
			   //continue;
			}
		}

		/*****************************Merge Result*****************************/
		//printf("Start workThread[5]:Merge Result...\n");		
		res = "null,-1,null,-1";
		if ( ( nRet == 0 ) && ( LabelInfo.size() != 0 ) )
		{
			res = "";
			saveResNum = 2;	//output 2 img info
			saveResNum = (saveResNum>LabelInfo.size())?LabelInfo.size():saveResNum;
			for ( i=0;i<saveResNum;i++ )
			{
				sprintf(szImgPath, "%s,%.2f,", LabelInfo[i].first.c_str(), LabelInfo[i].second );
				res += szImgPath;
			}

			if( 1 == saveResNum )
			{
				res += "null,-1";
			}
		}

		/*******************************curl_post***************************************/
		//printf("Start workThread[6]:curl_post...\n");
		vecRes.push_back( vecImNode[idx]._id + "," + res );

		/*****************************Release*****************************/
		if(img) { cvReleaseImage(&img);img = 0; }

		/*****************************rm img file*****************************/
		sprintf(rmImgPath, "rm %s", vecImNode[idx]._imf.c_str() );
		nRet = system(rmImgPath);
		
		//printf("Time:Wget-%.4f,GetLabel-%.4f,MergeLabel-%.4f,WriteData-%.4f\n",
		//	tWget,tGetLabel,tMergeLabel,tWriteData);
	}

	/*******************************clr img***************************************/
	//printf("thread %d Over!\n", pParam->para);
	pthread_exit (0);

	return NULL;
}

int batch(const std::vector<inim>& imlist, std::string &results, int &numRes)
{
	int i,j,t;
	vecImNode.clear();
	vecRes.clear();
	numRes = 0;
	
	if(imlist.size()<1) 
		return -1;
	
	/*********************************Load Img Data*************************************/
	//printf("Start batch[1]:Load Img Data...\n");
	for(i=0; i<imlist.size(); ++i)
	{
		imnode nds;
		nds._id = imlist[i]._id;
		nds._url = imlist[i]._url;
		nds._imf = "tmp/" + imlist[i]._id + ".jpg";
		vecImNode.push_back(nds);
	}

	/*********************************MutiThread*************************************/
	//printf("Start batch[2]:MutiThread...\n");		
	{
		pthread_t *pThread = new pthread_t[THREADCOUNT];
		ThreadParam *pParam = new ThreadParam[THREADCOUNT];
		
		for(i=0; i<THREADCOUNT; ++i)
		{
			pParam[i].para = i;
			pParam[i].nThreads = THREADCOUNT;

			int rc = pthread_create(pThread+i, NULL, workThread,(void*)(pParam+i));
		}

		for(i=0; i<THREADCOUNT; ++i)
		{
			pthread_join(pThread[i], NULL);
		}
		sleep (1);
		delete [] pThread;
		delete [] pParam;
	}

	if(vecRes.size()<1) 
		return -1;
	
	/*********************************Merge Res*************************************/
	//printf("Start batch[3]:Merge Res...\n");
	numRes = vecRes.size();
	results = POSTHEAD;
	for(i=0;i<vecRes.size();i++)
	{
		results += vecRes[i] + "\\n";
	}
	vecRes.clear();

	return 0;
}

int IN_ImgLabel( const char *szKeyFiles, const int binGPU )
{
	time_t timep;

	char szImgPath[256];
	int numRes,nRet = 0;	
	RunTimer<double> rtTmp;
	float tGet_im_list,tBatch,tCurl_post;

	/***********************************rm && mkdir tmpdata*************************************/
	nRet = system("mkdir tmp/");

	/***********************************Init*************************************/
	printf("Start api_in36class_classification.Init...\n");
	nRet = api_in36class_classification.Init( szKeyFiles, "fc7", binGPU, 0 ); 
	if (nRet != 0)
	{
	   cout<<"Fail to initialization "<<endl;
	   return nRet;
	}

	/***********************************Process*************************************/
	while(true)
	{
		rt.start();
		
		/***********************************save file**************************************/
		//std::string res_file = number_to_string<long>(time(&timep)) + ".csv";

		/***********************************get_im_list*************************************/
		//printf("Start get_im_list(%d)...\n",NLIMIT);
		std::vector<inim> imlist;
		imlist.clear();
		rtTmp.start();
		nRet = get_im_list( NLIMIT, imlist );
		rtTmp.end();
		tGet_im_list = rtTmp.time();
		if (nRet != 0)
		{
			printf("get_im_list err:breakNum\n");
			sleep(2);	//s
			continue;
		}

		/***********************************MutiThread Process******************************/		
		//printf("Start batch(imlist)...\n");
		numRes = 0;
		std::string results = "";
		rtTmp.start();
		if ( nRet == 0 )
			nRet = batch(imlist, results, numRes);
		rtTmp.end();
		tBatch = rtTmp.time();
		if (nRet != 0)
		{
			printf("MutiThread Process err:breakNum\n");
			sleep(2);	//s
			//continue;
		}

		/***********************************curl_send*************************************/
		//printf("Start curl_post...\n");		
		rtTmp.start();
		std::string ResCallback = "";
		nRet = curl_send(results, ResCallback);
		rtTmp.end();
		tCurl_post = rtTmp.time();
		if (nRet != 0)
		{
			printf("curl_send err:breakNum\n");
			sleep(2);	//s
			//continue;
		}
		
		/***********************************Time*************************************/
		rt.end();

		if ( numRes > 0 )
		{
			printf( "numRes:%d,Run Time:%.4f,avg:%.4f\n", numRes,rt.time(),rt.time()/numRes );
			printf( "Time:Get_im_list-%.4f,Batch-%.4f,Curl_post-%.4f\n", tGet_im_list,tBatch,tCurl_post );
		}
	}
	
	/*********************************Release*************************************/
	printf("Start api_in36class_classification.Release...\n");
	api_in36class_classification.Release();

	return 0;
}

/*********************************threadpool*************************************/
void task_threadpool( const int TaskID )
{
	int i,j,t,numRes_threadpool=0;
	vector<string> vecRes_threadpool;
	vector<imnode> vecImNode_threadpool;

	int idx,saveResNum,nRet = 0;
	const long ImageID = 1;
	char szImgPath[256] = {0};
	char rmImgPath[256] = {0};
	std::string res = "";
	std::string resSend = "";
	RunTimer<double> rtTmp;
	float tGet_im_list,tThreadPool,tCurl_post=0;

	rt.start();
	/*********************************get_im_list*************************************/
	//printf("Start test_threadpool(%d):TaskID-%d\n",NLIMIT,TaskID);
	vector<inim> imlist;
	rtTmp.start();
	nRet = get_im_list( NLIMIT, imlist );
	rtTmp.end();
	tGet_im_list = rtTmp.time();
	if ( (nRet != 0) || ( imlist.size() == 0 ) )
	{
		printf("get_im_list err:breakNum\n");
		sleep(1);	//s
	}

	if ( imlist.size() > 0 )
	{
		/*********************************Load Img Data*************************************/
		//printf("Start task_threadpool[1]:Change Data...\n");
		for(i=0; i<imlist.size(); ++i)
		{
			imnode nds;
			nds._id = imlist[i]._id;
			nds._url = imlist[i]._url;
			nds._imf = "tmp/" + imlist[i]._id + ".jpg";
			vecImNode_threadpool.push_back(nds);
		}

		//printf("Start workThread[%d],size-%ld\n",TaskID,vecImNode_threadpool.size());
		
		rtTmp.start();
		for (idx = 0; idx < vecImNode_threadpool.size(); idx++ ) 
		{
			vector< pair< string, float > > LabelInfo;
			LabelInfo.clear();
			//float tWget,tGetLabel,tMergeLabel,tWriteData;
		
			/*****************************Get Image File*****************************/
			//printf("Start workThread[1]:Get Image File...\n");
			//rt.start();
			nRet = wget_image_file(vecImNode_threadpool[idx]._url, vecImNode_threadpool[idx]._imf);
			//rt.end();
			//tWget = rt.time();	
			if (nRet != 0)
			{
				printf("Start workThread[1]:wget_image_file err,continue!\n");
				//continue;
			}

			/*****************************Load Image*****************************/
			//printf("Start workThread[2]:Load Image:%s\n",vecImNode_threadpool[idx]._imf.c_str());
			IplImage *img = cvLoadImage( vecImNode_threadpool[idx]._imf.c_str() );
			if(!img || (img->width<16) || (img->height<16) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
			{	
				cout<<"Can't open " << vecImNode_threadpool[idx]._imf.c_str() << endl;
				nRet = -1;
				//continue;
			}	
		
			/*****************************Get Label*****************************/
			if( 0 == nRet )
			{
				//printf("Start workThread[3]:Get Label...\n");			
				//pthread_mutex_lock (&g_mutex);
				//rt.start();
				boost::mutex::scoped_lock lock(m_io_monitor);
				nRet = api_in36class_classification.Predict(img, ImageID, "fc7", 3, LabelInfo);
				//rt.end();
				//tGetLabel = rt.time();
				//pthread_mutex_unlock (&g_mutex);		
				if (nRet != 0)
				{
				   cout<<"Fail to GetLabel!! "<<endl;
				   //continue;
				}
			}

			/*****************************Merge Result*****************************/
			//printf("Start workThread[5]:Merge Result...\n");		
			res = "null,-1,null,-1";
			if ( ( nRet == 0 ) && ( LabelInfo.size() != 0 ) )
			{
				res = "";
				saveResNum = 2;	//output 2 img info
				saveResNum = (saveResNum>LabelInfo.size())?LabelInfo.size():saveResNum;
				for ( i=0;i<saveResNum;i++ )
				{
					sprintf(szImgPath, "%s,%.2f,", LabelInfo[i].first.c_str(), LabelInfo[i].second );
					res += szImgPath;
				}

				if( 1 == saveResNum )
				{
					res += "null,-1";
				}
			}

			/*******************************curl_post***************************************/
			//printf("Start workThread[6]:curl_post...\n");
			vecRes_threadpool.push_back( vecImNode_threadpool[idx]._id + "," + res );

			/*****************************Release*****************************/
			if(img) { cvReleaseImage(&img);img = 0; }

			/*****************************rm img file*****************************/
			sprintf(rmImgPath, "rm %s", vecImNode_threadpool[idx]._imf.c_str());
			nRet = system(rmImgPath);
			//printf("Time:Wget-%.4f,GetLabel-%.4f,MergeLabel-%.4f,WriteData-%.4f\n",
			//	tWget,tGetLabel,tMergeLabel,tWriteData);
		}
		rtTmp.end();
		tThreadPool = rtTmp.time();
		
		if(vecRes_threadpool.size()>0)
		{
			/*********************************Merge Res*************************************/
			//printf("Start batch[3]:Merge Res...\n");
			resSend = POSTHEAD;
			for(i=0;i<vecRes_threadpool.size();i++)
			{
				resSend += vecRes_threadpool[i] + "\\n";
			}

			/***********************************curl_send*************************************/	
			//printf("Start workThread[6]:curl_post...\n");
			rtTmp.start();
			std::string ResCallback = "";
			nRet = curl_send(resSend, ResCallback);
			rtTmp.end();
			tCurl_post = rtTmp.time();
			if (nRet != 0)
			{
				printf("curl_send err:breakNum\n");
				sleep(1);	//s
			}
		}

		rt.end();
		/*******************************Print***************************************/
		if ( vecRes_threadpool.size() > 0 )
		{
			printf( "numRes:%ld,Run Time:%.4f,avg:%.4f,ThreadPool-%.4f,Curl_post-%.4f\n", 
				vecRes_threadpool.size(),rt.time(),rt.time()/vecRes_threadpool.size(),tThreadPool,tCurl_post );
		}
		vecRes_threadpool.clear();
	}
}

int test_threadpool( const char *szKeyFiles, const int binGPU )
{
	char szImgPath[256];
	int i,numRes,nRet = 0;	
	int numThreadPool = 4;
	int idThreadPool = -1;
	RunTimer<double> rtTmp;
	float tGet_im_list,tThreadPool;

	/***********************************rm && mkdir tmpdata*************************************/
	nRet = system("mkdir tmp/");

	/***********************************Init*************************************/
	printf("Start api_in36class_classification.Init...\n");
	nRet = api_in36class_classification.Init( szKeyFiles, "fc7", binGPU, 0 ); 
	if (nRet != 0)
	{
	   cout<<"Fail to api_in36class_classification.Init "<<endl;
	   return nRet;
	}

	/***********************************Init threadPool*************************************/
	printf("Start Init threadPool...\n");
	pool threadPool(numThreadPool);

	/***********************************Process*************************************/
	while(true)
	{
		/***********************************Process*************************************/
		for(i=0;i<numThreadPool;i++)
		{
			threadPool.schedule(boost::bind(task_threadpool, i));
		}

		// Wait until all tasks are finished.
		threadPool.wait();
	}
	
	/*********************************Release*************************************/
	printf("Start DL_IMG_LABEL::Release...\n");
	api_in36class_classification.Release();

	return 0;
}
