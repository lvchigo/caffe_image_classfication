#include <iostream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <unistd.h>
#include <pthread.h>
#include <curl/curl.h>

using namespace std;

struct tNode
{
	FILE *fp;
	long startPos;
	long endPos;
	void *curl;
	pthread_t tid;
};

int threadCnt = 0;
static pthread_mutex_t g_mutex = PTHREAD_MUTEX_INITIALIZER;



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

static size_t writeFunc (void *ptr, size_t size, size_t nmemb, void *userdata)
{
	tNode *node = (tNode *) userdata;
	size_t written = 0;
	pthread_mutex_lock (&g_mutex);
	if (node->startPos + size * nmemb <= node->endPos)
	{
		fseek (node->fp, node->startPos, SEEK_SET);
		written = fwrite (ptr, size, nmemb, node->fp);
		node->startPos += size * nmemb;
	}
	else
	{
		fseek (node->fp, node->startPos, SEEK_SET);
		written = fwrite (ptr, 1, node->endPos - node->startPos + 1, node->fp);
		node->startPos = node->endPos;
	}
	pthread_mutex_unlock (&g_mutex);
	return written;
}

int progressFunc (void *ptr, double totalToDownload, double nowDownloaded, double totalToUpLoad, double nowUpLoaded)
{
	int percent = 0;
	if (totalToDownload > 0)
	{
		percent = (int) (nowDownloaded / totalToDownload * 100);
	}

    if(percent % 20 == 0)
	    printf ("downloading %0d%%\n", percent);
	return 0;
}

size_t write_data(void *ptr, size_t size, size_t nmemb, void *stream)
{
    return size * nmemb;
}

long getDownloadFileLenth (const char *url)
{
	double downloadFileLenth = 0;
	CURL *handle = curl_easy_init ();
	set_share_handle( handle );
	curl_easy_setopt (handle, CURLOPT_URL, url);
	curl_easy_setopt (handle, CURLOPT_HEADER, 1);	
	curl_easy_setopt (handle, CURLOPT_NOBODY, 1);
	curl_easy_setopt(handle, CURLOPT_WRITEFUNCTION, write_data);
	if (curl_easy_perform (handle) == CURLE_OK)
	{
		curl_easy_getinfo (handle, CURLINFO_CONTENT_LENGTH_DOWNLOAD, &downloadFileLenth);
	}
	else
	{
		downloadFileLenth = -1;
	}
	return downloadFileLenth;
}



void *workThread (void *pData)
{
	tNode *pNode = (tNode *) pData;

	int res = curl_easy_perform (pNode->curl);

	if (res != 0)
	{

	}

	curl_easy_cleanup (pNode->curl);

	pthread_mutex_lock (&g_mutex);
	threadCnt--;
	//printf ("thred %ld exit\n", pNode->tid);
	pthread_mutex_unlock (&g_mutex);
	delete pNode;
	pthread_exit (0);

	return NULL;
}

bool downLoad (int threadNum, string Url,  string outFile)
{
	long fileLength = getDownloadFileLenth (Url.c_str ());

	if (fileLength <= 0)
	{
		printf ("get the file length error...");
		return false;
	}

	// Create a file to save package.
	FILE *fp = fopen (outFile.c_str (), "wb");
	if (!fp)
	{
		return false;
	}

	long partSize = fileLength / threadNum;

	for (int i = 0; i <= threadNum; i++)
	{
		tNode *pNode = new tNode ();

		if (i < threadNum)
		{
			pNode->startPos = i * partSize;
			pNode->endPos = (i + 1) * partSize - 1;
		}
		else
		{
			if (fileLength % threadNum != 0)
			{
				pNode->startPos = i * partSize;
				pNode->endPos = fileLength - 1;
			}
			else
				break;
		}

		CURL *curl = curl_easy_init ();
        set_share_handle( curl );
		pNode->curl = curl;
		pNode->fp = fp;

		char range[64] = { 0 };
		snprintf (range, sizeof (range), "%ld-%ld", pNode->startPos, pNode->endPos);

		// Download pacakge
		curl_easy_setopt (curl, CURLOPT_URL, Url.c_str ());
		curl_easy_setopt (curl, CURLOPT_WRITEFUNCTION, writeFunc);
		curl_easy_setopt (curl, CURLOPT_WRITEDATA, (void *) pNode);
		curl_easy_setopt (curl, CURLOPT_NOPROGRESS, 1L);
		//curl_easy_setopt (curl, CURLOPT_PROGRESSFUNCTION, progressFunc);
		curl_easy_setopt (curl, CURLOPT_NOSIGNAL, 1L);
		curl_easy_setopt (curl, CURLOPT_LOW_SPEED_LIMIT, 1L);
		curl_easy_setopt (curl, CURLOPT_LOW_SPEED_TIME, 1L);
		curl_easy_setopt (curl, CURLOPT_RANGE, range);

		pthread_mutex_lock (&g_mutex);
		threadCnt++;
		pthread_mutex_unlock (&g_mutex);
		int rc = pthread_create (&pNode->tid, NULL, workThread, pNode);
	}

	while (threadCnt > 0)
	{
		usleep (100L);
	}

	fclose (fp);

	//printf ("download succed......\n");
	return true;
}

int main (int argc, char *argv[])
{
	if (argc < 4)
		std::cout<<"usage: do <n threads> <url> <output file>"<<std::endl;
	
	curl_global_init( CURL_GLOBAL_ALL );
	int np = atoi(argv[1]);
	std::string url(argv[2]);
	std::string output(argv[3]);
	
	downLoad(np, url, output);
	//std::cout<<"it's done!"<<std::endl;
	curl_global_cleanup();
	
	return 0;
}
