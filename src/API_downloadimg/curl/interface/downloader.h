/*
 * =====================================================================================
 *
 *       filename:  downloader.h
 *
 *    description:  download html/image from url by curl
 *
 *        version:  1.0
 *        created:  2013-04-20
 *       revision:  none
 *       compiler:  g++
 *
 *         author:  amadeuzou
 *        company:  
 *
 *      copyright:  2013  Inc. All Rights Reserved.
 *      
 * =====================================================================================
 */

#ifndef DOWNLOADER_H
#define DOWNLOADER_H

#include <iostream>
#include <string>
#include <vector>
#include <fstream>

#include <curl/curl.h>
#include "log.h"


	/*
	 * @author          amadeuzou
	 * @brief           download html/image from url by curl
	 */
	class Downloader
	{
	public:
		/* constructor */
		Downloader();

		/* destructor */
		~Downloader();

        /* reset curl configure */
        void reset();

		/*release*/
		void release();
		
		/* download html file from url */
	    int download_html_file(const std::string &url, const std::string &html_file);

		/* download image file from url */
		int download_image_file(const std::string &url, const std::string &image_file);

		/* download to string content from url */
		std::string get_content(const std::string &url);

		/*set timeout*/
		void set_curl_timeout(long t)
		{
			timeout = t;
		}

		/*log configure*/
		void set_log(Logger &g)
		{
			log = g;
		}

		void set_log_file(const std::string &log_file)
		{
			log.set_file(log_file);
		}

		void set_log_level(int v)
		{
			log.set_level(v);
		}
		
		// set proxy service
		void set_proxy_host(const std::string &host_str)
		{
			curl_easy_setopt(curl_handle, CURLOPT_PROXY, host_str.c_str()); 
		}
		// set proxy port
		void set_proxy_port(int port)
		{
			curl_easy_setopt(curl_handle, CURLOPT_PROXYPORT, port); 
		}
		// set proxy: "192.168.2.150:2516"
		void set_proxy(const std::string &proxy_str)
		{
			curl_easy_setopt(curl_handle, CURLOPT_PROXY, proxy_str.c_str()); 
		}

		void set_share_handle()
		{
			//static CURLSH* share_handle = NULL;                                               
			if (!share_handle)
				{
					share_handle = curl_share_init();
					curl_share_setopt(share_handle, CURLSHOPT_SHARE, CURL_LOCK_DATA_DNS);
				}
			curl_easy_setopt(curl_handle, CURLOPT_SHARE, share_handle);
			curl_easy_setopt(curl_handle, CURLOPT_DNS_CACHE_TIMEOUT, 60 * 5);
		}    

		int wget_image_file(const std::string &url, const std::string &image_file)
		{
			std::string cmd = "wget -T 3 -t 3 " + url + " -O " + image_file + " -q ";
			FILE* fptr = popen(cmd.c_str(), "r");
			pclose(fptr);
			return 1;
		}

	private:
		
		/* curl callback function for memory data*/
		static size_t write_to_data(void *ptr, size_t size, size_t nmemb, void *data);
		

		/* curl callback function for stream */
		static size_t write_to_stream(void *ptr, size_t size, size_t nmemb, void *data);
		
		
		
	private:

		/* curl handle */
		CURL *curl_handle;

		static CURLSH* share_handle;// = NULL;                                                            
		/* timeout */
		long timeout;

		/*log*/
		Logger log;
		
	}; /* Downloader */
	

#endif
