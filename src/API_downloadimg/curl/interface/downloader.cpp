/*
 * =====================================================================================
 *
 *       filename:  downloader.cpp
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

#include "downloader.h"

#include <exception>
#include <typeinfo>

CURLSH* Downloader::share_handle = NULL;                                                            

	/*
	 * @brief           constructor
	 */
	Downloader::Downloader()
	{
		/* init */
		curl_global_init(CURL_GLOBAL_ALL);

		/* init the curl session */
		curl_handle = curl_easy_init();

		timeout = 5L;
		curl_easy_setopt(curl_handle, CURLOPT_TIMEOUT, timeout + 5L);
		curl_easy_setopt(curl_handle, CURLOPT_CONNECTTIMEOUT, timeout);

		curl_easy_setopt(curl_handle, CURLOPT_LOW_SPEED_LIMIT, 1L);
		curl_easy_setopt(curl_handle, CURLOPT_LOW_SPEED_TIME, 5L);

		curl_easy_setopt(curl_handle, CURLOPT_NOSIGNAL, 1L);
		//curl_easy_setopt(curl_handle, CURLOPT_PROXY, "192.168.2.150:2516");
		if (!share_handle)
			{
				share_handle = curl_share_init();
				curl_share_setopt(share_handle, CURLSHOPT_SHARE, CURL_LOCK_DATA_DNS);
			}
		curl_easy_setopt(curl_handle, CURLOPT_SHARE, share_handle);
		curl_easy_setopt(curl_handle, CURLOPT_DNS_CACHE_TIMEOUT, 60 * 5);
	}

	/*
	 * @brief           destructor
	 */
	Downloader::~Downloader()
	{
		/* cleanup curl stuff */
		if(NULL != curl_handle)
		{
			//curl_easy_cleanup(curl_handle);
			//curl_handle = NULL;
		}
			

	}

	/*
	 * @brief           release
	 */
	void Downloader::release()
	{
		/* cleanup curl stuff */
		if(NULL != curl_handle)
		{
			//curl_easy_cleanup(curl_handle);
			//curl_handle = NULL;
		}

	}
	
	/*
	 * @brief           reset curl configure
	 */
	void Downloader::reset()
	{
		
	}

	/*
	 * @brief   download html file from url
	 *
	 * @param   string          url string
	 * @param   string          html file string
	 *
	 * @return  int             status
	 */
	int Downloader::download_html_file(const std::string &url, const std::string &html_file)
	{
		if(NULL == curl_handle)
		{
			log.fatal()<<"curl handle is null."<<std::endl;
			return 0;
		}
		
		/* set URL to get here */
		curl_easy_setopt(curl_handle, CURLOPT_URL, url.c_str());
		//curl_easy_setopt(curl_handle, CURLOPT_TIMEOUT, timeout + 5L);
		//curl_easy_setopt(curl_handle, CURLOPT_CONNECTTIMEOUT, timeout);
		
		/* send all data to this function  */
		curl_easy_setopt(curl_handle, CURLOPT_WRITEFUNCTION, Downloader::write_to_stream);

		std::ofstream os(html_file.c_str(), std::ios::app);
		if(NULL == os)
		{
		  log.fatal()<<"curl download file stream is null"<<std::endl;
			return 0;
		}
		
		curl_easy_setopt(curl_handle, CURLOPT_WRITEDATA, static_cast<void *>(&os) );
	
		/* get it! */
		curl_easy_perform(curl_handle);
		os.close();
	
		return 1;
	}


	/*
	 * @brief   download image file from url
	 *
	 * @param   string          url string
	 * @param   string          image file string
	 *
	 * @return  int             status
	 */
	int Downloader::download_image_file(const std::string &url, const std::string &image_file)
	{
		if(NULL == curl_handle)
		{
			std::cout<<"curl handle is null."<<std::endl;
			return 0;
		}
		//std::cout<<"curl:"<<url.c_str()<<std::endl;
		/* set URL to get here */
		curl_easy_setopt(curl_handle, CURLOPT_URL, url.c_str());
		//curl_easy_setopt(curl_handle, CURLOPT_TIMEOUT, timeout + 5L);
		//curl_easy_setopt(curl_handle, CURLOPT_CONNECTTIMEOUT, timeout );
		
		/* send all data to this function  */
		curl_easy_setopt(curl_handle, CURLOPT_WRITEFUNCTION, Downloader::write_to_stream);

		std::ofstream os(image_file.c_str(), std::ios::app|std::ios::binary);
		if(NULL == os)
		{
			std::cout<<"curl download file stream is null"<<std::endl;
		  curl_easy_perform(curl_handle);
		  os.close();
			return 0;
		}

		
		curl_easy_setopt(curl_handle, CURLOPT_WRITEDATA, static_cast<void *>(&os) );
		
		/* get it! */
		curl_easy_perform(curl_handle);
		os.close();
	
		return 1;
	}

	/*
	 * @brief   get string content from url
	 *
	 * @param   string          url string
	 *
	 * @return  string          content string
	 */
	std::string Downloader::get_content(const std::string &url)
	{
		/* init */
		//curl_global_init(CURL_GLOBAL_ALL);

		/* init the curl session */
		//curl_handle = curl_easy_init();
		std::string str("");
		try
		{
			//
			//log.info()<<"curl get content."<<std::endl;
		
			if(NULL == curl_handle)
			{
				log.fatal()<<"curl handle is null."<<std::endl;
				return str;
			}
		
			/* set URL to get here */
			curl_easy_setopt(curl_handle, CURLOPT_URL, url.c_str());
			//curl_easy_setopt(curl_handle, CURLOPT_TIMEOUT, timeout + 5L);
			//curl_easy_setopt(curl_handle, CURLOPT_CONNECTTIMEOUT, timeout);
			
			/* send all data to this function  */
			curl_easy_setopt(curl_handle, CURLOPT_WRITEFUNCTION, Downloader::write_to_data);
			
			/* download */
			curl_easy_setopt(curl_handle, CURLOPT_WRITEDATA, static_cast<void *>(&str) );
			
			/* get it! */
			if (curl_easy_perform(curl_handle) != CURLE_OK)
				log.error()<<"curl error."<<std::endl;
			
			return str;
		}
		catch (std::exception& e) {
            log.fatal()<< "Exception: " << e.what() <<std::endl;
			return str;
        }
		
	
	}

	/*
	* @brief    curl callback function for memory data
	*
	* @param    void *          input data pointer        
	* @param    size_t          size of data element
	* @param    size_t          count of data element
	* @param    void *          user data pointer
	*
	* @return   size_t          written data size
	*/
	size_t Downloader::write_to_data(void *ptr, size_t size, size_t nmemb, void *data)
	{
		size_t written = size*nmemb;
		std::string *str = static_cast<std::string *>(data);
		str->append( (char *)ptr, written);
		
		return written;
	}


	/*
	* @brief    curl callback function for stream 
	*
	* @param    void *          input data pointer        
	* @param    size_t          size of data element
	* @param    size_t          count of data element
	* @param    void *          user data pointer
	*
	* @return   size_t          written data size
	*/
	size_t Downloader::write_to_stream(void *ptr, size_t size, size_t nmemb, void *data)
	{
		size_t written = size*nmemb;
		std::ostream *os = static_cast<std::ostream *>(data);
		os->write( (char *)ptr, written);
		return written;
	}


