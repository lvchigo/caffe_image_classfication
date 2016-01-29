/*
 * =====================================================================================
 *
 *       filename:  string_operator.h
 *
 *    description:  string operator functions
 *
 *        version:  1.0
 *        created:  2014-10-28
 *       revision:  none
 *       compiler:  g++
 *
 *         author:  tangyuan
 *        company:  itugo.com
 *
 *      copyright:  2014 itugo Inc. All Rights Reserved.
 *      
 * =====================================================================================
 */

#ifndef STRING_OPERATOR_H
#define STRING_OPERATOR_H

#include <iostream>
#include <cstring>
#include <string>
#include <sstream>
#include <ctime>
#include <sys/time.h>
#include <vector>
#include <algorithm>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>

	const char SLASH_CHR     = '/';        /* slash char in string */
	const char DOT_CHR       = '.';        /* dot char in string */
        
static inline void string_replace(std::string & strBig, const std::string & strsrc, const std::string &strdst)
	{
		std::string::size_type pos=0;
		std::string::size_type srclen=strsrc.size();
		std::string::size_type dstlen=strdst.size();
		while( (pos=strBig.find(strsrc, pos)) != std::string::npos)
	{
		strBig.replace(pos, srclen, strdst);
		pos += dstlen;
	}
	}

	/** local (static) compare function for the qsort() call */
	static inline int str_compare( const void *arg1, const void *arg2 )
	{
		return strcmp ( ( * ( std::string* ) arg1 ).c_str (), ( * ( std::string* ) arg2 ).c_str () );
	}

	/**
	 * @brief split a string by delim
	 *
	 * @param str string to be splited
	 * @param c delimiter, const char*, just like " .,/", white space, dot, comma, splash
	 *
	 * @return a string vector saved all the splited world
	 */
	static inline std::vector<std::string> split(const std::string& str, const char* c)
    {
			char *cstr, *p;
			std::vector<std::string> res;
			cstr = new char[str.size()+1];
			strcpy(cstr,str.c_str());
			p = strtok(cstr,c);
			while(p!=NULL)
			{
				res.push_back(p);
				p = strtok(NULL,c);
			}
			return res;
	}
 
	/* get file name(with suffix) from file full path */
	static inline std::string get_file_name(const std::string &file_path)
	{
		int len = file_path.length();
		int slash_pos = file_path.find_last_of(SLASH_CHR);
		if(slash_pos < 0)
	    {
			return file_path;
		}
		
		return file_path.substr(slash_pos + 1, len - slash_pos);
	}

	/* get file short name(without suffix) from file full path */
	static inline std::string get_file_short_name(const std::string &file_path)
	{
		int dot_pos = file_path.find_last_of(DOT_CHR);
		dot_pos = (dot_pos < 0)? 0 : dot_pos;
		
		int slash_pos = file_path.find_last_of(SLASH_CHR);
		if(slash_pos < 0)
		{
			return (dot_pos > 0)? file_path.substr(0, dot_pos) : file_path;
		}
		
		if(dot_pos - slash_pos < 1)
		{
			return file_path;
		}
		
		return file_path.substr(slash_pos + 1, dot_pos - slash_pos - 1);
	}

	/* get file path from file full path */
	static inline std::string get_file_path(const std::string file_path)
	{
		
		int slash_pos = file_path.find_last_of(SLASH_CHR);
		if(slash_pos < 1)
		{
			std::string ret("");
			return ret;
		}
		
		return file_path.substr(0, slash_pos );
	}

	/* get category name from directory path */
	static inline std::string get_category_name(const std::string path)
	{
		int len = path.length();
		if( len < 2 )
		{
			std::string ret("");
			return ret;
		}

		int slash_pos = path.find_last_of(SLASH_CHR);
		slash_pos = (slash_pos < 0)? 0 : slash_pos;
		if( slash_pos == len - 1 )
		{
			std::string tmp = path.substr(0, slash_pos);
			slash_pos = tmp.find_last_of(SLASH_CHR);
			slash_pos = (slash_pos < 0)? 0 : slash_pos;
			return tmp.substr(slash_pos + 1, len - 1 - slash_pos);
		}
    
        return path.substr(slash_pos + 1, len - slash_pos);
	}

	/* get category name from file full path*/
	static inline std::string get_file_category_name(const std::string &file_path)
	{
		return get_category_name( get_file_path(file_path) );
	}

	/* get file full path with dir and name */
	static inline std::string get_full_file(const std::string &file_dir, const std::string &file_name)
	{
		int len = file_dir.length();
		if( SLASH_CHR == file_dir[len - 1])
		{
			return file_dir + file_name;
		}
		
		std::string slash_str("");
		slash_str = SLASH_CHR;
		return file_dir + slash_str + file_name;
	}

	
	/* string to number */
	template<typename T>
	T string_to_number(const std::string &str)
	{
		T val;
		std::stringstream stream(str);
		stream >> val;
		
		return val;
	}

	/* number to string */
	template<typename T>
	std::string  number_to_string(const T &val)
	{
		
		std::ostringstream stream;
		stream << val;

		return stream.str();
	}
        
        /*=================================================*/
        /* file exists */
	static inline int file_exists(const std::string &filename)   
	{   // access(filename,   0) == 0 :  file exist   
		return   (access(filename.c_str(),   0)   ==   0);   
	}

        /* dir exists */
        static inline int dir_exists(const std::string dir_path)
        {
            struct stat global_stat;
            return (stat(dir_path.c_str(), &global_stat) == 0 && S_ISDIR(global_stat.st_mode)) ;
        }

static inline int do_mkdir(const char *path, mode_t mode)
{
    struct stat            st;
    int             status = 0;

    if (stat(path, &st) != 0)
    {
        /* Directory does not exist. EEXIST for race condition */
        if (mkdir(path, mode) != 0 && errno != EEXIST)
            status = -1;
    }
    else if (!S_ISDIR(st.st_mode))
    {
        errno = ENOTDIR;
        status = -1;
    }

    return (status);
}
static inline int do_mkdir(const std::string path)
{
  return do_mkdir(path.c_str(), 0777);
}
	/*=================================================*/
	/**
	 * @brief   program runtimer
	 */
    template<class T = double>
		class RunTimer
		{
		public:
		
		/*constructor*/
		RunTimer()
		:rt_start(0), rt_end(0)
		{}
		
		/*destructor*/
		~RunTimer()
		{}

		void start()
		{
			//rt_start = static_cast<T>(clock()) / CLOCKS_PER_SEC;
			//rt_end   = rt_start;
			gettimeofday(&tv_start, NULL);
			gettimeofday(&tv_end, NULL);
		}

		void end()
		{
			//rt_end = static_cast<T>(clock()) / CLOCKS_PER_SEC;
			gettimeofday(&tv_end, NULL);
		}

		T time()
		{
			//return (rt_end - rt_start);
			return static_cast<T>(timediff(&tv_start, &tv_end) / 1000000.0);
		}
		private:
		T rt_start;
		T rt_end;
		timeval tv_start;
		timeval tv_end;

		long long int timediff(timeval *start, timeval *end)
		{
			return (
					(end->tv_sec * 1000000 + end->tv_usec) -
					(start->tv_sec * 1000000 + start->tv_usec)
					);
		}
		};/*RunTimer*/


	

#endif
