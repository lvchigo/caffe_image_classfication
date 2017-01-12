#ifndef _API_IMAGEPROCESS_H_
#define _API_IMAGEPROCESS_H_

#include <iostream>

#include <sys/time.h>

using namespace std;

class API_IMAGEPROCESS
{

///////////////////////////////////////////////////////////////////////////////////////
	#define MIN_IMAGE_WIDTH  	32 
	#define MIN_IMAGE_HEIGHT 	32

/***********************************public***********************************/
public:

	/// construct function 
    API_IMAGEPROCESS();
    
	/// distruct function
	~API_IMAGEPROCESS(void);

	//image process
	unsigned char * ImageCopy(const unsigned char *src, int width, int height, int nChannel);
	unsigned char * ImageROI(const unsigned char *src, int width, int height, int nChannel, int roi_x, int roi_y, int roi_w, int roi_h);
	unsigned char * ImageResize(const unsigned char *src, int width, int height, int nChannel, int scale_w, int scale_h);
	unsigned char * ImageRGB2Gray(const unsigned char *src, int width, int height, int nChannel);

	//image filter
	unsigned char * ImageMedianFilter(const unsigned char *src, int width, int height, int nChannel, int Filter_Kernal );

/***********************************private***********************************/
private:

	unsigned char GetMedianNum(unsigned char *bArray, int iFilterLen);

};



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

