////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Applicable File Name: SD_global.h (SD:SimilarDetect)
// Editor: xiaogao
//
// Copyright (c) IN Inc(2016-2017)
// 
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef _SD_GLOBAL_H_
#define _SD_GLOBAL_H_

#include <vector>

#if defined(_WIN32) && defined(SD_DLL_EXPORTS)
	#define SD_API __declspec(dllexport)
#else
	#define SD_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned long long UInt64;

namespace SD_GLOBAL_1_1_0{

/**********************************************************************************
 * Common API
***********************************************************************************/ 
/** Version
 * @brief : print the version info
 *
 * @return : void
 */
SD_API void Version();

/** Init
 * @brief : Initialization with global parameters 
 * @return : 0(TOK) if succeeds, otherwise Error code is returned. (lookup in TErrorCode.h)
 */
SD_API int Init( const char*  path );		//[In]load/save keyfile path;

/** SimilarDetect 
 * @brief : detect the input image for similar copy .
 * @return : 0(TOK) if succeeds, otherwise Error code is returned. (lookup in TErrorCode.h)
 */
SD_API int SimilarDetect(
			 unsigned char *pImage, 		//[In]input image data:w-h-c;
			 int 			width, 			//[In]input image width;
			 int 			height, 		//[In]input image height;
			 int 			nChannel,		//[In]input image channel;
			 UInt64			ImageID,		//[In]input image id;
			 int			&res_mode,		//[Out]:1-similar image,0-none;
			 UInt64     	&res_iid);		//[Out]:res_mode=1-similar image id,res_mode=0-input image id;

/** release_resource
 * @brief : release resource to free memory
 * @return : 0(TOK) if succeeds, otherwise Error code is returned. (lookup in TErrorCode.h)
 */
SD_API int Uninit();


} //end of namespace 

#ifdef __cplusplus
}
#endif

#endif 

