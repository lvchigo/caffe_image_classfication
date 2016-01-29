////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Applicable File Name: SD_global.h (SD:SimilarDetect)
// Editor: Achates
//
// Copyright (c) Taotaosou Inc(2010-2011)
// 
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef _SD_GLOBAL_H_
#define _SD_GLOBAL_H_

#include "cv.h"
#include "SD_comm.h"
#include <vector>

#if defined(_WIN32) && defined(SD_DLL_EXPORTS)
	#define SD_API __declspec(dllexport)
#else
	#define SD_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

namespace SD_GLOBAL_1_1_0{

struct ClassInfo
{
	int ClassID;
	int SubClassID;
};

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
 *
 * @param path :[IN] the path of the CDdat file
 * @param iSwapLimit :[IN] Limit number of images swapped into disc file.
 *
 * @return : 0(TOK) if succeeds, otherwise Error code is returned. (lookup in TErrorCode.h)
 */
SD_API int Init( const char*  path , int iSwapLimit = 200000);

/** LoadClassData
 * @brief : Load Class feature data by specified Class ID 
 *
 * @param classList :[IN] the vector of ClassInfo to be loaded.
 *
 * @return : 0(TOK) if succeeds, otherwise Error code is returned. (lookup in TErrorCode.h)
 */
SD_API int LoadClassData(std::vector< ClassInfo >& classList);

/** SimilarDetect 
 * @brief : detect the input image for similar copy .
 *
 * @param pImage : [IN] The pointer to image. the image must be color image (3-Channels).
 * @param ImageID: [IN] The Image ID.
 * @param ClassID : [IN] The Class ID of image.
 * @param SubClassID : [IN] The subClass ID of image.
 * @param BinSaveFeat : [IN] Save Feat(1) or Not(0).
 * 
 * @return : 0(TOK) if succeeds, otherwise Error code is returned. (lookup in TErrorCode.h)
 */
SD_API int SimilarDetect(
			 IplImage*   	pImage, 
			 UInt64			ImageID,
			 int 	   	 	ClassID, 
			 int 		 	SubClassID,
	 		 SD_RES*     	result,
	 		 int 			BinSaveFeat = 1);

/** EraseClassData    //
 * @brief : Erase Class feature data in memory and delete the feature files in disc for the specified Class ID 
 *
 * @param classList :[IN] the vector of ClassInfo to be erased.
 *
 * @return : 0(TOK) if succeeds, otherwise Error code is returned. (lookup in TErrorCode.h)
 */			 
SD_API int EraseClassData(std::vector< ClassInfo >& classList);

/** release_resource
 * @brief : release resource to free memory
 *
 * @return : 0(TOK) if succeeds, otherwise Error code is returned. (lookup in TErrorCode.h)
 */
SD_API int Uninit();


} //end of namespace 

#ifdef __cplusplus
}
#endif

#endif 

