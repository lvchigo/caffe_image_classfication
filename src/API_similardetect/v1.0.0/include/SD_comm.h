////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Applicable File Name: SD_comm.h  (SimilarDetect Common type)
// Editor: Achates
//
// Copyright (c) Taotaosou Inc(2010-2011)
// 
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef _SD_BASE_H_
#define _SD_BASE_H_

typedef unsigned long long UInt64;

typedef enum tagSD_ResultType {
	SD_eUnique = 0,
	SD_eSame = 1,
	SD_eSimiliar = 2
}SD_RESTYPE;

typedef struct tagResultRes{
	SD_RESTYPE 	sMode;
	UInt64      id;
}SD_RES;

#endif 
