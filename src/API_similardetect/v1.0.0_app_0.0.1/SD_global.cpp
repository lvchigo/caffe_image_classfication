#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <time.h>

#include <sys/types.h>
#include <dirent.h>
#include <sys/stat.h>
#include <pthread.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <list>
#include <map>
#include <algorithm>	
#include "TErrorCode.h"
#include "SD_global.h"
#include "ColorLayout.h"

using namespace std;
using namespace SIMILARDETECT_INTERNAL;

#define SD_GLOBAL_FEAT_LEN 36 
#define SD_PARTIAL_DIFF    2     //threshold for each dimension
#define SD_TOTAL_DIFF 	   8     //total threshold
#define BACKUP_INTERVAL    12    //back up timer interval, in hours
////////////////////////////////////////////////////////////////////////////
//configuration Macro definition
////////////////////////////////////////////////////////////////////////////

//-------------------------------------------------------------------------

namespace SD_GLOBAL_1_1_0 {

#ifndef MABS
//fast abs of integer number
#define MABS(x)                (((x)+((x)>>31))^((x)>>31))     
#endif //MABS

struct ClassInfo
{
	int ClassID;
	int SubClassID;
};

typedef struct tagSD_Global_Feature
{
	UInt64 ImageID;
	unsigned char m_feat[SD_GLOBAL_FEAT_LEN];
}SD_Global_Feature;

typedef struct tagSDCluster
{
	unsigned int nWritedIdx;
	vector<SD_Global_Feature> m_vecFeat;
}SDCluster;

typedef struct tagSDClass
{
	int                 nClassID;
    int 				nSubClassID;
	UInt64 			    m_lFileCounter;
	map< unsigned int, SDCluster > m_clusterMap;
}SDClass;

typedef struct tagSDParam
{
	string 				 m_strFolder;
	unsigned int 		 m_iSwapLimit;
	map< int,  SDClass > m_classpool;
}SDParam;
/////////////////////Global variable/////////////////////////////////////////////////
SDParam g_SDParam;
static pthread_mutex_t s_mutex = PTHREAD_MUTEX_INITIALIZER;
//////////////////////////////////////////////////////////////////////////////////////
void Version()
{
	cout<<"V1_0_0"<<endl;
}

unsigned int EncodeFeature(unsigned char *pFeat)
{
	int i, nDim;
	unsigned int code = 0;

	nDim = min(SD_GLOBAL_FEAT_LEN, 32);
	for (i = 0; i < nDim; i++) {
		code <<= 1;
		code |= (pFeat[i] >> 5);
	}
	return code;
}

inline int GetClassIDFromFilePath(char *filepath)
{
	int  ID = 0;
	int  atom =0;
	string tmpPath = filepath;
	for (int i=tmpPath.rfind('/')+1;i<tmpPath.rfind('.');i++)
	{
		atom = filepath[i] - '0';
		if (atom < 0 || atom >9)
			break;
		ID = ID * 10 + atom;
	}
	return ID;
}

bool IsSDdatFile(char *filepath)
{
	string tmpPath = filepath;
	int idx =  tmpPath.rfind('.');
	if (idx<0) return false;  //not find '.' in tmpPath
	if (0 == tmpPath.compare(idx, 6 ,".SDdat"))
		return true;
	else
		return false;
}

int ReadSDFeatures(int  SubClassID)
{
	int i;
	char szCurrFile[1024];
	FILE *fp = NULL;
	DIR *dir = NULL;
	struct dirent *s_dir;// struct stat file_stat;
	map < int, SDClass >::iterator itPool;
	map< unsigned int, SDCluster >::iterator itCluster;
	SD_Global_Feature *pFeat = NULL;
	unsigned int code;
	int nRet = TOK;

	itPool = g_SDParam.m_classpool.find(SubClassID);
	if (itPool == g_SDParam.m_classpool.end() )
	{
		printf("Error Raise from ReadSDFeatures: ClassID:%d is not in current classPool !\n", SubClassID);
		return TEC_BAD_STATE;
	}

	pFeat = new SD_Global_Feature;
	if (!pFeat) 
	{
		printf("Fail to alloc Memory for pFeat in ReadSDFeatures\n");
		return TEC_NO_MEMORY;
	}

	dir = opendir(g_SDParam.m_strFolder.c_str());
	if (NULL == dir)
	{
		printf("The specified folder %s doesn't exist, or have no access right to the folder\n", g_SDParam.m_strFolder.c_str());
		nRet = TEC_FILE_OPEN;
		goto ERR;
	}
	while ((s_dir = readdir(dir))!=NULL)
	{
		//if ((strcmp(s_dir->d_name, ".") == 0) || (strcmp(s_dir->d_name, "..")==0))
			//continue;
		if ( IsSDdatFile(s_dir->d_name) && GetClassIDFromFilePath(s_dir->d_name) == SubClassID )
		{
			sprintf(szCurrFile, "%s%s", g_SDParam.m_strFolder.c_str(), s_dir->d_name);
			fp = fopen(szCurrFile, "rb");
			if (!fp)
			{
				printf("Fail to read %s\n", szCurrFile);
				nRet = TEC_FILE_OPEN;
				goto ERR;
			}
			while ( 1 == fread(&(pFeat->ImageID), sizeof(pFeat->ImageID), 1, fp) && 
				    1 == fread(pFeat->m_feat, SD_GLOBAL_FEAT_LEN, 1, fp) )
			{
				code = EncodeFeature(pFeat->m_feat);
				itCluster = itPool->second.m_clusterMap.find(code);
				if (itCluster != itPool->second.m_clusterMap.end())
					itCluster->second.m_vecFeat.push_back(*pFeat);
				else
					itPool->second.m_clusterMap[code].m_vecFeat.push_back(*pFeat);
			}
			fclose(fp);
			fp = NULL;
		}
	}
	closedir(dir);
	dir = NULL;

	//set status for each cluster 
	itPool->second.m_lFileCounter  = 0;
	for (itCluster = itPool->second.m_clusterMap.begin(); itCluster != itPool->second.m_clusterMap.end(); itCluster++)
	{
		itCluster->second.nWritedIdx = itCluster->second.m_vecFeat.size();
		itPool->second.m_lFileCounter += itCluster->second.nWritedIdx ;
	}
ERR:
	if (fp) fclose(fp); 
	if (dir) closedir(dir);
	if (pFeat) delete pFeat;
	return nRet;
}

//按SubClassID来组织特征文件的文件名： SubClassID_Year_Month_day_section.SDdat			
int WriteSDFeatures(int SubClassID)   
{
	int i;
	char szFile[1024];
	FILE *fp = NULL;
	map < int , SDClass >::iterator itPool;
	map< unsigned int, SDCluster >::iterator itCluster;
	time_t timep;
	struct tm *p;

	time(&timep);
	p = localtime(&timep);
	//sprintf(szFile, "%s%d_%d_%d_%d_%d_%d.SDdat", g_SDParam.m_strFolder.c_str(),(1900+p->tm_year),
			//(1+p->tm_mon),p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
	sprintf(szFile, "%s%d_%d_%d_%d_%d.SDdat", g_SDParam.m_strFolder.c_str(), SubClassID, (1900+p->tm_year),
			(1+p->tm_mon),p->tm_mday, (p->tm_hour)/BACKUP_INTERVAL);

	itPool = g_SDParam.m_classpool.find(SubClassID);
	if (itPool == g_SDParam.m_classpool.end() )
	{
		printf("ClassID:%d is not in current classPool.\n",SubClassID);
		return TEC_BAD_STATE;
	}
	for (itCluster = itPool->second.m_clusterMap.begin(); itCluster != itPool->second.m_clusterMap.end(); itCluster++)
	{
		int nSize = itCluster->second.m_vecFeat.size();
		if (itCluster->second.nWritedIdx < nSize)
		{
			if (!fp)
			{
				fp = fopen(szFile, "ab+");
				if (!fp) return TEC_FILE_OPEN;
			}
			for (i = itCluster->second.nWritedIdx; i < nSize; i++) 
			{
				if (1 != fwrite(&(itCluster->second.m_vecFeat[i].ImageID), sizeof(UInt64), 1, fp))
				{
					fclose(fp);
					printf("Raise error when writing %s \n", szFile);
					return TEC_FILE_WRITE;
				}
				if (1 != fwrite(itCluster->second.m_vecFeat[i].m_feat, SD_GLOBAL_FEAT_LEN, 1, fp))
				{
					fclose(fp);
					printf("Raise error when writing %s \n", szFile);
					return TEC_FILE_WRITE;
				}
			}
			itCluster->second.nWritedIdx = nSize;
		}
	}
	if (fp) fclose(fp);
	return TOK;
}

inline void ClearSDParam()
{
	map< int, SDClass >::iterator itPool; 
	map< unsigned int, SDCluster >::iterator itCluster;

	for (itPool = g_SDParam.m_classpool.begin();itPool != g_SDParam.m_classpool.end();itPool++)
	{
		for (itCluster = itPool->second.m_clusterMap.begin(); itCluster != itPool->second.m_clusterMap.end(); itCluster++) {
			itCluster->second.m_vecFeat.clear();
		}
		itPool->second.m_clusterMap.clear();
	}
	g_SDParam.m_classpool.clear();
}

int LoadClassData(vector< ClassInfo >& classList)
{
	int i, nRet = TOK;
	vector< ClassInfo >::iterator itClassList;
	map< int, SDClass >::iterator itPool; 
	vector< int > removeClass;

	for(itClassList = classList.begin();itClassList != classList.end(); itClassList++)
	{
		if (itClassList->ClassID < 0 || itClassList->SubClassID < 0)
			return TEC_INVALID_PARAM;
	}

	pthread_mutex_lock(&s_mutex);
	for(itPool = g_SDParam.m_classpool.begin(); itPool != g_SDParam.m_classpool.end(); itPool++)
	{
		for(itClassList = classList.begin(); itClassList != classList.end(); itClassList++)
		{
			if (itPool->second.nClassID == itClassList->ClassID && itPool->second.nSubClassID == itClassList->SubClassID)
				break;
		}
		if (itClassList != classList.end())
			continue;
		else
			removeClass.push_back(itPool->first);
	}
	for (i=0;i<removeClass.size();i++)
	{
		nRet = WriteSDFeatures(removeClass[i]);
		if ( TOK != nRet)
			printf(" Fail to save Class:%d feature data into file when exchange classPool !!!!!!!!\n", removeClass[i]);
		g_SDParam.m_classpool.erase(removeClass[i]);
	}
	for(itClassList = classList.begin();itClassList != classList.end(); itClassList++)
	{
		itPool = g_SDParam.m_classpool.find(itClassList->SubClassID);
		if ( itPool != g_SDParam.m_classpool.end() )
			continue;
		else
		{
			g_SDParam.m_classpool[itClassList->SubClassID].nClassID = itClassList->ClassID;
			g_SDParam.m_classpool[itClassList->SubClassID].nSubClassID = itClassList->SubClassID;
			g_SDParam.m_classpool[itClassList->SubClassID].m_clusterMap.clear();
			nRet = ReadSDFeatures(itClassList->SubClassID);
			if ( TOK != nRet)
				printf(" Fail to load Class:%d feature data from file when exchange classPool !!!!!!!!\n", itClassList->SubClassID);
		}
	}
	pthread_mutex_unlock(&s_mutex);
	return nRet;
}

int Init(const char *path)
{
	if (!path )
		return TEC_INVALID_PARAM;

	int nRet = TOK;
	const int iSwapLimit = 200000;
	ClassInfo ci;
	vector < ClassInfo > classList;

	//Version();

	g_SDParam.m_iSwapLimit = iSwapLimit;
	g_SDParam.m_strFolder = path;
	if (g_SDParam.m_strFolder[g_SDParam.m_strFolder.length()-1] != '/')
		g_SDParam.m_strFolder.append(1,'/');
	g_SDParam.m_classpool.clear();

	ci.ClassID = 70020;
	ci.SubClassID = 70020;
	classList.push_back(ci);
	nRet = LoadClassData(classList);
	if (nRet != 0)
	{
	   cout<<"Fail to LoadClassData. Error code:"<< nRet << endl;
	   return TEC_INVALID_PARAM;
	}
	
	return TOK;
}

int Uninit()
{
	int nRet = TOK;
	map < int, SDClass >::iterator itPool;

	pthread_mutex_lock(&s_mutex);
	for (itPool = g_SDParam.m_classpool.begin(); itPool != g_SDParam.m_classpool.end(); itPool++)
	{
		nRet = WriteSDFeatures(itPool->first);
		if ( TOK != nRet)
			printf(" Fail to save Class:%d feature data into file in Uninit() !!!!!!!!\n", itPool->first);
	}
	ClearSDParam();
	pthread_mutex_unlock(&s_mutex);
	return nRet;
}

bool IsSimilar(unsigned char *query_feat, unsigned char *db_feat)
{
	int i, nTmp, nSum = 0;
	for (i = 0; i < SD_GLOBAL_FEAT_LEN; i++) {
		nTmp = query_feat[i];
		nTmp -= db_feat[i];
		nTmp = MABS(nTmp);
		if (nTmp > SD_PARTIAL_DIFF)
			return false;
		nSum += nTmp;
		if (nSum > SD_TOTAL_DIFF)
			return false;
	}
	return true;
}

int SimilarDetect(
			 unsigned char *pImage, 
			 int 			width, 
			 int 			height, 
			 int 			nChannel,
			 UInt64			ImageID,
			 int			&res_mode,
			 UInt64     	&res_iid )
{
	SD_Global_Feature *pFeat = NULL;
	map< int,SDClass >::iterator itPool;
	map<unsigned int, SDCluster>::iterator itCluster;
	unsigned int code;
	const int ClassID = 70020;
	const int SubClassID = 70020;
	
	if ( (!pImage) || (width<16) || (height<16) || (nChannel!=3) || (ImageID<0) )
		return TEC_INVALID_PARAM;

	pFeat = new SD_Global_Feature;
	if (!pFeat)
		return TEC_NO_MEMORY;

	pFeat->ImageID = ImageID;
	MultiBlock_LayoutExtractor(pImage, width, height, nChannel, pFeat->m_feat);

	code = EncodeFeature(pFeat->m_feat);
	res_mode = 0;
	res_iid = ImageID;

	pthread_mutex_lock(&s_mutex);
	itPool = g_SDParam.m_classpool.find(SubClassID);
	if (itPool == g_SDParam.m_classpool.end())
	{
		pthread_mutex_unlock(&s_mutex);
		if (pFeat) delete pFeat;
		return TEC_BAD_STATE;
	}
	itCluster = itPool->second.m_clusterMap.find(code);
	if (itCluster == itPool->second.m_clusterMap.end()) // not find the cluster code, create new cluster nod
	{	
		itPool->second.m_clusterMap[code].nWritedIdx = 0;
		itPool->second.m_clusterMap[code].m_vecFeat.clear();
		itPool->second.m_clusterMap[code].m_vecFeat.push_back(*pFeat);
		itPool->second.m_lFileCounter++;
	}
	else
	{
		vector<SD_Global_Feature>::iterator it;
		for (it = itCluster->second.m_vecFeat.begin();it != itCluster->second.m_vecFeat.end();it++)
		{
			if (IsSimilar(pFeat->m_feat, it->m_feat))
			{
				res_mode = 1;
				res_iid = it->ImageID;
				break;
			}
		}
		if (it == itCluster->second.m_vecFeat.end() )
		{
			itCluster->second.m_vecFeat.push_back(*pFeat);
			itPool->second.m_lFileCounter++;
		}
	}
	if ((itPool->second.m_lFileCounter % g_SDParam.m_iSwapLimit) == 0)
		WriteSDFeatures(itPool->first);
	pthread_mutex_unlock(&s_mutex);

	if (pFeat) delete pFeat;
	return TOK;
}

} //end of namespace 
