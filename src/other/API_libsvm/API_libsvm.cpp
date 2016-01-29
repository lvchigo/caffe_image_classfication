#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

#include "API_libsvm.h"
#include "svm.h"
#include <TErrorCode.h>

using namespace std;

/***********************************Init*************************************/
/// construct function 
API_LIBSVM::API_LIBSVM()
{
}

/// destruct function 
API_LIBSVM::~API_LIBSVM(void)
{
}

static bool ImgSortComp(const pair <int, float> elem1, const pair <int, float> elem2)
{
	return (elem1.second > elem2.second);
}

static bool ImgSortComp2(const double elem1, const double elem2)
{
	return (elem1 > elem2);
}


/***********************************Init*************************************/
int API_LIBSVM::Init(const char* featPath)
{
	if(featPath == NULL)
	{
		printf("feat file is null!!\n");
		return TEC_INVALID_PARAM;
	}
	if(sizeof(char)*sizeof(featPath) > 1024)
	{
		printf("feat path is too long!!\n");
		return TEC_INVALID_PARAM;
	}

	if( ( model = svm_load_model(featPath) ) == 0 )
	{
		printf("can't open model file %s\n", featPath );
		return TEC_INVALID_PARAM;
	}

	if( SVM_PREDICT_PROBABILITY )
	{
		if(svm_check_probability_model(model)==0)
		{
			printf("Model does not support probabiliy estimates!!\n");
			return TEC_INVALID_PARAM;
		}
	}
	else
	{
		if(svm_check_probability_model(model)!=0)
			printf("Model supports probability estimates, but disabled in prediction.\n");
	}

	return TOK;
}

/***********************************Predict**********************************/
int API_LIBSVM::Predict(
	vector< vector<float> > 			Feat,					//[In]:Feat
	vector< pair< int, float > > 		&Res)					//[Out]:Res
{
	if ( Feat.size()<1 )
		return TEC_INVALID_PARAM;

	Res.clear();
	int i,j,svm_type,nr_class,nRet = 0;
	double predict_label = -1;
	vector<double> tmpEstimates;

	svm_type = svm_get_svm_type(model);
	nr_class = svm_get_nr_class(model);
	//printf("GetSVMPredict[0]:svm_type-%d,nr_class-%d.\n",svm_type,nr_class);

	for( i=0;i<Feat.size();i++ )
	{
		tmpEstimates.clear();
		struct svm_node *svm_node = (struct svm_node *) malloc((Feat[i].size()+1)*sizeof(struct svm_node));
		double *prob_estimates = (double *) malloc(nr_class*sizeof(double));
		
		/***********************************data change**********************************/
		for( j=0;j<Feat[i].size();j++ )
		{
			svm_node[j].index = j+1;	//start from 1
			svm_node[j].value = Feat[i][j];
		}
		svm_node[Feat[i].size()].index = -1;	//end svm_node

		/***********************************Predict**********************************/
		if ( SVM_PREDICT_PROBABILITY && (svm_type==C_SVC || svm_type==NU_SVC))
		{
			predict_label = svm_predict_probability(model,svm_node,prob_estimates);
			//printf("In:	Feat[%d]:predict_label-%.4f,score:",i,predict_label);
			for(j=0;j<nr_class;j++)
			{
				tmpEstimates.push_back( prob_estimates[j] );
				//printf("%.4f ",prob_estimates[j]);
			}
			//printf("\n");

			//sort label result
			sort(tmpEstimates.begin(), tmpEstimates.end(),ImgSortComp2);

			//get max score
			Res.push_back( std::make_pair( int(predict_label), float(tmpEstimates[0]) ) );
			//printf("Out:	Feat[%d]:predict_label-%.4f,score-%.4f\n",i,predict_label,tmpEstimates[0]);
		}
		else
		{
			predict_label = svm_predict(model,svm_node);
			Res.push_back( std::make_pair( int(predict_label), 1.0 ) );
		}

		free(svm_node);
		free(prob_estimates);
	}
	
	return nRet;
}

/***********************************Release**********************************/
void API_LIBSVM::Release()
{
	//libsvm3.2.0
	svm_free_and_destroy_model(&model);
}



