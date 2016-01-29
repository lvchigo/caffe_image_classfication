#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

#include "API_linearsvm.h"
#include "linear.h"
#include <TErrorCode.h>
#include "plog/Log.h"


using namespace std;

/***********************************Init*************************************/
/// construct function 
API_LINEARSVM::API_LINEARSVM()
{
}

/// destruct function 
API_LINEARSVM::~API_LINEARSVM(void)
{
}

static bool ImgSortComp(const pair <int, double> elem1, const pair <int, double> elem2)
{
	return (elem1.second > elem2.second);
}

static bool ImgSortComp2(const double elem1, const double elem2)
{
	return (elem1 > elem2);
}


/***********************************Init*************************************/
int API_LINEARSVM::Init(const char* featPath)
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

	if( ( linearModel = load_model(featPath) ) == 0 )
	{
		printf("can't open model file %s\n", featPath );
		return TEC_INVALID_PARAM;
	}

	if( SVM_PREDICT_PROBABILITY )
	{
		if(check_probability_model(linearModel)==0)
		{
			printf("Model does not support probabiliy estimates!!\n");
			return TEC_INVALID_PARAM;
		}
	}
	else
	{
		if(check_probability_model(linearModel)!=0)
			printf("Model supports probability estimates, but disabled in prediction.\n");
	}

	return TOK;
}

/***********************************Predict**********************************/
int API_LINEARSVM::Predict(
	vector< vector<float> > 			Feat,					//[In]:Feat
	vector< pair< int, float > > 		&Res)					//[Out]:Res
{
	if ( Feat.size()<1 )
		return TEC_INVALID_PARAM;

	Res.clear();
	int i,j,n,nr_class,nr_feature,nRet = 0;
	double predict_label = -1;
	vector<double> tmpEstimates;

	//svm_type = svm_get_svm_type(linearModel);
	nr_class = get_nr_class(linearModel);
	nr_feature = get_nr_feature(linearModel);
	//printf("GetSVMPredict[0]:svm_type-%d,nr_class-%d.\n",svm_type,nr_class);
		
	if( linearModel->bias >= 0 )
		n = nr_feature+1;
	else
		n = nr_feature;

	for( i=0;i<Feat.size();i++ )
	{
		if ( Feat[i].size() != nr_feature )
			return TEC_INVALID_PARAM;
		
		tmpEstimates.clear();
		struct feature_node *svm_node = (struct feature_node *) malloc((n+1)*sizeof(struct feature_node));
		double *prob_estimates = (double *) malloc(nr_class*sizeof(double));
		
		/***********************************data change**********************************/
		for( j=0;j<Feat[i].size();j++ )
		{
			svm_node[j].index = j+1;	//start from 1
			svm_node[j].value = Feat[i][j];
		}
		
		if( linearModel->bias >= 0 )	//add bias info
		{
			svm_node[nr_feature].index = n;
			svm_node[nr_feature].value = linearModel->bias;
		}
		svm_node[n].index = -1;	//end svm_node

		/***********************************Predict**********************************/
		if ( SVM_PREDICT_PROBABILITY )
		{
			predict_label = predict_probability(linearModel,svm_node,prob_estimates);
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
			predict_label = predict(linearModel,svm_node);
			Res.push_back( std::make_pair( int(predict_label), 1.0 ) );
		}

		free(svm_node);
		free(prob_estimates);
	}
	
	return nRet;
}

/***********************************Predict_mutilabel**********************************/
int API_LINEARSVM::Predict_mutilabel(
	vector< vector<float> > 			Feat,					//[In]:Feat
	vector< pair< int, float > > 		&Res)					//[Out]:Res
{
	if ( Feat.size()<1 )
		return TEC_INVALID_PARAM;

	Res.clear();
	int i,j,m,n,nr_class,nr_feature,numRes,nRet = 0;
	double predict_label = -1;
	vector< pair< int ,double > > tmpEstimates;

	//svm_type = svm_get_svm_type(linearModel);
	nr_class = get_nr_class(linearModel);
	nr_feature = get_nr_feature(linearModel);
	//printf("GetSVMPredict[0]:svm_type-%d,nr_class-%d.\n",svm_type,nr_class);
		
	if( linearModel->bias >= 0 )
		n = nr_feature+1;
	else
		n = nr_feature;

	for( i=0;i<Feat.size();i++ )
	{
		if ( Feat[i].size() != nr_feature )
			return TEC_INVALID_PARAM;
		
		tmpEstimates.clear();
		struct feature_node *svm_node = (struct feature_node *) malloc((n+1)*sizeof(struct feature_node));
		double *prob_estimates = (double *) malloc(nr_class*sizeof(double));
		
		/***********************************data change**********************************/
		for( j=0;j<Feat[i].size();j++ )
		{
			svm_node[j].index = j+1;	//start from 1
			svm_node[j].value = Feat[i][j];
		}
		
		if( linearModel->bias >= 0 )	//add bias info
		{
			svm_node[nr_feature].index = n;
			svm_node[nr_feature].value = linearModel->bias;
		}
		svm_node[n].index = -1;	//end svm_node

		/***********************************Predict**********************************/
		if ( SVM_PREDICT_PROBABILITY )
		{
			predict_label = predict_probability_muti(linearModel,svm_node,prob_estimates,tmpEstimates);

			//check data
/*			printf("In:	Feat[%d]:predict_label-%.4f,score:",i,predict_label);
			for(j=0;j<tmpEstimates.size();j++)
			{
				printf("%d-%.4f ", tmpEstimates[j].first, tmpEstimates[j].second);
			}
			printf("\n");

			printf("In:	Feat[%d]:predict_label-%.4f,score:",i,predict_label);
			for(j=0;j<nr_class;j++)
			{
				printf("%d-%.4f ", j, prob_estimates[j]);
			}
			printf("\n");*/

			//sort label result
			sort(tmpEstimates.begin(), tmpEstimates.end(),ImgSortComp);

			numRes = (tmpEstimates.size()>SVM_PREDICT_TOPN)?SVM_PREDICT_TOPN:tmpEstimates.size();
			for (m=0;m<numRes;m++)
			{
				if ( ( predict_label < 0 ) || ( tmpEstimates[0].first < 0 ) )
				{
					LOOGE<<"[SVM Err]predict_label:"<<predict_label<<"Estimates[0].first"<<tmpEstimates[0].first;
				}
				
				//get score
				if ( predict_label == tmpEstimates[0].first )
				{
					Res.push_back( std::make_pair( tmpEstimates[m].first, float(tmpEstimates[m].second) ) );
					//printf("Res[%d]:predict_label-%d,score-%.4f\n",m,tmpEstimates[m].first,tmpEstimates[m].second);
				}
				else if ( predict_label == tmpEstimates[0].first-1 )
				{
					Res.push_back( std::make_pair( tmpEstimates[m].first-1, float(tmpEstimates[m].second) ) );
					//printf("Res[%d]:predict_label-%d,score-%.4f\n",m,tmpEstimates[m].first-1,tmpEstimates[m].second);
				}
				else
				{
					LOOGE<<"[SVM Err]predict_label:"<<predict_label<<",Estimates[0].first:"<<tmpEstimates[0].first;
				}
			}
		}
		else
		{
			predict_label = predict(linearModel,svm_node);
			Res.push_back( std::make_pair( int(predict_label), 1.0 ) );
		}

		free(svm_node);
		free(prob_estimates);
	}
	
	return nRet;
}


/***********************************Release**********************************/
void API_LINEARSVM::Release()
{
	//liblinear svm 1.98
	free_and_destroy_model(&linearModel);
}



