/*
 * =====================================================================================
 *
 *       filename:  API_linearsvm.h
 *
 *    description:  linearsvm interface 
 *
 *        version:  1.0
 *        created:  2016-01-23
 *       revision:  none
 *       compiler:  g++
 *
 *         author:  xiaogao
 *        company:  in66.com
 *
 *      copyright:  2016 itugo Inc. All Rights Reserved.
 *      
 * =====================================================================================
 */

#ifndef _API_LINEARSVM_H_
#define _API_LINEARSVM_H_

#include <vector>
#include "linear.h"

using namespace std;

class API_LINEARSVM
{

/***********************************Config***********************************/
#define SVM_PREDICT_PROBABILITY 1					//predict_probability:1-probability estimates;
#define SVM_PREDICT_TOPN 		3					//TOPN:3;


/***********************************public***********************************/
public:

	/// construct function 
    API_LINEARSVM();
    
	/// distruct function
	~API_LINEARSVM(void);

	/***********************************Init*************************************/
	int Init( const char* ModelFile );					//[In]:ModelFile

	/***********************************Predict**********************************/
	int Predict(
		vector< vector<float> > 			Feat,		//[In]:Feat
		vector< pair< int, float > >	 	&Res);		//[Out]:Res

	/***********************************Predict_mutilabel**********************************/
	int Predict_mutilabel(
		vector< vector<float> > 			Feat,		//[In]:Feat
		vector< pair< int, float > > 		&Res);		//[Out]:Res

	/***********************************Release**********************************/
	void Release();

/***********************************private***********************************/
private:

	struct model *linearModel;		//liblinear svm 1.98
	
};

#endif


