#pragma once
#include <vector>
#include "svm.h"

using namespace std;

class API_LIBSVM
{

/***********************************Config***********************************/
#define SVM_PREDICT_PROBABILITY 1					//predict_probability:1-probability estimates;

/***********************************public***********************************/
public:

	/// construct function 
    API_LIBSVM();
    
	/// distruct function
	~API_LIBSVM(void);

	/***********************************Init*************************************/
	int Init( const char* ModelFile );					//[In]:ModelFile

	/***********************************Predict**********************************/
	int Predict(
		vector< vector<float> > 			Feat,		//[In]:Feat
		vector< pair< int, float > >	 	&Res);		//[Out]:Res

	/***********************************Release**********************************/
	void Release();

/***********************************private***********************************/
private:

	struct svm_model* model;	//libsvm3.2.0

};

