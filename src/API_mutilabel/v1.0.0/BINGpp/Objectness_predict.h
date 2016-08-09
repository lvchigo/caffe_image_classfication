/************************************************************************/
/* This source code is	free for both academic and industry use.         */
/* Some important information for better using the source code could be */
/* found in the project page: http://mmcheng.net/bing					*/
/************************************************************************/

#pragma once
#include "FilterBING.h"


class Objectness
{
public:
	// base for window size quantization, feature window size (W, W), and non-maximal suppress size NSS
	Objectness(double base = 2, int W = 8, int NSS = 2);
	~Objectness(void);

	// Load trained model. 
	int loadTrainedModelOnly(string modelName); // Return -1, 0, or 1 if partial, none, or all loaded

	// Get potential bounding boxes, each of which is represented by a Vec4i for (minX, minY, maxX, maxY).
	// The trained model should be prepared before calling this function: loadTrainedModel() or trainStageI() + trainStageII().
	// Use numDet to control the final number of proposed bounding boxes, and number of per size (scale and aspect ratio)
	void getObjBndBoxes(Mat &img3u, ValStructVec<float, Vec4i> &valBoxes, int numDetPerSize = 120);

	// Training and testing on the dataset
	double interUnio(const Vec4i &bb, const Vec4i &bbgt);
	
	enum {MAXBGR, HSV, G};

public: // For illustration etc.

	// Read matrix from binary file
	static bool matRead( const string& filename, Mat& M);
	static Mat aFilter(float delta, int sz);

private: // Parameters
	const double _base, _logBase; // base for window size quantization
	const int _W; // As described in the paper: #Size, Size(_W, _H) of feature window. 
	const int _NSS; // Size for non-maximal suppress
	const int _maxT, _minT, _numT; // The minimal and maximal dimensions of the template
	
	int _Clr;
	static const char* _clrName[3];
	
	string _modelName;

	vecI _svmSzIdxs; // Indexes of active size. It's equal to _svmFilters.size() and _svmReW1f.rows
	Mat _svmFilter; // Filters learned at stage I, each is a _H by _W CV_32F matrix
	FilterBING _bingF; // BING filter
	Mat _svmReW1f; // Re-weight parameters learned at stage II. 	

private: // Help functions

	bool filtersLoaded() {int n = _svmSzIdxs.size(); return n > 0 && _svmReW1f.size() == Size(2, n) && _svmFilter.size() == Size(_W, _W);}

	void predictBBoxSI(Mat &mag3u, ValStructVec<float, Vec4i> &valBoxes, vecI &sz, int NUM_WIN_PSZ = 100, bool fast = true);
	void predictBBoxSII(ValStructVec<float, Vec4i> &valBoxes, const vecI &sz);
	
	// Calculate the image gradient: center option as in VLFeat
	void gradientMag(CMat &imgBGR3u, Mat &mag1u);

	static void gradientRGB(CMat &bgr3u, Mat &mag1u);
	static void gradientGray(CMat &bgr3u, Mat &mag1u);
	static void gradientHSV(CMat &bgr3u, Mat &mag1u);
	static void gradientXY(CMat &x1i, CMat &y1i, Mat &mag1u);

	static inline int bgrMaxDist(const Vec3b &u, const Vec3b &v) { int b = abs(u[0] - v[0]), g = abs(u[1] - v[1]), r = abs(u[2] - v[2]); b = max(b,g); return max(b,r); }
	static inline int vecDist3b(const Vec3b &u, const Vec3b &v) {return abs(u[0]-v[0]) + abs(u[1]-v[1]) + abs(u[2]-v[2]);}

	//Non-maximal suppress
	static void nonMaxSup(CMat &matchCost1f, ValStructVec<float, Point> &matchCost, int NSS = 1, int maxPoint = 50, bool fast = true);

	int minIdx(Mat &input, int x1, int y1, int x2, int y2);
	int maxIdx(Mat &input, int x1, int y1, int x2, int y2);
};

