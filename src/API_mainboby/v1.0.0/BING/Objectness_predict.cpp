#include "kyheader.h"
#include "Objectness_predict.h"
#include "CmShow.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
void print_null(const char *s) {}
const char* Objectness::_clrName[3] = {"MAXBGR", "HSV", "I"};
const int CN = 21; // Color Number
const char* COLORs[CN] = {"'k'", "'b'", "'g'", "'r'", "'c'", "'m'", "'y'",
    "':k'", "':b'", "':g'", "':r'", "':c'", "':m'", "':y'",
    "'--k'", "'--b'", "'--g'", "'--r'", "'--c'", "'--m'", "'--y'"
};


// base for window size quantization, R orientation channels, and feature window size (_W, _W)
Objectness::Objectness(double base, int W, int NSS)
	: _base(base)
	, _W(W)
	, _NSS(NSS)
	, _logBase(log(_base))
	, _minT(cvCeil(log(10.)/_logBase))
	, _maxT(cvCeil(log(500.)/_logBase))
	, _numT(_maxT - _minT + 1)
	, _Clr(MAXBGR)
{
}


Objectness::~Objectness(void)
{
}

int Objectness::loadTrainedModelOnly(string modelName) // Return -1, 0, or 1 if partial, none, or all loaded
{
	if (modelName.size() == 0)
		modelName = _modelName;
	CStr s1 = modelName + ".wS1", s2 = modelName + ".wS2", sI = modelName + ".idx";
	Mat filters1f, reW1f, idx1i, show3u;
	if (!matRead(s1, filters1f) || !matRead(sI, idx1i)){
		printf("Can't load model: %s or %s\n", _S(s1), _S(sI));
		return 0;
	}

	//filters1f = aFilter(0.8f, 8);
	//normalize(filters1f, filters1f, p, 1, NORM_MINMAX);

	normalize(filters1f, show3u, 1, 255, NORM_MINMAX, CV_8U);
    _tigF.update(filters1f);
    _tigF.reconstruct(filters1f);

	_svmSzIdxs = idx1i;
	CV_Assert(_svmSzIdxs.size() > 1 && filters1f.size() == Size(_W, _W) && filters1f.type() == CV_32F);
	_svmFilter = filters1f;

	if (!matRead(s2, _svmReW1f) || _svmReW1f.size() != Size(2, _svmSzIdxs.size())){
		_svmReW1f = Mat();
		return -1;
	}
	return 1;
}


void Objectness::predictBBoxSI(CMat &img3u, ValStructVec<float, Vec4i> &valBoxes, vecI &sz, int NUM_WIN_PSZ, bool fast)
{
	const int numSz = _svmSzIdxs.size();
	const int imgW = img3u.cols, imgH = img3u.rows;
	valBoxes.reserve(10000);
	sz.clear(); sz.reserve(10000);
	for (int ir = numSz - 1; ir >= 0; ir--){
		int r = _svmSzIdxs[ir]; 
		int height = cvRound(pow(_base, r/_numT + _minT)), width = cvRound(pow(_base, r%_numT + _minT));
		if (height > imgH * _base || width > imgW * _base)
			continue;

		height = min(height, imgH), width = min(width, imgW);
		Mat im3u, matchCost1f, mag1u;
		resize(img3u, im3u, Size(cvRound(_W*imgW*1.0/width), cvRound(_W*imgH*1.0/height)));
		gradientMag(im3u, mag1u);
		
		//imwrite(_voc.localDir + format("%d.png", r), mag1u);	
        //Mat mag1f;
        //mag1u.convertTo(mag1f, CV_32F);
        //matchTemplate(mag1f, _svmFilter, matchCost1f, CV_TM_CCORR);
		
        matchCost1f = _tigF.matchTemplate(mag1u);

		ValStructVec<float, Point> matchCost;
		nonMaxSup(matchCost1f, matchCost, _NSS, NUM_WIN_PSZ, fast);

		// Find true locations and match values
		double ratioX = width/_W, ratioY = height/_W;
		int iMax = min(matchCost.size(), NUM_WIN_PSZ);
		for (int i = 0; i < iMax; i++){
			float mVal = matchCost(i);
			Point pnt = matchCost[i];
			Vec4i box(cvRound(pnt.x * ratioX), cvRound(pnt.y*ratioY));
			box[2] = cvRound(min(box[0] + width, imgW));
			box[3] = cvRound(min(box[1] + height, imgH));
			box[0] ++;
			box[1] ++;
			valBoxes.pushBack(mVal, box); 
			sz.push_back(ir);
		}
	}
	//exit(0);
}

void Objectness::predictBBoxSII(ValStructVec<float, Vec4i> &valBoxes, const vecI &sz)
{
	int numI = valBoxes.size();
	for (int i = 0; i < numI; i++){
		const float* svmIIw = _svmReW1f.ptr<float>(sz[i]);
		valBoxes(i) = valBoxes(i) * svmIIw[0] + svmIIw[1]; 
	}
	valBoxes.sort();
}

// Get potential bounding boxes, each of which is represented by a Vec4i for (minX, minY, maxX, maxY).
// The trained model should be prepared before calling this function: loadTrainedModel() or trainStageI() + trainStageII().
// Use numDet to control the final number of proposed bounding boxes, and number of per size (scale and aspect ratio)
void Objectness::getObjBndBoxes(CMat &img3u, ValStructVec<float, Vec4i> &valBoxes, int numDetPerSize)
{
	CV_Assert_(filtersLoaded() , ("SVM filters should be initialized before getting object proposals\n"));
	vecI sz;
	predictBBoxSI(img3u, valBoxes, sz, numDetPerSize, false);
	predictBBoxSII(valBoxes, sz);
	return;
}

void Objectness::nonMaxSup(CMat &matchCost1f, ValStructVec<float, Point> &matchCost, int NSS, int maxPoint, bool fast)
{
	const int _h = matchCost1f.rows, _w = matchCost1f.cols;
	Mat isMax1u = Mat::ones(_h, _w, CV_8U), costSmooth1f;
	ValStructVec<float, Point> valPnt;
	matchCost.reserve(_h * _w);
	valPnt.reserve(_h * _w);
	if (fast){
		blur(matchCost1f, costSmooth1f, Size(3, 3));
		for (int r = 0; r < _h; r++){
			const float* d = matchCost1f.ptr<float>(r);
			const float* ds = costSmooth1f.ptr<float>(r);
			for (int c = 0; c < _w; c++)
				if (d[c] >= ds[c])
					valPnt.pushBack(d[c], Point(c, r));
		}
	}
	else{
		for (int r = 0; r < _h; r++){
			const float* d = matchCost1f.ptr<float>(r);
			for (int c = 0; c < _w; c++)
				valPnt.pushBack(d[c], Point(c, r));
		}
	}

	valPnt.sort();
	for (int i = 0; i < valPnt.size(); i++){
		Point &pnt = valPnt[i];
		if (isMax1u.at<byte>(pnt)){
			matchCost.pushBack(valPnt(i), pnt);
			for (int dy = -NSS; dy <= NSS; dy++) for (int dx = -NSS; dx <= NSS; dx++){
				Point neighbor = pnt + Point(dx, dy);
				if (!CHK_IND(neighbor))
					continue;
				isMax1u.at<byte>(neighbor) = false;
			}
		}
		if (matchCost.size() >= maxPoint)
			return;
	}
}

void Objectness::gradientMag(CMat &imgBGR3u, Mat &mag1u)
{
	switch (_Clr){
	case MAXBGR:
		gradientRGB(imgBGR3u, mag1u); break;
	case G:		
		gradientGray(imgBGR3u, mag1u); break;
	case HSV:
		gradientHSV(imgBGR3u, mag1u); break;
	default:
		printf("Error: not recognized color space\n");
	}
}

void Objectness::gradientRGB(CMat &bgr3u, Mat &mag1u)
{
	const int H = bgr3u.rows, W = bgr3u.cols;
	Mat Ix(H, W, CV_32S), Iy(H, W, CV_32S);

	// Left/right most column Ix
	for (int y = 0; y < H; y++){
		Ix.at<int>(y, 0) = bgrMaxDist(bgr3u.at<Vec3b>(y, 1), bgr3u.at<Vec3b>(y, 0))*2;
		Ix.at<int>(y, W-1) = bgrMaxDist(bgr3u.at<Vec3b>(y, W-1), bgr3u.at<Vec3b>(y, W-2))*2;
	}

	// Top/bottom most column Iy
	for (int x = 0; x < W; x++)	{
		Iy.at<int>(0, x) = bgrMaxDist(bgr3u.at<Vec3b>(1, x), bgr3u.at<Vec3b>(0, x))*2;
		Iy.at<int>(H-1, x) = bgrMaxDist(bgr3u.at<Vec3b>(H-1, x), bgr3u.at<Vec3b>(H-2, x))*2; 
	}

	// Find the gradient for inner regions
	for (int y = 0; y < H; y++){
		const Vec3b *dataP = bgr3u.ptr<Vec3b>(y);
		for (int x = 2; x < W; x++)
			Ix.at<int>(y, x-1) = bgrMaxDist(dataP[x-2], dataP[x]); //  bgr3u.at<Vec3b>(y, x+1), bgr3u.at<Vec3b>(y, x-1));
	}
	for (int y = 1; y < H-1; y++){
		const Vec3b *tP = bgr3u.ptr<Vec3b>(y-1);
		const Vec3b *bP = bgr3u.ptr<Vec3b>(y+1);
		for (int x = 0; x < W; x++)
			Iy.at<int>(y, x) = bgrMaxDist(tP[x], bP[x]);
	}
	gradientXY(Ix, Iy, mag1u);
}

void Objectness::gradientGray(CMat &bgr3u, Mat &mag1u)
{
	Mat g1u;
	cvtColor(bgr3u, g1u, CV_BGR2GRAY); 
	const int H = g1u.rows, W = g1u.cols;
	Mat Ix(H, W, CV_32S), Iy(H, W, CV_32S);

	// Left/right most column Ix
	for (int y = 0; y < H; y++){
		Ix.at<int>(y, 0) = abs(g1u.at<byte>(y, 1) - g1u.at<byte>(y, 0)) * 2;
		Ix.at<int>(y, W-1) = abs(g1u.at<byte>(y, W-1) - g1u.at<byte>(y, W-2)) * 2;
	}

	// Top/bottom most column Iy
	for (int x = 0; x < W; x++)	{
		Iy.at<int>(0, x) = abs(g1u.at<byte>(1, x) - g1u.at<byte>(0, x)) * 2;
		Iy.at<int>(H-1, x) = abs(g1u.at<byte>(H-1, x) - g1u.at<byte>(H-2, x)) * 2; 
	}

	// Find the gradient for inner regions
	for (int y = 0; y < H; y++)
		for (int x = 1; x < W-1; x++)
			Ix.at<int>(y, x) = abs(g1u.at<byte>(y, x+1) - g1u.at<byte>(y, x-1));
	for (int y = 1; y < H-1; y++)
		for (int x = 0; x < W; x++)
			Iy.at<int>(y, x) = abs(g1u.at<byte>(y+1, x) - g1u.at<byte>(y-1, x));

	gradientXY(Ix, Iy, mag1u);
}


void Objectness::gradientHSV(CMat &bgr3u, Mat &mag1u)
{
	Mat hsv3u;
	cvtColor(bgr3u, hsv3u, CV_BGR2HSV);
	const int H = hsv3u.rows, W = hsv3u.cols;
	Mat Ix(H, W, CV_32S), Iy(H, W, CV_32S);

	// Left/right most column Ix
	for (int y = 0; y < H; y++){
		Ix.at<int>(y, 0) = vecDist3b(hsv3u.at<Vec3b>(y, 1), hsv3u.at<Vec3b>(y, 0));
		Ix.at<int>(y, W-1) = vecDist3b(hsv3u.at<Vec3b>(y, W-1), hsv3u.at<Vec3b>(y, W-2));
	}

	// Top/bottom most column Iy
	for (int x = 0; x < W; x++)	{
		Iy.at<int>(0, x) = vecDist3b(hsv3u.at<Vec3b>(1, x), hsv3u.at<Vec3b>(0, x));
		Iy.at<int>(H-1, x) = vecDist3b(hsv3u.at<Vec3b>(H-1, x), hsv3u.at<Vec3b>(H-2, x)); 
	}

	// Find the gradient for inner regions
	for (int y = 0; y < H; y++)
		for (int x = 1; x < W-1; x++)
			Ix.at<int>(y, x) = vecDist3b(hsv3u.at<Vec3b>(y, x+1), hsv3u.at<Vec3b>(y, x-1))/2;
	for (int y = 1; y < H-1; y++)
		for (int x = 0; x < W; x++)
			Iy.at<int>(y, x) = vecDist3b(hsv3u.at<Vec3b>(y+1, x), hsv3u.at<Vec3b>(y-1, x))/2;

	gradientXY(Ix, Iy, mag1u);
}

void Objectness::gradientXY(CMat &x1i, CMat &y1i, Mat &mag1u)
{
	const int H = x1i.rows, W = x1i.cols;
	mag1u.create(H, W, CV_8U);
	for (int r = 0; r < H; r++){
		const int *x = x1i.ptr<int>(r), *y = y1i.ptr<int>(r);
		byte* m = mag1u.ptr<byte>(r);
		for (int c = 0; c < W; c++)
			m[c] = min(x[c] + y[c], 255);   //((int)sqrt(sqr(x[c]) + sqr(y[c])), 255);
	}
}

Mat Objectness::getFeature(CMat &img3u, const Vec4i &bb)
{
	int x = bb[0] - 1, y = bb[1] - 1;
	Rect reg(x, y, bb[2] -  x, bb[3] - y);
	Mat subImg3u, mag1f, mag1u;
	resize(img3u(reg), subImg3u, Size(_W, _W));
	gradientMag(subImg3u, mag1u);
	mag1u.convertTo(mag1f, CV_32F);
	return mag1f;
}

int Objectness::gtBndBoxSampling(const Vec4i &bbgt, vector<Vec4i> &samples, vecI &bbR)
{
	double wVal = bbgt[2] - bbgt[0] + 1, hVal = (bbgt[3] - bbgt[1]) + 1;
	wVal = log(wVal)/_logBase, hVal = log(hVal)/_logBase;
	int wMin = max((int)(wVal - 0.5), _minT), wMax = min((int)(wVal + 1.5), _maxT);
	int hMin = max((int)(hVal - 0.5), _minT), hMax = min((int)(hVal + 1.5), _maxT);
	for (int h = hMin; h <= hMax; h++) for (int w = wMin; w <= wMax; w++){
		int wT = tLen(w) - 1, hT = tLen(h) - 1;
		Vec4i bb(bbgt[0], bbgt[1], bbgt[0] + wT, bbgt[1] + hT);
		if (DataSetVOC::interUnio(bb, bbgt) >= 0.5){
			samples.push_back(bb);
			bbR.push_back(sz2idx(w, h));
			//if (bbgt[3] > hT){
			//	bb = Vec4i(bbgt[0], bbgt[3] - hT, bbgt[0] + wT, bbgt[3]);
			//	CV_Assert(DataSetVOC::interUnio(bb, bbgt) >= 0.5);
			//	samples.push_back(bb);
			//	bbR.push_back(sz2idx(w, h));
			//}
			//if (bbgt[2] > wT){
			//	bb = Vec4i(bbgt[2] - wT, bbgt[1], bbgt[2], bbgt[1] + hT);
			//	CV_Assert(DataSetVOC::interUnio(bb, bbgt) >= 0.5);
			//	samples.push_back(bb);
			//	bbR.push_back(sz2idx(w, h));
			//}
			//if (bbgt[2] > wT && bbgt[3] > hT){
			//	bb = Vec4i(bbgt[2] - wT, bbgt[3] - hT, bbgt[2], bbgt[3]);
			//	CV_Assert(DataSetVOC::interUnio(bb, bbgt) >= 0.5);
			//	samples.push_back(bb);
			//	bbR.push_back(sz2idx(w, h));
			//}
		}
	}
	return samples.size();
}

void Objectness::meanStdDev(CMat &data1f, Mat &mean1f, Mat &stdDev1f)
{
	const int DIM = data1f.cols, NUM = data1f.rows;
	mean1f = Mat::zeros(1, DIM, CV_32F), stdDev1f = Mat::zeros(1, DIM, CV_32F);
	for (int i = 0; i < NUM; i++)
		mean1f += data1f.row(i);
	mean1f /= NUM;
	for (int i = 0; i < NUM; i++){
		Mat tmp;
		pow(data1f.row(i) - mean1f, 2, tmp);
		stdDev1f += tmp;
	}
	pow(stdDev1f/NUM, 0.5, stdDev1f);
}

vecD Objectness::getVector(const Mat &_t1f)
{
	Mat t1f;
	_t1f.convertTo(t1f, CV_64F);
	return (vecD)(t1f.reshape(1, 1));
}

// Get potential bounding boxes for all test images
void Objectness::Predict( string imgfile, int numDetPerSize)
{
	const static Scalar colors[] =  { CV_RGB(0,0,255),
        CV_RGB(0,128,255),
        CV_RGB(0,255,255),
        CV_RGB(0,255,0),
        CV_RGB(255,128,0),
        CV_RGB(255,255,0),
        CV_RGB(255,0,0),
        CV_RGB(255,0,255)} ;
	
	int i,topN=5;
	char szImgPath[256];
	loadTrainedModelOnly("/home/chigo/working/research/Bing/Objectness-master/VOC2007/Results/ObjNessB2W8MAXBGR");

	ValStructVec<float, Vec4i> boxTests;

	IplImage *img = cvLoadImage(imgfile.c_str());
	if(!img || (img->width<32) || (img->height<32) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
	{	
		cout<<"img err" << endl;

		cvReleaseImage(&img);img = 0;
		return;
	}	
	
	Mat matImg(img);
	boxTests.reserve(10000);

	getObjBndBoxes(matImg, boxTests, numDetPerSize);

	/************************getRandomID*****************************/
	topN = (topN>boxTests.size())?boxTests.size():topN;
	for(i=0;i<topN;i++) 
	{	
		Scalar color = colors[i%8];
		rectangle( matImg, cvPoint(boxTests[i][0], boxTests[i][1]),
                   cvPoint(boxTests[i][2], boxTests[i][3]), color, 3, 8, 0);
		sprintf(szImgPath, "1.jpg");
		imwrite( szImgPath, matImg );
	}	
	
	/*********************************Release*************************************/
	cvReleaseImage(&img);img = 0;
}

void Objectness::PrintVector(FILE *f, const vecD &v, CStr &name)
{
	fprintf(f, "%s = [", name.c_str());
	for (size_t i = 0; i < v.size(); i++)
		fprintf(f, "%g ", v[i]);
	fprintf(f, "];\n");
}

// Write matrix to binary file
bool Objectness::matWrite(CStr& filename, CMat& _M){
	Mat M;
	_M.copyTo(M);
	FILE* file = fopen(_S(filename), "wb");
	if (file == NULL || M.empty())
		return false;
	fwrite("CmMat", sizeof(char), 5, file);
	int headData[3] = {M.cols, M.rows, M.type()};
	fwrite(headData, sizeof(int), 3, file);
	fwrite(M.data, sizeof(char), M.step * M.rows, file);
	fclose(file);
	return true;
}

// Read matrix from binary file
bool Objectness::matRead(const string& filename, Mat& _M){
	FILE* f = fopen(_S(filename), "rb");
	if (f == NULL)
		return false;
	char buf[8];
	int pre = fread(buf,sizeof(char), 5, f);
	if (strncmp(buf, "CmMat", 5) != 0)	{
		printf("Invalidate CvMat data file %s\n", _S(filename));
		return false;
	}
	int headData[3]; // Width, height, type
	fread(headData, sizeof(int), 3, f);
	Mat M(headData[1], headData[0], headData[2]);
	fread(M.data, sizeof(char), M.step * M.rows, f);
	fclose(f);
	M.copyTo(_M);
	return true;
}

float distG(float d, float delta) {return exp(-d*d/(2*delta*delta));}

Mat Objectness::aFilter(float delta, int sz)
{
	float dis = float(sz-1)/2.f;
	Mat mat(sz, sz, CV_32F);
	for (int r = 0; r < sz; r++)
		for (int c = 0; c < sz; c++)
			mat.at<float>(r, c) = distG(sqrt(sqr(r-dis)+sqr(c-dis)) - dis, delta);
	return mat;
}

