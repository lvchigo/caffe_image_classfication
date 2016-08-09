#include "stdafx.h"
#include "Objectness_predict.h"
#include <cstdlib>
#include <execinfo.h>

const char* Objectness::_clrName[3] = {"MAXBGR", "HSV", "I"};


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
	CStr s1 = modelName + ".wS1", s2 = modelName + ".wS2", sI = modelName + ".idx";
	Mat filters1f, reW1f, idx1i, show3u;
	if (!matRead(s1, filters1f) || !matRead(sI, idx1i)){
		printf("Can't load model: %s or %s\n", _S(s1), _S(sI));
		return 0;
	}

	//filters1f = aFilter(0.8f, 8);
	//normalize(filters1f, filters1f, p, 1, NORM_MINMAX);

	normalize(filters1f, show3u, 1, 255, NORM_MINMAX, CV_8U);
    _bingF.update(filters1f);
    //_tigF.reconstruct(filters1f);

	_svmSzIdxs = idx1i;
	CV_Assert(_svmSzIdxs.size() > 1 && filters1f.size() == Size(_W, _W) && filters1f.type() == CV_32F);
	_svmFilter = filters1f;

	if (!matRead(s2, _svmReW1f) || _svmReW1f.size() != Size(2, _svmSzIdxs.size())){
		_svmReW1f = Mat();
		return -1;
	}
	return 1;
}

void Objectness::predictBBoxSI(Mat &img3u, ValStructVec<float, Vec4i> &valBoxes, vecI &sz, int NUM_WIN_PSZ, bool fast)
{
	const int numSz = _svmSzIdxs.size();
	const int imgW = img3u.cols, imgH = img3u.rows;

	valBoxes.reserve(10000);
	sz.clear(); sz.reserve(10000);
	for (int ir = numSz - 1; ir >= 0; ir--){
		int r = _svmSzIdxs[ir];
		int height = cvRound(pow(_base, r / _numT + _minT)), width = cvRound(pow(_base, r%_numT + _minT));
		if (height > imgH * _base || width > imgW * _base)
			continue;

		height = min(height, imgH), width = min(width, imgW);
		Mat im3u, matchCost1f, mag1u;
		resize(img3u, im3u, Size(cvRound(_W*imgW*1.0 / width), cvRound(_W*imgH*1.0 / height)));
		double ratioX = width / _W, ratioY = height / _W;
		/*double ratioX = _W*1.0 / width, ratioY = _W*1.0 / height;
		int maxM = max(imgH, imgW);
		if (maxM > 500) {
			height = min(height, imgH*500 / maxM), width = min(width, imgW*500 / maxM);
			ratioX = _W*1.0 / width * 500 / maxM, ratioY = _W*1.0 / height * 500 / maxM;
		}
		resize(img3u, im3u, Size(cvRound(ratioX*imgW), cvRound(ratioY*imgH)));
		ratioX = 1 / ratioX, ratioY = 1 / ratioY;*/

		gradientMag(im3u, mag1u);
		matchCost1f = _bingF.matchTemplate(mag1u);
		ValStructVec<float, Point> matchCost;
		nonMaxSup(matchCost1f, matchCost, _NSS, NUM_WIN_PSZ, fast);

		// Find true locations and match values
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
void Objectness::getObjBndBoxes(Mat &img3u, ValStructVec<float, Vec4i> &valBoxes, int numDetPerSize)
{
	CV_Assert(filtersLoaded());
	vecI sz;
	ValStructVec<float, Vec4i> valBoxesTemp;
	double ratioX = img3u.cols / 500.0, ratioY = img3u.rows / 375.0;
	resize(img3u, img3u, Size(500, 375));
	predictBBoxSI(img3u, valBoxesTemp, sz, numDetPerSize, false);
	predictBBoxSII(valBoxesTemp, sz);
	//if (valBoxesTemp.size() > 1000)
		//valBoxesTemp.resize(1000);

	const int imgW = img3u.cols, imgH = img3u.rows, imgS = imgH*imgW;
	Mat gray, edge;
	cvtColor(img3u, gray, CV_BGR2GRAY);	//opencv
	GaussianBlur(gray, gray, Size(7, 7), 1, 1);	//opencv
	vector<uchar> power(256);
	for (int i = 0; i < 256; i++)
		power[i] = (uchar)pow(i, 0.9);
	for (int row = 0; row < imgH; row++) {
		uchar *data = gray.ptr<uchar>(row);
		for (int col = 0; col < imgW; col++)
			data[col] = power[data[col]];
	}
	Canny(gray, edge, 30, 90, 3);	//opencv

	int counter = 0;
	double *edges = new double[imgS];
	const double MAX_DOUBLE = numeric_limits<double>::max();
	for (int i = 0; i < imgH; i++) {
		uchar *src = edge.ptr<uchar>(i);
		for (int j = 0; j < imgW; j++)
			edges[counter++] = (src[j] == 0) ? MAX_DOUBLE : 0;
	}
	vl_uindex *indexes = new vl_uindex[imgS];
	for (vl_uindex i = 0; i < imgS; i++)
		indexes[i] = i;
	vl_image_distance_transform_d(edges, imgW, imgH, 1, imgW, edges, indexes, 1.0, 0.0);
	vl_image_distance_transform_d(edges, imgH, imgW, imgW, 1, edges, indexes, 1.0, 0.0);
	Mat v(imgH, imgW, CV_32S), u(imgH, imgW, CV_32S);
	counter = 0;
	for (int i = 0; i < imgH; i++) {
		int *vdata = v.ptr<int>(i);
		int *udata = u.ptr<int>(i);
		for (int j = 0; j < imgW; j++) {
			vdata[j] = indexes[counter] / imgW;
			udata[j] = indexes[counter++] % imgW;
		}
	}
	delete[] indexes;
	delete[] edges;

	const int NUM_BOX = valBoxesTemp.size();
	for (int i = 0; i < NUM_BOX; i++) {
		float mVal = valBoxesTemp(i);
		Vec4i &box = valBoxesTemp[i];
		box[0] -= 1, box[1] -= 1, box[2] -= 1, box[3] -= 1;
		for (int j = 0; j < 4; j++) {
			Vec4i tmp(box);
			box[0] = minIdx(u, tmp[0], tmp[1], tmp[0], tmp[3]);
			box[2] = maxIdx(u, tmp[2], tmp[1], tmp[2], tmp[3]);
			box[1] = minIdx(v, tmp[0], tmp[1], tmp[2], tmp[1]);
			box[3] = maxIdx(v, tmp[0], tmp[3], tmp[2], tmp[3]);
			double ov = interUnio(tmp, box);
			if (ov > 0.95)
				break;
		}
		valBoxesTemp[i][0] = max(1, box[0]);
		valBoxesTemp[i][1] = max(1, box[1]);
		valBoxesTemp[i][2] = min(imgW, box[2] + 2);
		valBoxesTemp[i][3] = min(imgH, box[3] + 2);
	}
	
	Mat imgLab;
	vector<Vec4i> spBoxes;
	cvtColor(img3u, imgLab, CV_BGR2Lab);
	FelzenSegmentIndex(spBoxes, imgLab, 120.0f, 6);
	vecF thetas(5);
	thetas[0] = 0.1, thetas[1] = 0.2, thetas[2] = 0.3, thetas[3] = 0.4, thetas[4] = 0.5;
	mtse(spBoxes, valBoxesTemp, valBoxes, thetas, 0.8, true);

	for (int i = 0; i < valBoxes.size(); i++) {
		valBoxes[i][0] = (int)(valBoxes[i][0] * ratioX);
		valBoxes[i][1] = (int)(valBoxes[i][1] * ratioY);
		valBoxes[i][2] = (int)(valBoxes[i][2] * ratioX);
		valBoxes[i][3] = (int)(valBoxes[i][3] * ratioY);
	}
}

double Objectness::interUnio(const Vec4i &bb, const Vec4i &bbgt)
{
	int bi[4];
	bi[0] = max(bb[0], bbgt[0]);
	bi[1] = max(bb[1], bbgt[1]);
	bi[2] = min(bb[2], bbgt[2]);
	bi[3] = min(bb[3], bbgt[3]);	

	double iw = bi[2] - bi[0] + 1;
	double ih = bi[3] - bi[1] + 1;
	double ov = 0;
	if (iw>0 && ih>0){
		double ua = (bb[2]-bb[0]+1)*(bb[3]-bb[1]+1)+(bbgt[2]-bbgt[0]+1)*(bbgt[3]-bbgt[1]+1)-iw*ih;
		ov = iw*ih/ua;
	}	
	return ov;
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
		cout << "Error: not recognized color space" << endl;
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

// Read matrix from binary file
bool Objectness::matRead(const string& filename, Mat& _M){
	FILE* f = fopen(_S(filename), "rb");
	if (f == NULL)
		return false;
	char buf[8];
	int pre = fread(buf,sizeof(char), 5, f);
	if (strncmp(buf, "CmMat", 5) != 0)	{
		cout << "Invalidate CvMat data file " << _S(filename) << endl;
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

inline float distG(float d, float delta) {return exp(-d*d/(2*delta*delta));}

Mat Objectness::aFilter(float delta, int sz)
{
	float dis = float(sz-1)/2.f;
	Mat mat(sz, sz, CV_32F);
	for (int r = 0; r < sz; r++)
		for (int c = 0; c < sz; c++)
			mat.at<float>(r, c) = distG(sqrt(sqr(r-dis)+sqr(c-dis)) - dis, delta);
	return mat;
}

int Objectness::minIdx(Mat &input, int x1, int y1, int x2, int y2) {
	int min = 1000000;
	for (int i = y1; i <= y2; i++) {
		int *data = input.ptr<int>(i);
		for (int j = x1; j <= x2; j++) {
			if (data[j] < min)
				min = data[j];
		}
	}
	return min;
}

int Objectness::maxIdx(Mat &input, int x1, int y1, int x2, int y2) {
	int max = -1;
	for (int i = y1; i <= y2; i++) {
		int *data = input.ptr<int>(i);
		for (int j = x1; j <= x2; j++) {
			if (data[j] > max)
				max = data[j];
		}
	}
	return max;
}

