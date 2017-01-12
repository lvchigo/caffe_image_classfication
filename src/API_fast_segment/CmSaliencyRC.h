#pragma once

#include <string>
#include <vector>
#include <opencv/cv.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;


struct CmSaliencyRC
{
	typedef Mat (*GET_SAL_FUNC)(const Mat &);

	// Luminance Contrast [2]
	static Mat GetLC(const Mat &img3f);

	// Histogram Contrast of [3]
	static Mat GetHC(const Mat &img3f);

	// Region Contrast 
	static Mat GetRC(const Mat &img3f, string iid );
	

	static void SmoothByHist(const Mat &img3f, Mat &sal1f, float delta);
	static void SmoothByRegion(Mat &sal1f, const Mat &idx1i, int regNum, bool bNormalize = true);
	static void SmoothByGMMs(const Mat &img3f, Mat &sal1f, int fNum = 5, int bNum = 5, int wkSize = 0);

private:

	// Histogram based Contrast
	static void GetHC(const Mat &binColor3f, const Mat &colorNums1i, Mat &colorSaliency);

	// Region Contrast 
	static Mat GetRC(const Mat &img3f, string iid, const Mat &idx1i, int regNum, double sigmaDist = 0.4);
	static Mat GetRC(const Mat &img3f, string iid, double sigmaDist, double segK, int segMinSize, double segSigma);

	static void SmoothSaliency(Mat &sal1f, float delta, const vector<vector<pair<float, int>>> &similar);
	static void SmoothSaliency(const Mat &colorNum1i, Mat &sal1f, float delta, const vector<vector<pair<float, int>>> &similar);

	struct Region{
		Region() { pixNum = 0; ad2c = Point2d(0, 0);}
		int pixNum;  // Number of pixels
		vector<pair<float, int>> freIdx;  // Frequency of each color and its index
		Point2d centroid;
		Point2d ad2c; // Average distance to image center
	};
	static void BuildRegions(const Mat& regIdx1i, vector<Region> &regs, const Mat &colorIdx1i, int colorNum);
	static void RegionContrast(const vector<Region> &regs, const Mat &color3fv, Mat& regSal1d, double sigmaDist);

	static int Quantize(const Mat& img3f, Mat &idx1i, Mat &_color3f, Mat &_colorNum, int colorNums[3], double ratio = 0.95);

	// Get border regions, which typically corresponds to background region
	static Mat GetBorderReg(const Mat &idx1i, int regNum, double ratio = 0.02, double thr = 0.3);  

	// AbsAngle: Calculate magnitude and angle of vectors.
	static void AbsAngle(const Mat& cmplx32FC2, Mat& mag32FC1, Mat& ang32FC1);

	// GetCmplx: Get a complex value image from it's magnitude and angle.
	static void GetCmplx(const Mat& mag32F, const Mat& ang32F, Mat& cmplx32FC2);
};

/************************************************************************/
/*[1]R. Achanta, S. Hemami, F. Estrada and S. Susstrunk, Frequency-tuned*/
/*   Salient Region Detection, IEEE CVPR, 2009.							*/
/*[2]Y. Zhai and M. Shah. Visual attention detection in video sequences */
/*   using spatiotemporal cues. In ACM Multimedia 2006.					*/
/*[3]M.-M. Cheng, N. J. Mitra, X. Huang, P.H.S. Torr S.-M. Hu. Global	*/
/*   Contrast based Salient Region Detection. IEEE PAMI, 2014.			*/
/*[4]X. Hou and L. Zhang. Saliency detection: A spectral residual		*/
/*   approach. In IEEE CVPR 2007, 2007.									*/
/************************************************************************/
