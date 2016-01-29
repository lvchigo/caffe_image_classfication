#pragma once
#include <cmath>
#include "Get_ColorGist.h"

#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include <malloc.h>

namespace ImageTypeAJudge_2_2_0
{
	static const int fc = 4;
	static const int padding = 5;
	static const int gist_image_size = 32;
	static const int nblocks=4;
	static const int n_scale=3;
	static const int orientations_per_scale[n_scale]={8,8,4};

	static Mat image_add_padding(const Mat& src, const int padding)
	{
		Mat img = Mat::zeros(src.rows+2*padding, src.cols+2*padding, src.type());
		Mat roi(img, Rect(padding, padding, src.cols, src.rows));
		src.copyTo(roi);

		for(int j=0; j<padding; ++j)
		{
			float* imgData = img.ptr<float>(j) + padding;
			const float* srcData = src.ptr<float>(padding-j-1);
			memcpy(imgData, srcData, sizeof(float)*src.cols);
			
			imgData = img.ptr<float>(j+padding+src.rows) + padding;
			srcData = src.ptr<float>(src.rows-j-1);
			memcpy(imgData, srcData, sizeof(float)*src.cols);
		}

		for(int j=0; j<img.rows; ++j)
		{
			float* imgData = img.ptr<float>(j);
			for(int i=0; i<padding; ++i)
			{
				imgData[i] = imgData[padding+padding-i-1];
				imgData[i+padding+src.cols] = imgData[img.cols-padding-i-1];
			}
		}

		return img;
	}

	static void image_rem_padding(Mat dst, const Mat& src, const int padding)
	{
		Mat roi(src, Rect(padding, padding, src.cols-padding-padding, src.rows-padding-padding));
		roi.copyTo(dst);
	}

	static void fftshift(Mat& img)
	{
		const int cx = img.cols/2;
		const int cy = img.rows/2;
		Mat tmp;
		Mat q0(img, Rect(0, 0, cx, cy));
		Mat q1(img, Rect(cx, 0, cx, cy));
		Mat q2(img, Rect(0, cy, cx, cy));
		Mat q3(img, Rect(cx, cy, cx, cy));

		q0.copyTo(tmp);
		q3.copyTo(q0);
		tmp.copyTo(q3);

		q1.copyTo(tmp);
		q2.copyTo(q1);
		tmp.copyTo(q2);
	}

	static void fftshift(float *data, const int w, const int h)
	{
		float *buff = (float *) malloc(w*h*sizeof(float));
		memcpy(buff, data, w*h*sizeof(float));

		for(int j = 0; j < (h+1)/2; j++)
		{
			for(int i = 0; i < (w+1)/2; i++) {
				data[(j+h/2)*w + i+w/2] = buff[j*w + i];
			}

			for(int i = 0; i < w/2; i++) {
				data[(j+h/2)*w + i] = buff[j*w + i+(w+1)/2];
			}
		}

		for(int j = 0; j < h/2; j++)
		{
			for(int i = 0; i < (w+1)/2; i++) {
				data[j*w + i+w/2] = buff[(j+(h+1)/2)*w + i];
			}

			for(int i = 0; i < w/2; i++) {
				data[j*w + i] = buff[(j+(h+1)/2)*w + i+(w+1)/2];
			}
		}

		free(buff);
	}

	vector<Mat> create_gabor(const int nscales, const int *orientations_per_scale, const int width, const int height)
	{
		int nfilters = 0;
		for(int i=0;i<nscales;i++)  nfilters+=orientations_per_scale[i];

		float **param = (float **) malloc(nscales * nfilters * sizeof(float *));
		for(int i = 0; i < nscales * nfilters; i++) {
			param[i] = (float *) malloc(4*sizeof(float));
		}

		float *fr = (float *) malloc(width*height*sizeof(float));
		float *f  = (float *) malloc(width*height*sizeof(float));

		int l = 0;
		for(int i = 1; i <= nscales; i++)
		{
			for(int j = 1; j <= orientations_per_scale[i-1]; j++)
			{
				param[l][0] = 0.35f;
				param[l][1] = 0.3/pow(1.85f, i-1);
				param[l][2] = 16*pow(double(orientations_per_scale[i-1]), 2)/pow(32.0, 2);
				param[l][3] = CV_PI/(orientations_per_scale[i-1])*(j-1);
				l++;
			}
		}

		for(int j = 0; j < height; j++)
		{
			for(int i = 0; i < width; i++)
			{
				const float fx = (float) i - width/2.0f;
				const float fy = (float) j - height/2.0f;
				fr[j*width + i] = sqrt(fx*fx + fy*fy);
				f[j*width + i]  = atan2(fy, fx);
			}
		}

		fftshift(fr, width, height);
		fftshift(f, width, height);

		vector<Mat> G;
		for(int fn = 0; fn < nfilters; fn++)
		{
			Mat G0(height, width, CV_32F);

			float *f_ptr = f;
			float *fr_ptr = fr;

			for(int j = 0; j < height; j++)
			{
				float* gData = G0.ptr<float>(j);
				for(int i = 0; i < width; i++)
				{
					float tmp = *f_ptr++ + param[fn][3];

					if(tmp < -CV_PI) {
						tmp += 2.0f*CV_PI;
					}
					else if (tmp > CV_PI) {
						tmp -= 2.0f*CV_PI;
					}

					gData[i] = exp(-10.0f*param[fn][0]*(*fr_ptr/height/param[fn][1]-1)*(*fr_ptr/width/param[fn][1]-1)-2.0f*param[fn][2]*CV_PI*tmp*tmp);
					fr_ptr++;
				}
			}
			
			G.push_back(G0);
		}

		for(int i = 0; i < nscales * nfilters; i++) 
		{
			free(param[i]);
		}
		free(param);

		free(fr);
		free(f);

		return G;
	}
	
	Mat create_filter(const int fc, const int width, const int height)
	{
		/* Alloc memory */
		Mat gfc(height, width, CV_32F);

		/* Build whitening filter */
		const float s1 = fc/sqrt(log(2.0));
		for(int j = 0; j < height; j++)
		{
			float* gfcData = gfc.ptr<float>(j);
			for(int i = 0; i < width; i++)
			{
				const float fx = (float) i - width/2.0f;
				const float fy = (float) j - height/2.0f;

				gfcData[i] = exp(-(fx*fx + fy*fy) / (s1*s1));
			}
		}
		fftshift(gfc);
		return gfc;
	}

	void prefilt(Mat& src, Mat& gfc, const int padding=5)
	{
		/* Log */
		src += Scalar::all(1.0);
		log(src, src);

		Mat img_pad = image_add_padding(src, padding);

		/* Get sizes */
		int width = img_pad.cols;
		int height = img_pad.rows;

		Mat planesIn[] = {img_pad, Mat::zeros(img_pad.size(), CV_32F)};
		Mat in, out;
		merge(planesIn, 2, in);

		/* FFT */
		dft(in, out);

		/* Apply whitening filter */
		Mat planesOut[2];
		split(out, planesOut);
		multiply(planesOut[0], gfc, planesOut[0]);
		multiply(planesOut[1], gfc, planesOut[1]);
		merge(planesOut, 2, out);

		/* IFFT */		
		dft(out, out, DFT_INVERSE);
		split(out, planesOut);
			
		/* Local contrast normalisation */
		planesOut[0] /= (width*height);
		img_pad -= planesOut[0];
		multiply(img_pad, img_pad, planesOut[0]);
		planesOut[1] = Scalar::all(0.0);
		
		/* FFT */
		merge(planesOut, 2, out);
		dft(out, out);

		/* Apply contrast normalisation filter */
		split(out, planesOut);
		multiply(planesOut[0], gfc, planesOut[0]);
		multiply(planesOut[1], gfc, planesOut[1]);

		/* IFFT */
		merge(planesOut, 2, out);
		dft(out, out, DFT_INVERSE);
		
		/* Get result from contrast normalisation filter */
		split(out, planesOut);
		magnitude(planesOut[0], planesOut[1], planesOut[0]);
		planesOut[0] /= (width*height);
		sqrt(planesOut[0], planesOut[0]);
		planesOut[0] += Scalar::all(0.2);

		divide(img_pad, planesOut[0], img_pad);

		image_rem_padding(src, img_pad, padding);
	}

	void color_prefilt(Mat& src, Mat& gfc, const int padding=5)
	{
		/* Log */
		Mat planes[3];
		split(src, planes);
		planes[0] += Scalar::all(1.0);
		planes[1] += Scalar::all(1.0);
		planes[2] += Scalar::all(1.0);
		log(planes[0], planes[0]);
		log(planes[1], planes[1]);
		log(planes[2], planes[2]);

		Mat R_pad = image_add_padding(planes[2], padding);
		Mat G_pad = image_add_padding(planes[1], padding);
		Mat B_pad = image_add_padding(planes[0], padding);

		/* Get sizes */
		int width = R_pad.cols;
		int height = R_pad.rows;

		Mat planesIn[] = {R_pad, Mat::zeros(R_pad.size(), CV_32F)};
		Mat in, out;
		Mat planesOut[2];
		/****************************R********************************************/
		/* FFT */	
		planesIn[0] = R_pad;
		merge(planesIn, 2, in);
		dft(in, out);
		split(out, planesOut);
		
		/* Apply whitening filter */
		multiply(planesOut[0], gfc, planesOut[0]);
		multiply(planesOut[1], gfc, planesOut[1]);
		
		/* IFFT */
		merge(planesOut, 2, out);
		dft(out, out, DFT_INVERSE);	
		
		/* Local contrast normalisation */
		split(out, planesOut);
		planesOut[0] /= (width*height);
		R_pad -= planesOut[0];
		/****************************G********************************************/
		/* FFT */
		planesIn[0] = G_pad;
		merge(planesIn, 2, in);
		dft(in, out);
		split(out, planesOut);
				
		/* Apply whitening filter */
		multiply(planesOut[0], gfc, planesOut[0]);
		multiply(planesOut[1], gfc, planesOut[1]);
		
		/* IFFT */
		merge(planesOut, 2, out);
		dft(out, out, DFT_INVERSE);	
		
		/* Local contrast normalisation */
		split(out, planesOut);
		planesOut[0] /= (width*height);
		G_pad -= planesOut[0];
		/****************************G********************************************/
		/* FFT */
		planesIn[0] = B_pad;
		merge(planesIn, 2, in);
		dft(in, out);
		split(out, planesOut);
				
		/* Apply whitening filter */
		multiply(planesOut[0], gfc, planesOut[0]);
		multiply(planesOut[1], gfc, planesOut[1]);
		
		/* IFFT */
		merge(planesOut, 2, out);
		dft(out, out, DFT_INVERSE);	
		
		/* Local contrast normalisation */
		split(out, planesOut);
		planesOut[0] /= (width*height);
		B_pad -= planesOut[0];

		planesOut[0] = (R_pad + G_pad + B_pad) / 3.0f;
		multiply(planesOut[0], planesOut[0], planesOut[0]);
		/* FFT */
		planesOut[1] = Scalar::all(0);
		merge(planesOut, 2, out);
		dft(out, out);

		/* Apply contrast normalisation filter */
		split(out, planesOut);
		multiply(planesOut[0], gfc, planesOut[0]);
		multiply(planesOut[1], gfc, planesOut[1]);

		/* IFFT */
		merge(planesOut, 2, out);
		dft(out, out, DFT_INVERSE);

		/* Get result from contrast normalisation filter */
		split(out, planesOut);
		magnitude(planesOut[0], planesOut[1], planesOut[0]);
		planesOut[0] /= (width*height);
		sqrt(planesOut[0], planesOut[0]);
		planesOut[0] += Scalar::all(0.2);
		divide(R_pad, planesOut[0], R_pad);
		divide(G_pad, planesOut[0], G_pad);
		divide(B_pad, planesOut[0], B_pad);

		image_rem_padding(planes[2], R_pad, padding);
		image_rem_padding(planes[1], G_pad, padding);
		image_rem_padding(planes[0], B_pad, padding);
		merge(planes, 3, src);
	}

	static void down_N(float *res, Mat src, int N)
	{
		int *nx = (int *) malloc((N+1)*sizeof(int));
		int *ny = (int *) malloc((N+1)*sizeof(int));

		for(int i = 0; i < N+1; i++)
		{
			nx[i] = i*src.cols/(N);
			ny[i] = i*src.rows/(N);
		}

		for(int l = 0; l < N; l++)
		{
			for(int k = 0; k < N; k++)
			{
				float mean = 0.0f;

				for(int j = ny[l]; j < ny[l+1]; j++)
				{
					float* srcData = src.ptr<float>(j);
					for(int i = nx[k]; i < nx[k+1]; i++) 
					{
						mean += srcData[i];
					}
				}

				float denom = (float)(ny[l+1]-ny[l])*(nx[k+1]-nx[k]);

				res[k*N+l] = mean / denom;
			}
		}

		free(nx);
		free(ny);
	}

	void gist_gabor(Mat src, const int w, const vector<Mat>& G,float *res)
	{
		/* Get sizes */
		int width = src.cols;
		int height = src.rows;

		//float *res = (float *) malloc(w*w*G.size()*sizeof(float));

		Mat planesIn[] = { src, Mat::zeros(src.size(), CV_32F)};
		Mat in, out, iout;
		Mat planesOut[2];
		/* FFT */
		merge(planesIn, 2, in);
		dft(in, out);

		for(int k = 0; k < G.size(); k++)
		{
			split(out, planesOut);
			multiply(planesOut[0], G[k], planesOut[0]);
			multiply(planesOut[1], G[k], planesOut[1]);
			
			merge(planesOut, 2, iout);
			dft(iout, iout, DFT_INVERSE);
			split(iout, planesOut);
		
			magnitude(planesOut[0], planesOut[1], planesOut[0]);
			planesOut[0] /= (width*height);
			
			down_N(res+k*w*w, planesOut[0], w);
		}

		//return res;
	}

	void color_gist_gabor(Mat src, const int w, const vector<Mat>& G,float *res)
	{
		//float *res = (float *) malloc(3*w*w*G.size()*sizeof(float));
		Mat planes[3];
		split(src, planes);
		float *resR = new float[w*w*G.size()];
		float *resG = new float[w*w*G.size()];
		float *resB = new float[w*w*G.size()];
		gist_gabor(planes[2], w, G,resR);
		gist_gabor(planes[1], w, G,resG);
		gist_gabor(planes[0], w, G,resB);
		memcpy(res, resR, sizeof(float)*w*w*G.size());
		memcpy(res+w*w*G.size(), resG, sizeof(float)*w*w*G.size());
		memcpy(res+w*w*G.size()*2, resB, sizeof(float)*w*w*G.size());
		if(resR){delete []resR; resR = NULL;}
		if(resG){delete []resG; resG = NULL;}
		if(resB){delete []resB; resB = NULL;}
		//return res;
	}

	int bw_gist_scaletab(Mat src, int w, int n_scale, const int *n_orientation,float *res)
	{
		if(src.cols < 8 || src.rows < 8)
		{
			fprintf(stderr, "Error: bw_gist_scaletab() - Image not big enough !\n");
			return -1;
		}

		int numberBlocks = w;
		int tot_oris=0;
		for(int i=0;i<n_scale;i++) tot_oris+=n_orientation[i];

		Mat img;
		src.convertTo(img, CV_32F);
		if(img.channels() != 1)
			cvtColor(img, img, CV_BGR2GRAY);
		vector<Mat> G = create_gabor(n_scale, n_orientation, img.cols, img.rows);

		Mat gfc = create_filter(fc, img.cols+padding*2, img.rows+padding*2);
		prefilt(img, gfc, padding);

		gist_gabor(img, numberBlocks, G,res);

		//return gist_gabor(img, numberBlocks, G);
		return 0;
	}

	void bw_gist(Mat src, int w, int a, int b, int c,float *res)
	{
		int orientationsPerScale[3];

		orientationsPerScale[0] = a;
		orientationsPerScale[1] = b;
		orientationsPerScale[2] = c;

		bw_gist_scaletab(src,w,3,orientationsPerScale,res);

		//return bw_gist_scaletab(src,w,3,orientationsPerScale);
	}

	int color_gist_scaletab(Mat src, int w, int n_scale, const int *n_orientation,float *res) 
	{
		if(src.cols < 8 || src.rows < 8)
		{
			fprintf(stderr, "Error: color_gist_scaletab() - Image not big enough !\n");
			return -1;
		}

		int numberBlocks = w;
		int tot_oris=0;
		for(int i=0;i<n_scale;i++) tot_oris+=n_orientation[i];

		Mat img;
		src.convertTo(img, CV_32F);
		vector<Mat> G = create_gabor(n_scale, n_orientation, img.cols, img.rows);

		Mat gfc = create_filter(fc, img.cols+padding*2, img.rows+padding*2);
		color_prefilt(img, gfc, padding);

		color_gist_gabor(img, nblocks, G,res);

		//return color_gist_gabor(img, nblocks, G);
		return 0;
	}

	void color_gist(Mat src, int w, int a, int b, int c,float *res) 
	{  
		int orientationsPerScale[3];

		orientationsPerScale[0] = a;
		orientationsPerScale[1] = b;
		orientationsPerScale[2] = c;

		color_gist_scaletab(src,w,3,orientationsPerScale,res);

		//return color_gist_scaletab(src,w,3,orientationsPerScale);

	}

/*	ColorGistExtracter::ColorGistExtracter(const bool _bColor) : BaseFeatureExtracter(BaseFeatureExtracter::CGIST), bColor(_bColor)
	{
		gfc = create_filter(fc, gist_image_size+padding*2, gist_image_size+padding*2);
		G = create_gabor(n_scale, orientations_per_scale, gist_image_size, gist_image_size);
	}

	ColorGistExtracter::~ColorGistExtracter()
	{
	}

	BaseFeature* ColorGistExtracter::Extract(IplImage* image)
	{
		if(image==NULL || (bColor&&image->nChannels!=3))
			return NULL;
		Mat img;
		resize(Mat(image), img, Size(gist_image_size, gist_image_size));
		float *desc = NULL;
		int descsize=0;
		// compute descriptor size
		for(int i=0;i<n_scale;i++) 
			descsize+=nblocks*nblocks*orientations_per_scale[i];
		if(bColor)
		{
			Mat fimg;
			img.convertTo(fimg, CV_32F);

			color_prefilt(fimg, gfc, padding);

			desc = color_gist_gabor(fimg, nblocks, G);  

			descsize*=3; // color
		}
		else
		{
			if(img.channels() != 1)
				cvtColor(img, img, CV_BGR2GRAY);
			Mat fimg;
			img.convertTo(fimg, CV_32F);

			prefilt(fimg, gfc, padding);

			desc = gist_gabor(fimg, nblocks, G);  
		}
		
		VectorFeature<float>* feature = NULL;
		if(desc)
		{
			feature = new VectorFeature<float>(BaseFeature::VECTOR_FLT, descsize);
			memcpy(feature->data, desc, sizeof(float)*descsize);
			//for(int i=0; i<feature->nLength; ++i)
			//	printf("%d:%.4f ", i, feature->data[i]);
			//printf("\n\n");
			//getchar();
			free(desc);
		}
		return feature;
	}*/
}
