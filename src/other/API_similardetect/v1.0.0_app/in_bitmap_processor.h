
#ifndef IN_BITMAP_PROCESSOR
#define IN_BITMAP_PROCESSOR

#include <android/bitmap.h>

///
/*
 *
 *
 *  http://mobilepearls.com/labs/native-android-api/include/android/bitmap.h
 *  ANDROID_BITMAP_FORMAT_NONE      = 0,
    ANDROID_BITMAP_FORMAT_RGBA_8888 = 1,
    ANDROID_BITMAP_FORMAT_RGB_565   = 4,
    ANDROID_BITMAP_FORMAT_RGBA_4444 = 7,
    ANDROID_BITMAP_FORMAT_A_8       = 8,
 */
//
#define RGB565_R(p) ((((p) & 0xF800) >> 11) << 3)
#define RGB565_G(p) ((((p) & 0x7E0 ) >> 5)  << 2)
#define RGB565_B(p) ( ((p) & 0x1F  )        << 3)
#define MAKE_RGB565(r,g,b) ((((r) >> 3) << 11) | (((g) >> 2) << 5) | ((b) >> 3))
//
#define RGBA8888_A(p) (((p) & 0xFF000000) >> 24)
#define RGBA8888_R(p) (((p) & 0x00FF0000) >> 16)
#define RGBA8888_G(p) (((p) & 0x0000FF00) >>  8)
#define RGBA8888_B(p)  ((p) & 0x000000FF)
#define MAKE_RGBA8888(r,g,b,a) (((a) << 24) | ((r) << 16) | ((g) << 8) | (b))
//
#define RGBA4444_R(p) ((((p) & 0xF000) >> 12) << 4)
#define RGBA4444_G(p) ((((p) & 0x0F00) >> 8) << 4)
#define RGBA4444_B(p) ((((p) & 0x00F0) >> 4) << 4)
#define RGBA4444_A(p) ( ((p) & 0x000F) << 4)
#define MAKE_RGBA4444(r,g,b,a) ( (((r)>>4)<<12) | (((g)>>4)<<8) | (((b)>>4)<<4) | ((a)>>4) )
//
#define RGB_TO_GRAY(r,g,b) ( ( (r) * 38 + (g) * 75 + (b) * 15 ) >> 7 )
//#define RGB_TO_GRAY(r,g,b) ( ( (r)  + (g)  + (b)  ) / 3 )
#define RGB565_TO_GRAY(p) ( RGB_TO_GRAY(RGB565_R(p),RGB565_G(p),RGB565_B(p)) )
#define RGBA8888_TO_GRAY(p) ( RGB_TO_GRAY(RGBA8888_R(p),RGBA8888_G(p),RGBA8888_B(p)) )
#define RGBA4444_TO_GRAY(p) ( RGB_TO_GRAY(RGBA4444_R(p),RGBA4444_G(p),RGBA4444_B(p)) )

// convert input bitmap to grayscale bitmap
int bitmap_update_to_gray(JNIEnv *env, jobject bmp) {
	if (bmp == NULL)
		return -1;
	//
	AndroidBitmapInfo info;
	int ret;
	if ((ret = AndroidBitmap_getInfo(env, bmp, &info)) < 0) {
		return -1;
	}
	if (info.width <= 0 || info.height <= 0
			|| (info.format != ANDROID_BITMAP_FORMAT_RGB_565
					&& info.format != ANDROID_BITMAP_FORMAT_RGBA_8888
					&& info.format != ANDROID_BITMAP_FORMAT_RGBA_4444)) {
		return -1;
	}

	// Lock the bitmap to get the buffer
	void * pixels = NULL;
	int res = AndroidBitmap_lockPixels(env, bmp, &pixels);
	if (res < 0 || pixels == NULL)  return -1;

	int x = 0, y = 0;
	// From top to bottom
	for (y = 0; y < info.height; ++y) {
		// From left to right
		for (x = 0; x < info.width; ++x) {
			int a = 0, r = 0, g = 0, b = 0;
			void *pixel = NULL;
			// Get each pixel by format
			switch (info.format) {
			case ANDROID_BITMAP_FORMAT_RGB_565:
			{
				pixel = ((uint16_t *) pixels) + y * info.width + x;
				uint16_t va = *(uint16_t *) pixel;
				r = RGB565_R(va);
				g = RGB565_G(va);
				b = RGB565_B(va);
				break;
			}
			case ANDROID_BITMAP_FORMAT_RGBA_8888:
			{
				pixel = ((uint32_t *) pixels) + y * info.width + x;
				uint32_t vb = *(uint32_t *) pixel;
				a = RGBA8888_A(vb);
				r = RGBA8888_R(vb);
				g = RGBA8888_G(vb);
				b = RGBA8888_B(vb);
				break;
			}
			case ANDROID_BITMAP_FORMAT_RGBA_4444:
			{
				pixel = ((uint16_t *) pixels) + y * info.width + x;
				uint16_t vc = *(uint16_t *) pixel;
				a = RGBA4444_A(vc);
				r = RGBA4444_R(vc);
				g = RGBA4444_G(vc);
				b = RGBA4444_B(vc);
				break;
			}
			}//switch

			// Grayscale
			int gray = RGB_TO_GRAY(r, g, b);

			// Write the pixel back
			switch(info.format){
			case ANDROID_BITMAP_FORMAT_RGB_565:
				*((uint16_t *) pixel) = MAKE_RGB565(gray, gray, gray);
				break;
			case ANDROID_BITMAP_FORMAT_RGBA_8888: // RGBA_8888
				*((uint32_t *) pixel) = MAKE_RGBA8888(gray, gray, gray, a);
				break;
			case ANDROID_BITMAP_FORMAT_RGBA_4444: // RGBA_4444
				*((uint16_t *) pixel) = MAKE_RGBA4444(gray, gray, gray, a);
				break;
			} //switch
		} //for-x
	} //for-y

	//
	AndroidBitmap_unlockPixels(env, bmp) ;
	return 0;
}

//======================= for mat ====================================
//
template<class T>
int rgb565_resize_to_gray_mat(uint16_t* rgb, int src_width, int src_height, T* &mat,
		int dst_width, int dst_height) {
	if (NULL == rgb)
		return -1;
	mat = (T*) malloc(dst_width * dst_height * sizeof(T));

	float a = 0.0f;
	float b = 0.0f;

	for (int y = 0; y < dst_height; y++) {
		for (int x = 0; x < dst_width; x++) {
			a = x * src_width / dst_width;
			b = y * src_height / dst_height;
			if (a == 0 || b == 0) {
				//fdata[y * dst_width + x] = int32_to_float(argb[(int) b * src_width + (int) a]);
				mat[y + x * dst_height] = (T) RGB565_TO_GRAY(
						rgb[(int) b * src_width + (int) a]);
				continue;
			}

			if (a / (unsigned int) a == 1.0 && b / (unsigned int) b == 1.0) {
				//fdata[y * dst_width + x] = int32_to_float(argb[(int) b * src_width + (int) a]);
				mat[y + x * dst_height] = (T) RGB565_TO_GRAY(
						rgb[(int) b * src_width + (int) a]);
				continue;
			}

			float x11 = (float) RGB565_TO_GRAY(
					rgb[(int) b * src_width + (int) a]);
			float x12 = (float) RGB565_TO_GRAY(
					rgb[(int) (b + 1) * src_width + (int) a]);
			float x21 = (float) RGB565_TO_GRAY(
					rgb[(int) b * src_width + (int) (a + 1)]);
			float x22 = (float) RGB565_TO_GRAY(
					rgb[(int) (b + 1) * src_width + (int) (a + 1)]);
			float f1 = (x - a) * x22 + (a + 1 - x) * x12;
			float f2 = (x - a) * x21 + (a + 1 - x) * x11;
			float f = (b + 1 - y) * f2 + (y - b) * f1;
			f = f < 0 ? 0 : f;
			f = f > 255 ? 255 : f;
			//fdata[y * dst_width + x] =  f;
			mat[y + x * dst_height] = (T) f;
		}
	}

	return 0;
}
//
template<class T>
int rgba8888_resize_to_gray_mat(uint32_t * rgba, int src_width, int src_height, T* &mat,
		int dst_width, int dst_height) {
	if (NULL == rgba)
		return -1;
	mat = (T*) malloc(dst_width * dst_height * sizeof(T));

	float a = 0.0f;
	float b = 0.0f;

	for (int y = 0; y < dst_height; y++) {
		for (int x = 0; x < dst_width; x++) {
			a = x * src_width / dst_width;
			b = y * src_height / dst_height;
			if (a == 0 || b == 0) {
				//fdata[y * dst_width + x] = int32_to_float(argb[(int) b * src_width + (int) a]);
				mat[y + x * dst_height] = (T) RGBA8888_TO_GRAY(
						rgba[(int) b * src_width + (int) a]);
				continue;
			}

			if (a / (unsigned int) a == 1.0 && b / (unsigned int) b == 1.0) {
				//fdata[y * dst_width + x] = int32_to_float(argb[(int) b * src_width + (int) a]);
				mat[y + x * dst_height] = (T) RGBA8888_TO_GRAY(
						rgba[(int) b * src_width + (int) a]);
				continue;
			}

			float x11 = (float) RGBA8888_TO_GRAY(
					rgba[(int) b * src_width + (int) a]);
			float x12 = (float) RGBA8888_TO_GRAY(
					rgba[(int) (b + 1) * src_width + (int) a]);
			float x21 = (float) RGBA8888_TO_GRAY(
					rgba[(int) b * src_width + (int) (a + 1)]);
			float x22 = (float) RGBA8888_TO_GRAY(
					rgba[(int) (b + 1) * src_width + (int) (a + 1)]);
			float f1 = (x - a) * x22 + (a + 1 - x) * x12;
			float f2 = (x - a) * x21 + (a + 1 - x) * x11;
			float f = (b + 1 - y) * f2 + (y - b) * f1;
			f = f < 0 ? 0 : f;
			f = f > 255 ? 255 : f;
			//fdata[y * dst_width + x] =  f;
			mat[y + x * dst_height] = (T) f;
		}
	}

	return 0;
}

//
template<class T>
int rgba4444_resize_to_gray_mat(uint16_t* rgba, int src_width, int src_height, T* &mat,
		int dst_width, int dst_height) {
	if (NULL == rgba)
		return -1;
	mat = (T*) malloc(dst_width * dst_height * sizeof(T));

	float a = 0.0f;
	float b = 0.0f;

	for (int y = 0; y < dst_height; y++) {
		for (int x = 0; x < dst_width; x++) {
			a = x * src_width / dst_width;
			b = y * src_height / dst_height;
			if (a == 0 || b == 0) {
				//fdata[y * dst_width + x] = int32_to_float(argb[(int) b * src_width + (int) a]);
				mat[y + x * dst_height] = (T) RGBA4444_TO_GRAY(
						rgba[(int) b * src_width + (int) a]);
				continue;
			}

			if (a / (unsigned int) a == 1.0 && b / (unsigned int) b == 1.0) {
				//fdata[y * dst_width + x] = int32_to_float(argb[(int) b * src_width + (int) a]);
				mat[y + x * dst_height] = (T) RGBA4444_TO_GRAY(
						rgba[(int) b * src_width + (int) a]);
				continue;
			}

			float x11 = (float) RGBA4444_TO_GRAY(
					rgba[(int) b * src_width + (int) a]);
			float x12 = (float) RGBA4444_TO_GRAY(
					rgba[(int) (b + 1) * src_width + (int) a]);
			float x21 = (float) RGBA4444_TO_GRAY(
					rgba[(int) b * src_width + (int) (a + 1)]);
			float x22 = (float) RGBA4444_TO_GRAY(
					rgba[(int) (b + 1) * src_width + (int) (a + 1)]);
			float f1 = (x - a) * x22 + (a + 1 - x) * x12;
			float f2 = (x - a) * x21 + (a + 1 - x) * x11;
			float f = (b + 1 - y) * f2 + (y - b) * f1;
			f = f < 0 ? 0 : f;
			f = f > 255 ? 255 : f;
			//fdata[y * dst_width + x] =  f;
			mat[y + x * dst_height] = (T) f;
		}
	}

	return 0;
}

//
template<class T>
int bitmap_resize_to_gray_mat(JNIEnv *env, jobject bmp,
		T* &mat, int dst_width, int dst_height)
{
	int flag  = -100;
	if (bmp == NULL)
		return flag;

	//
	AndroidBitmapInfo info;
	int ret;
	if ((ret = AndroidBitmap_getInfo(env, bmp, &info)) < 0) {
		flag = -101;
		return flag;
	}
	if (info.width <= 0 || info.height <= 0
			|| (info.format != ANDROID_BITMAP_FORMAT_RGB_565
					&& info.format != ANDROID_BITMAP_FORMAT_RGBA_8888
					&& info.format != ANDROID_BITMAP_FORMAT_RGBA_4444)) {
		flag = -102;
		return flag;
	}

	// Lock the bitmap to get the buffer
	void * pixels = NULL;
	ret = AndroidBitmap_lockPixels(env, bmp, &pixels);
	if (ret < 0 || pixels == NULL)
		return ret;

	// Write the pixel back
	switch (info.format) {
	case ANDROID_BITMAP_FORMAT_RGB_565:
		flag = rgb565_resize_to_gray_mat<T>((uint16_t *) pixels, info.width, info.height, mat, dst_width, dst_height);
		break;
	case ANDROID_BITMAP_FORMAT_RGBA_8888:
		flag = rgba8888_resize_to_gray_mat<T>((uint32_t *) pixels, info.width, info.height, mat, dst_width, dst_height);
		break;
	case ANDROID_BITMAP_FORMAT_RGBA_4444:
		flag = rgba4444_resize_to_gray_mat<T>((uint16_t *) pixels, info.width, info.height, mat, dst_width, dst_height);
		break;
	} //switch

	//
	AndroidBitmap_unlockPixels(env, bmp);
	return flag;
}

//======================= normal ====================================
//
template<class T>
int rgb565_resize(uint16_t* rgb, int src_width, int src_height, T* &mat,
		int dst_width, int dst_height) {
	if (NULL == rgb)
		return -1;

	int nchannels = 3;
	mat = (T*) malloc(dst_width * dst_height * nchannels * sizeof(T));

	int nx = 0;
	int ny = 0;
	uint16_t p;
	for (int y = 0; y < dst_height; y++) {
		ny = (int)(1.0f*y * src_height / dst_height+0.5);
		for (int x = 0; x < dst_width; x++) {
			nx = (int)(1.0f*x * src_width / dst_width+0.5);

			p = rgb[ny * src_width + nx];
			mat[y * dst_width*nchannels + x*nchannels] =  RGB565_R(p);
			mat[y * dst_width*nchannels + x*nchannels + 1] =  RGB565_G(p);
			mat[y * dst_width*nchannels + x*nchannels + 2] =  RGB565_B(p);

		}//for-x
	}//for-y

	return 0;
}
//
template<class T>
int rgba8888_resize(uint32_t * rgba, int src_width, int src_height, T* &mat,
		int dst_width, int dst_height) {
	if (NULL == rgba)
		return -1;

	int nchannels = 3;
	mat = (T*) malloc(dst_width * dst_height * nchannels * sizeof(T));

	int nx = 0;
	int ny = 0;
	uint16_t p;
	for (int y = 0; y < dst_height; y++) {
		ny = (int) (1.0f * y * src_height / dst_height + 0.5);
		for (int x = 0; x < dst_width; x++) {
			nx = (int) (1.0f * x * src_width / dst_width + 0.5);

			p = rgba[ny * src_width + nx];
			mat[y * dst_width * nchannels + x * nchannels] = RGBA8888_R(p);
			mat[y * dst_width * nchannels + x * nchannels + 1] = RGBA8888_G(p);
			mat[y * dst_width * nchannels + x * nchannels + 2] = RGBA8888_B(p);

		} //for-x
	} //for-y

	return 0;
}

//
template<class T>
int rgba4444_resize(uint16_t* rgba, int src_width, int src_height, T* &mat,
		int dst_width, int dst_height) {
	if (NULL == rgba)
		return -1;

	int nchannels = 3;
	mat = (T*) malloc(dst_width * dst_height * nchannels * sizeof(T));

	int nx = 0;
	int ny = 0;
	uint16_t p;
	for (int y = 0; y < dst_height; y++) {
		ny = (int) (1.0f * y * src_height / dst_height + 0.5);
		for (int x = 0; x < dst_width; x++) {
			nx = (int) (1.0f * x * src_width / dst_width + 0.5);

			p = rgba[ny * src_width + nx];
			mat[y * dst_width * nchannels + x * nchannels] = RGBA4444_R(p);
			mat[y * dst_width * nchannels + x * nchannels + 1] = RGBA4444_G(p);
			mat[y * dst_width * nchannels + x * nchannels + 2] = RGBA4444_B(p);

		} //for-x
	} //for-y

	return 0;
}

//
template<class T>
int bitmap_resize(JNIEnv *env, jobject bmp,
		T* &mat, int dst_width, int dst_height)
{
	int flag  = -100;
	if (bmp == NULL)
		return flag;

	//
	AndroidBitmapInfo info;
	int ret;
	if ((ret = AndroidBitmap_getInfo(env, bmp, &info)) < 0) {
		flag = -101;
		return flag;
	}
	if (info.width <= 0 || info.height <= 0
			|| (info.format != ANDROID_BITMAP_FORMAT_RGB_565
					&& info.format != ANDROID_BITMAP_FORMAT_RGBA_8888
					&& info.format != ANDROID_BITMAP_FORMAT_RGBA_4444)) {
		flag = -102;
		return flag;
	}

	// Lock the bitmap to get the buffer
	void * pixels = NULL;
	ret = AndroidBitmap_lockPixels(env, bmp, &pixels);
	if (ret < 0 || pixels == NULL)
		return ret;

	// Write the pixel back
	switch (info.format) {
	case ANDROID_BITMAP_FORMAT_RGB_565:
		flag = rgb565_resize<T>((uint16_t *) pixels, info.width, info.height, mat, dst_width, dst_height);
		break;
	case ANDROID_BITMAP_FORMAT_RGBA_8888:
		flag = rgba8888_resize<T>((uint32_t *) pixels, info.width, info.height, mat, dst_width, dst_height);
		break;
	case ANDROID_BITMAP_FORMAT_RGBA_4444:
		flag = rgba4444_resize<T>((uint16_t *) pixels, info.width, info.height, mat, dst_width, dst_height);
		break;
	} //switch

	//
	AndroidBitmap_unlockPixels(env, bmp);
	return flag;
}

#endif
