
#include "com_jiuyan_infashion_inimagecolor_InImageColorNativeLibrary.h"
#include "in_image_color.h"

#include <android/log.h>
#define LOG(...) ((void)__android_log_print(ANDROID_LOG_WARN, "native-activity", __VA_ARGS__))

#include <android/bitmap.h>


JNIEXPORT jint JNICALL Java_com_jiuyan_infashion_inimagecolor_InImageColorNativeLibrary_MainColorByARGB
  (JNIEnv *env, jobject job, jintArray src, jint w, jint h)
{

	int maincolor = 0;
	jint *argb = env->GetIntArrayElements(src, 0);
	int len = env->GetArrayLength (src);
	//__android_log_print(ANDROID_LOG_INFO, "color", "len=%d",len);

	//
	//maincolor = main_color(argb, w, h);

	// release
	env->ReleaseIntArrayElements(src, argb, 0);

	return maincolor;
}

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
JNIEXPORT jint JNICALL Java_com_jiuyan_infashion_inimagecolor_InImageColorNativeLibrary_MainColorFromBitmap
  (JNIEnv *env, jobject job, jobject bmp, jfloat lightness)
{
	int maincolor = 0;
	float value = (lightness < 0.1 || lightness > 1.0)? 0.8:lightness;

	//
	AndroidBitmapInfo  info;
	int ret;
	if ((ret = AndroidBitmap_getInfo(env, bmp, &info)) < 0)
	{
		maincolor = (128 & 0xFF) | ((128 & 0xFF) << 8) | ((128 & 0xFF) << 16) | ((255 & 0xFF) << 24);
		return maincolor;
	}
	if(1 != info.format)
	{
		maincolor = (128 & 0xFF) | ((128 & 0xFF) << 8) | ((128 & 0xFF) << 16) | ((255 & 0xFF) << 24);
		return maincolor;
	}

	//
	int *bgra = NULL;
	if ((ret = AndroidBitmap_lockPixels(env, bmp, (void**)&bgra)) < 0)
	{
		maincolor = (128 & 0xFF) | ((128 & 0xFF) << 8) | ((128 & 0xFF) << 16) | ((255 & 0xFF) << 24);
		return maincolor;
	}

	//
	//maincolor = main_color(bgra, info.width, info.height);
	maincolor = main_color_from_bgra(bgra, info.width, info.height, value);

	//
	AndroidBitmap_unlockPixels(env, bmp) ;
	bgra = NULL;

	return maincolor;
}
