

#include "com_jiuyan_infashion_geekeye_InGeekeyeNativeLibrary.h"

//#include <android/log.h>
//#define LOG(...) ((void)__android_log_print(ANDROID_LOG_WARN, "native-activity", __VA_ARGS__))

#include <android/bitmap.h>

#include "in_geekeye.h"
#include "in_signa_verify.h"
#include "in_bitmap_processor.h"
#include "similar/SD_global.h"
///
JNIEXPORT jfloat JNICALL Java_com_jiuyan_infashion_geekeye_InGeekeyeNativeLibrary_ClassifyFromBitmap
  (JNIEnv *env, jclass cls, jobject bmp)
{
	float flag = 0;

	//if(!Verify(env, cls)) return -66;

	//
	uint8_t* lbp_img_dat = NULL;
	int idx = -1;
	float prob = -1.0f;
	int lbp_size = 0;
	flag = bitmap_resize_to_gray_mat<uint8_t>(env, bmp, lbp_img_dat,
			MAX_LBP_SIZE, MAX_LBP_SIZE);
	double* lbp_desc = mblbp_from_mat8(lbp_img_dat, MAX_LBP_SIZE, MAX_LBP_SIZE,
			lbp_size);
	//
	//flag = linearsvm_predict_mblbp(lbp_desc, lbp_size, idx, prob);
	//flag = linearsvm_predict_mblbp_v2(lbp_desc, lbp_size, idx, prob);
	flag = linearsvm_predict_mblbp_c15(lbp_desc, lbp_size, idx, prob);
	//
    flag =  (flag == 0)? idx + prob:flag;
    free(lbp_img_dat);
    free(lbp_desc);

	return flag;
}

///
JNIEXPORT jfloat JNICALL Java_com_jiuyan_infashion_geekeye_InGeekeyeNativeLibrary_SimilarFromBitmap
  (JNIEnv *env, jclass cls, jobject bmp_from, jobject bmp_to)
{
	float flag = 0;
	int ret = 0;
	//if(!Verify(env, cls)) return -66;
	//
	int w = 256;
	int h = 256;
	int c = 3;
	//
	IN_IMAGE_SIMILAR_DETECT_1_0_0 similar;
    uint8_t* img_from = NULL;
    uint8_t* feat_from = new uint8_t[ALL_FEAT_DIM];
    float quality_from = 0.0f;
    flag = bitmap_resize<uint8_t>(env, bmp_from, img_from, w, h);
    ret = similar.Get_Feat_Score(img_from, w, h, c, feat_from, quality_from);
    free(img_from);
    //
	uint8_t* img_to = NULL;
	uint8_t* feat_to = new uint8_t[ALL_FEAT_DIM];
	float quality_to = 0.0f;
	flag = bitmap_resize<uint8_t>(env, bmp_to, img_to, w, h);
	ret = similar.Get_Feat_Score(img_to, w, h, c, feat_to, quality_to);
	free(img_to);
	//
	int model = 0;
	ret = similar.SimilarDetect(feat_from, feat_to, model);
	flag = model;
	delete [] feat_from;
	delete [] feat_to;
    //
    return flag;
}
