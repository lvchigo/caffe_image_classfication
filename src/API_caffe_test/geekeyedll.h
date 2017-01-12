/*
 * =====================================================================================
 *
 *       filename:  geekeyedll.h
 *
 *    description:  caffe interface 
 *
 *        version:  1.0
 *        created:  2016-01-23
 *       revision:  none
 *       compiler:  g++
 *
 *         author:  tangyuan
 *        company:  in66.com
 *
 *      copyright:  2016 itugo Inc. All Rights Reserved.
 *      
 * =====================================================================================
 */

#ifndef IN_GEEKEYE_DLL
#define IN_GEEKEYE_DLL

#include <iostream>
#include <string>
#include <vector>

#include <cstdlib>
#include <cstdio>

//#include <opencv/cv.h>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/highgui/highgui.hpp>
#include "cv.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

// caffe-common
#include "caffe/caffe.hpp"

namespace geekeyelab{

  static const int INPUT_IMAGE_WIDTH = 256;//328;
  static const int INPUT_IMAGE_HEIGHT = 256;//328;
  static const int INPUT_BLOB_WIDTH = 224;//224 227
  static const int INPUT_BLOB_HEIGHT = 224;//224 227
  static const int INPUT_BLOB_CHANNEL = 3;
  static const int INPUT_BLOB_NUM = 1;
  static const int OUTPUT_NUM = 1000;


  /// @brief geekeye deep learning library
  class GeekeyeDLL
  {

  public:
    //
  GeekeyeDLL():_use_gpu(1),_device_id(0),_layer_name("fc7"),_init_flag(false),
      _input_blob_width(INPUT_BLOB_WIDTH),_input_blob_height(INPUT_BLOB_HEIGHT),
      _input_blob_channel(INPUT_BLOB_CHANNEL),_input_blob_num(INPUT_BLOB_NUM),_fc_output_num(OUTPUT_NUM),
      _input_image_width(INPUT_IMAGE_WIDTH), _input_image_height(INPUT_IMAGE_HEIGHT)
    {
    
    }

  
    /// @brief
    int init(const std::string& deploy_prototxt,
	     const std::string& caffe_model,
	     const int          use_gpu,
	     const int          device_id);
    /// @brief
    int init(const std::string& deploy_prototxt,
	     const std::string& mean_binaryproto,
	     const std::string& caffe_model,
	     const std::string& layer_name,
	     const int          use_gpu,
	     const int          device_id);
    /// @brief
    int init(const std::string& deploy_prototxt,
	     const std::string& mean_binaryproto,
	     const std::string& caffe_model);
    /// @brief
    void release();

    ///
    int predict_location(const IplImage* image, std::vector< std::pair<int, float> >& results, 
      const std::string& weight_layer, const std::string& activity_layer,
      std::vector<cv::Rect>& rects, cv::Mat& heatmap);
    int predict_location(const std::vector < IplImage* >& img_dl, std::vector< std::pair<int, float> >& results, 
      const std::string& weight_layer, const std::string& activity_layer,
      std::vector<cv::Rect>& rects, cv::Mat& heatmap);

    /// @brief
    int predict(const std::string& imf, std::vector< std::pair<int, float> >& results);
    int predict(const IplImage* image, std::vector< std::pair<int, float> >& results);
    int predict(const std::vector < IplImage* >& img_dl, std::vector< std::pair<int, float> >& results);
    /// @brief
    int get_layer_features(const std::vector<IplImage* >& img_dl, std::vector< std::vector<float> >& feat_all);  
    int get_layer_features(const std::vector<IplImage* >& img_dl, const std::string& layer_name, std::vector< std::vector<float> >& feat_all);
    /// @brief
    int gen_one_blob(const IplImage* input_img, std::vector<IplImage* >& output_imgs);
    int crop_one_blob(const IplImage* input_img, std::vector<IplImage* >& output_imgs);
    ///
    int get_layer_params(const std::string& layer_name, std::vector< std::vector<float> >& params_all);

    /// @brief
    void set_gpu(int c);
    void set_device(int c);
    int set_mean_value(const std::vector<float>& mean_value);
    void check_layers();
    void check_blob_names();

  private:

    /// @brief
    int image_to_blob_origin( const std::vector<IplImage* > &img_dl, caffe::Blob<float>& image_blob);
    /// @brief
    int image_to_blob_mean_value( const std::vector<IplImage* > &img_dl, caffe::Blob<float>& image_blob);
    /// @brief
    int image_to_blob_mean_file( const std::vector<IplImage* > &img_dl, caffe::Blob<float>& image_blob);

 
  private:
  
    // global var
    int _use_gpu;
    int _device_id;
    std::string _layer_name;
    bool _init_flag;

    // net
    caffe::shared_ptr<caffe::Net<float> > _net_dl;
    caffe::Blob<float> _mean_dl;
    int _mean_model;
    std::vector<float> _mean_value;

  public:
    // net param
    int _input_blob_width;
    int _input_blob_height;
    int _input_blob_channel;
    int _input_blob_num;
    int _fc_output_num;
    int _input_image_width;
    int _input_image_height;

  
  };//geekeye dll
}
#endif
