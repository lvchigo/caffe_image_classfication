/*
 * =====================================================================================
 *
 *       filename:  geekeyedll.cpp
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

#include "geekeyedll/geekeyedll.h"

namespace geekeyelab {

  //
  static bool predict_comp(const std::pair <int, float> elem1, const std::pair <int, float> elem2)
  {
    return (elem1.second > elem2.second);
  }

  //
  int GeekeyeDLL::init(const std::string& deploy_prototxt,
		       const std::string& mean_binaryproto,
		       const std::string& caffe_model,
		       const std::string& layer_name,
		       const int          use_gpu,
		       const int          device_id)
  {
    //
    _use_gpu = use_gpu;
    _device_id = device_id;
    int flag = init(deploy_prototxt,
		    caffe_model,
		    _use_gpu, _device_id);

    //
    caffe::BlobProto input_mean_dl;
    caffe::ReadProtoFromBinaryFile(mean_binaryproto, &input_mean_dl); 
    _mean_dl.FromProto(input_mean_dl);
    _mean_model = 1;

    //
    _layer_name = layer_name;

    //
    return flag;
  }

  //
  int GeekeyeDLL::init(const std::string& deploy_prototxt,
		       const std::string& caffe_model,
		       const int          use_gpu,
		       const int          device_id)
  {

    // set gpu device
    _use_gpu = use_gpu;
    _device_id = device_id;

    // set gpu model
    if ( 1 == _use_gpu )
    {
	     caffe::Caffe::set_mode(caffe::Caffe::GPU);
	     caffe::Caffe::SetDevice(_device_id);
      
    }
    else
    {
	     caffe::Caffe::set_mode(caffe::Caffe::CPU);
      
    }
  
  
    // data copy
    std::cout<<"reset."<<std::endl;
    _net_dl.reset(new caffe::Net<float>(deploy_prototxt, caffe::TEST));
    std::cout<<"copy."<<std::endl;
    _net_dl->CopyTrainedLayersFrom( caffe_model );
    std::cout<<deploy_prototxt<<" "<<caffe_model<<std::endl;
  
    // get net param
    std::vector<caffe::Blob<float>*> input_blobs = _net_dl->input_blobs();
    std::cout<<"input_blobs.size():"<<input_blobs.size()<<std::endl;
    if(input_blobs.size()>0){
      //std::cout<<"input blob:"<<std::cout<<input_blobs[0]->width()<<"x"<<input_blobs[0]->height()
      //	   <<"x"<<input_blobs[0]->channels()<<"x"<<input_blobs[0]->num()<<std::endl;
      _input_blob_width = input_blobs[0]->width();
      _input_blob_height = input_blobs[0]->height();
      _input_blob_channel = input_blobs[0]->channels();
      _input_blob_num = input_blobs[0]->num();
    }

    //
    check_layers();
    check_blob_names();

    //
    _mean_model = 0;
    _init_flag = true;
    _mean_value.assign(_input_blob_channel, 0.0f);
    return 0;
  }

  //
  int GeekeyeDLL::init(const std::string& deploy_prototxt,
		       const std::string& mean_binaryproto,
		       const std::string& caffe_model)
  {
    return init(deploy_prototxt,
		mean_binaryproto,
		caffe_model,
		_layer_name, _use_gpu, _device_id);
  }

  //
  void GeekeyeDLL::release()
  {
    if (_net_dl) {
      _net_dl.reset();
      _init_flag = false;
    }

  }

  //
  int GeekeyeDLL::predict(const std::string& imf, std::vector< std::pair<int, float> >& results)
  {
    int ret = -1;
    IplImage *img = cvLoadImage(imf.c_str(), 1);
    if(!img) 
    {
      return -1;
    }

    //
    ret = predict(img, results);
    cvReleaseImage(&img);
    //
    return ret;
  }

  //
  int GeekeyeDLL::predict(const IplImage* image, std::vector< std::pair<int, float> >& results)
  {
    int ret = 0;
    std::vector < IplImage* > img_dl;
    gen_one_blob(image, img_dl);
    if(img_dl.size() < 1) return -1;

    //
    predict(img_dl, results);
    
    //
    for(int i = 0; i<img_dl.size(); i++){
      if(img_dl[i] != NULL) cvReleaseImage(&(img_dl[i]));
    }

    return ret;

  }

  //
  int GeekeyeDLL::predict(const std::vector < IplImage* >& img_dl, std::vector< std::pair<int, float> >& results)
  {
    //
    if(img_dl.size() < 1) return -1;
    //std::cout<<"_mean_model="<<_mean_model<<std::endl;
    // image to blob
    caffe::Blob<float> image_blob( img_dl.size(), _input_blob_channel, _input_blob_height, _input_blob_width );
    switch(_mean_model){
    case 0:
      {
	image_to_blob_origin( img_dl, image_blob  );
	break;
      }
    case 1:
      {
	image_to_blob_mean_file( img_dl, image_blob  );
	break;
      }
    case 2:
      {
	image_to_blob_mean_value( img_dl, image_blob  );
	break;
      }
    default:
      image_to_blob_origin( img_dl, image_blob  );
    }
  
  
    // input layer
    std::vector<caffe::Blob<float>*> input_blobs = _net_dl->input_blobs();
    // image blob to input layer
    for (int i = 0; i < input_blobs.size(); ++i) {
      caffe::caffe_copy(input_blobs[i]->count(), image_blob.mutable_cpu_data(),input_blobs[i]->mutable_cpu_data());
    }

    // do forward
    float iter_loss = 0.0;
    std::vector<caffe::Blob<float>*> output_blobs = _net_dl->Forward(input_blobs, &iter_loss);
    //std::vector<caffe::Blob<float>*> output_blobs = _net_dl->ForwardPrefilled(NULL);
    //std::vector<caffe::Blob<float>*> output_blobs = _net_dl->Forward(&iter_loss);
    if(output_blobs.size() < 1){
      std::cout<<"do forward failed!"<<std::endl;
      exit(0);
    }
    _fc_output_num = output_blobs[0]->count();

    // output prob
    if(1 == output_blobs.size())
      {
	for (int k=0; k < _fc_output_num; ++k ) 
	  {
	    results.push_back( std::make_pair( k, output_blobs[0]->cpu_data()[k] ) );    
	  }
      } 
    else
      {
	// blob data to cpu
	std::vector<float> label_blob_data;
	label_blob_data.clear();
	for (int j = 0; j < output_blobs.size(); ++j) {
	  for (int k = 0; k < output_blobs[j]->count(); ++k) {
	    label_blob_data.push_back(output_blobs[j]->cpu_data()[k]);
	  }
	}
  
	//get label result init
	for (int k=0; k < _fc_output_num; ++k ) 
	  {
	    results.push_back( std::make_pair( k, label_blob_data[k] ) );    
	  }
	for (int i = 1; i < img_dl.size(); ++i ) 
	  {
	    //get label result    
	    for (int k=0; k < _fc_output_num; ++k ) 
	      {
		results[k].second += label_blob_data[k+i*_fc_output_num];		//get mean
	      }//for-k
	  }//for-i
  
	//
	for (int k = 0; k < _fc_output_num; ++k ) 
	  {
	    results[k].second /= img_dl.size() ;	//get mean 
	  }
      }//if-output-blob-size

    //sort label result
    sort(results.begin(), results.end(), predict_comp);		
  
    return 0;
  }

  //
  int GeekeyeDLL::get_layer_features(const std::vector < IplImage* >& img_dl, std::vector< std::vector<float> >& feat_all)
  {
    return get_layer_features(img_dl, _layer_name, feat_all);
  }

  //
  int GeekeyeDLL::get_layer_features(const std::vector < IplImage* >& img_dl, const std::string& layer_name, std::vector< std::vector<float> >& feat_all)
  {
    //
    if(img_dl.size() < 1){
      //LOOGE<<"img_dl size:"<<img_dl.size();
      return -1;
    }
    //
    caffe::Blob<float> image_blob( img_dl.size(), _input_blob_channel, _input_blob_height, _input_blob_width );
    switch(_mean_model){
    case 0:
      {
	image_to_blob_origin( img_dl, image_blob  );
	break;
      }
    case 1:
      {
	image_to_blob_mean_file( img_dl, image_blob  );
	break;
      }
    case 2:
      {
	image_to_blob_mean_value( img_dl, image_blob  );
	break;
      }
    default:
      image_to_blob_origin( img_dl, image_blob  );
    }
  

    //
    std::vector<caffe::Blob<float>*> input_blobs = _net_dl->input_blobs();
    for (int i = 0; i < input_blobs.size(); ++i) {
      caffe::caffe_copy(input_blobs[i]->count(), image_blob.mutable_cpu_data(),input_blobs[i]->mutable_cpu_data());
    }

    //
    float iter_loss = 0.0;
    std::vector<caffe::Blob<float>*> output_blobs = _net_dl->Forward(input_blobs, &iter_loss);

    //
    caffe::shared_ptr<caffe::Blob<float> > feature_blob;
    feature_blob = _net_dl->blob_by_name(layer_name);
    if(NULL == feature_blob){
      //LOOGE<<"feature blob null.";
      return -1;
    }

    int batch_size = feature_blob->num();
    int dim_features = feature_blob->count() / batch_size;
    //std::cout<<batch_size<<" dim sz:"<<dim_features<<" "<<std::endl;

    std::vector<float> feat_one;
    //std::vector< std::vector<float> > feat_all;
    feat_all.clear();
    const float* feature_blob_data;
    for (int i = 0; i < batch_size; ++i) {
      //feat_one.resize(dim_features);
      feature_blob_data = feature_blob->cpu_data() + feature_blob->offset(i);
      std::copy ( feature_blob_data, feature_blob_data + dim_features, std::back_inserter(feat_one));
      feat_all.push_back( feat_one );
      feat_one.clear();
    }
    std::vector<float>().swap( feat_one);

    return 0;
  }

  


  int GeekeyeDLL::image_to_blob_origin( const std::vector<IplImage* > &img_dl, caffe::Blob<float>& image_blob)
  {
    if( img_dl.size() < 1)
      return -1;
		
    int d,c,h,w;
    float pixel;
    float* blob_ptr= image_blob.mutable_cpu_data();
	
    for ( d = 0; d< img_dl.size(); d++){	
      for ( c = 0; c < _input_blob_channel; ++c) {
	for ( h = 0; h < _input_blob_height; ++h) {
	  for ( w = 0; w < _input_blob_width; ++w) {
	    pixel = (float)((uchar*)(img_dl[d]->imageData + img_dl[d]->widthStep*h))[w*3+c];
	    *blob_ptr = pixel;
	    blob_ptr++;
	  }
	}
      }
    }
	
    return 0;
  }

  int GeekeyeDLL::image_to_blob_mean_value( const std::vector<IplImage* > &img_dl, caffe::Blob<float>& image_blob)
  {
    if( img_dl.size() < 1)
      return -1;
		
    int d,c,h,w;
    float pixel;
    float* blob_ptr = image_blob.mutable_cpu_data();

    for ( d = 0; d< img_dl.size(); d++){
      for ( c = 0; c < _input_blob_channel; ++c) {
	for ( h = 0; h < _input_blob_height; ++h) {
	  for ( w = 0; w < _input_blob_width; ++w) {
	    pixel = (float)((uchar*)(img_dl[d]->imageData + img_dl[d]->widthStep*h))[w*3+c];
	    *blob_ptr = pixel - _mean_value[c];		//mean [ r g b]
	    blob_ptr++;
	  }
	}
      }
    }
	
    return 0;
  }

  //
  int GeekeyeDLL::image_to_blob_mean_file( const std::vector<IplImage* > &img_dl, caffe::Blob<float>& image_blob)
  {
    if( img_dl.size() < 1)
      return -1;
		
    int d,c,h,w;
    float pixel;
    float* blob_ptr = image_blob.mutable_cpu_data();
    float* mean;

    int w_off = int( (_input_image_width - _input_blob_width)*0.5 );
    int h_off = int( (_input_image_height - _input_blob_height)*0.5 );

    const float* tmp_mean = _mean_dl.cpu_data();
    mean = (float*)tmp_mean;
  
    for ( d = 0; d< img_dl.size(); d++){	
      for ( c = 0; c < _input_blob_channel; ++c) {
	for ( h = 0; h < _input_blob_height; ++h) {
	  for ( w = 0; w < _input_blob_width; ++w) {
	    pixel = (float)((uchar*)(img_dl[d]->imageData + img_dl[d]->widthStep*h))[w*3+c];
	    *blob_ptr = pixel - mean[(c * _input_image_height + h + h_off) * _input_image_width + w + w_off];		//mean:256*256
	    blob_ptr++;
	  }
	}
      }
    }
	
    return 0;
  }

  

  //
  void GeekeyeDLL::set_gpu(int c){
    _use_gpu = c;
  }

  //
  void GeekeyeDLL::set_device(int c){
    _device_id = c;
  }


  //
  int GeekeyeDLL::set_mean_value(const std::vector<float>& mean_value){
    if(_input_blob_channel != mean_value.size()){
      return -1;
    }
    for(int i = 0; i<mean_value.size(); i++)
      _mean_value[i] = mean_value[i];
    _mean_model = 2;
    return 0;
  }

  //
  void GeekeyeDLL::check_layers()
  {
    // 
    const std::vector<caffe::shared_ptr<caffe::Layer<float> > >& layers = _net_dl->layers();
    const std::vector<std::string>& layer_names = _net_dl->layer_names();
    const std::vector<std::string>& blob_names = _net_dl->blob_names();
    int num_layers = 0;
    {
		std::string prev_layer_name = "";
		for (unsigned int i = 0; i < layers.size(); ++i) 
		{
			std::vector<caffe::shared_ptr<caffe::Blob<float> > >& layer_blobs = layers[i]->blobs();
			if (layer_blobs.size() == 0) 
			{
				std::cout<<"layer_blobs.size() == 0, continue;"<<std::endl;
				continue;
			}
			std::cout<<"layer index:"<<i<<std::endl;
			std::cout<<"layer name:"<<layer_names[i]<<std::endl;
			std::cout<<"layer_blobs.size():"<<layer_blobs.size()<<std::endl;
			for(int b = 0; b < layer_blobs.size(); b++)
				std::cout<<layer_blobs[b]->width()<<"x"<<layer_blobs[b]->height()
				   <<"x"<<layer_blobs[b]->channels()<<"x"<<layer_blobs[b]->num()<<std::endl;
			//
			caffe::shared_ptr<caffe::Blob<float> > feature_blob = layer_blobs[0];
			std::cout<<"param:"<<feature_blob->num()<<" "<<feature_blob->count()<<std::endl;
		}//for-i
	}

  }

  //
  void GeekeyeDLL::check_blob_names(){
    const std::vector<std::string>& blob_names = _net_dl->blob_names();
    std::cout<<"blob names size:"<<blob_names.size()<<std::endl;
    for(int i = 0; i<blob_names.size(); i++){
      caffe::shared_ptr<caffe::Blob<float> > feature_blob;
      feature_blob = _net_dl->blob_by_name(blob_names[i]);
      std::cout<<"("<<i<<","<<blob_names[i]<<":"<<feature_blob->width()<<"x"<<feature_blob->height()
	       <<"x"<<feature_blob->channels()<<"x"<<feature_blob->num()<<") ";
    }
    std::cout<<std::endl;
  }



  ///
  int GeekeyeDLL::gen_one_blob(const IplImage* input_img, std::vector<IplImage* >& output_imgs)
  {

    int width = input_img->width;
    int height = input_img->height;
  
    int width_new = _input_image_width;
    int height_new = _input_image_height;

    IplImage* resize_img = NULL;
    resize_img = cvCreateImage(cvSize(_input_blob_width, _input_blob_height ), input_img->depth, input_img->nChannels);
    cvResize( input_img, resize_img );
    output_imgs.push_back(resize_img);

    return 0;
  }

///
  int GeekeyeDLL::crop_one_blob(const IplImage* input_img, std::vector<IplImage* >& output_imgs)
  {

    int min_sz = _input_image_width;
    int blob_sz = _input_blob_width;
    int width = input_img->width;
    int height = input_img->height;
  
    int width_new = width;
    int height_new = height;
    float factor = 1.0f;

    //
    factor = ( width>height)? min_sz*1.0f/height:min_sz*1.0f/width;
    width_new = (int) width*factor;
    height_new = (int) height*factor;
    
    //
    IplImage* resize_img = cvCreateImage(cvSize(width_new, height_new), input_img->depth, input_img->nChannels);
    cvResize( input_img, resize_img );

    //
    cvSetImageROI( resize_img, cvRect((width_new-blob_sz)/2, (height_new-blob_sz)/2, blob_sz, blob_sz ) );
    IplImage* roi_img = cvCreateImage(cvGetSize(resize_img), resize_img->depth, resize_img->nChannels);
    cvCopy(resize_img, roi_img, NULL );
    cvResetImageROI(resize_img);
    output_imgs.push_back(roi_img);
    cvReleaseImage(&resize_img);

    return 0;
  }

  

  int GeekeyeDLL::get_layer_params(const std::string& layer_name, std::vector< std::vector<float> >& params_all)
  {
    // 
    const std::vector<caffe::shared_ptr<caffe::Layer<float> > >& layers = _net_dl->layers();
    const std::vector<std::string>& layer_names = _net_dl->layer_names();
    const std::vector<std::string>& blob_names = _net_dl->blob_names();
    int num_layers = 0;
    {
      std::string prev_layer_name = "";
      for (unsigned int i = 0; i < layers.size(); ++i) {
	if(layer_names[i] != layer_name) continue;
	std::vector<caffe::shared_ptr<caffe::Blob<float> > >& layer_blobs = layers[i]->blobs();
	//
	caffe::shared_ptr<caffe::Blob<float> > feature_blob = layer_blobs[0];
	std::cout<<"param:"<<feature_blob->num()<<" "<<feature_blob->count()<<std::endl;
	int batch_size = feature_blob->num();
	int dim_features = feature_blob->count() / batch_size;
	std::vector<float> params_one;
	const float* feature_blob_data;
	params_all.clear();
	for (int i = 0; i < batch_size; ++i) {	  
	  feature_blob_data = feature_blob->cpu_data() + feature_blob->offset(i);
	  std::copy ( feature_blob_data, feature_blob_data + dim_features, std::back_inserter(params_one));
	  params_all.push_back( params_one );
	  params_one.clear();
    }

	
      }//for-i
    }

    return 0;
  }

  //============================================================
  //
  int GeekeyeDLL::predict_location(const IplImage* image, std::vector< std::pair<int, float> >& results, 
      const std::string& weight_layer, const std::string& activity_layer,
      std::vector<cv::Rect>& rects, cv::Mat& heatmap)
  {

    int ret = 0;
    std::vector < IplImage* > img_dl;
    gen_one_blob(image, img_dl);
    if(img_dl.size() < 1) return -1;

    //
    predict_location(img_dl, results, weight_layer, activity_layer, rects, heatmap);
    for(int i = 0; i<rects.size(); i++){
      rects[i].x = rects[i].x * image->width/256;
      rects[i].y = rects[i].y * image->height/256;
      rects[i].width = rects[i].width * image->width/256;
      rects[i].height = rects[i].height * image->height/256;
    }

    //
    for(int i = 0; i<img_dl.size(); i++){
      if(img_dl[i] != NULL) cvReleaseImage(&(img_dl[i]));
    }

    return ret;

  }

  //
  int GeekeyeDLL::predict_location(const std::vector < IplImage* >& img_dl, std::vector< std::pair<int, float> >& results, 
      const std::string& weight_layer, const std::string& activity_layer,
      std::vector<cv::Rect>& rects, cv::Mat& heatmap)
  {

    //
    if(img_dl.size() < 1) return -1;
    //std::cout<<"_mean_model="<<_mean_model<<std::endl;
    // image to blob
    caffe::Blob<float> image_blob( img_dl.size(), _input_blob_channel, _input_blob_height, _input_blob_width );
    switch(_mean_model){
    case 0:
      {
  image_to_blob_origin( img_dl, image_blob  );
  break;
      }
    case 1:
      {
  image_to_blob_mean_file( img_dl, image_blob  );
  break;
      }
    case 2:
      {
  image_to_blob_mean_value( img_dl, image_blob  );
  break;
      }
    default:
      image_to_blob_origin( img_dl, image_blob  );
    }
  
  
    // input layer
    std::vector<caffe::Blob<float>*> input_blobs = _net_dl->input_blobs();
    // image blob to input layer
    for (int i = 0; i < input_blobs.size(); ++i) {
      caffe::caffe_copy(input_blobs[i]->count(), image_blob.mutable_cpu_data(),input_blobs[i]->mutable_cpu_data());
    }

    // do forward
    float iter_loss = 0.0;
    std::vector<caffe::Blob<float>*> output_blobs = _net_dl->Forward(input_blobs, &iter_loss);
    //std::vector<caffe::Blob<float>*> output_blobs = _net_dl->ForwardPrefilled(NULL);
    //std::vector<caffe::Blob<float>*> output_blobs = _net_dl->Forward(&iter_loss);
    if(output_blobs.size() < 1){
      std::cout<<"do forward failed!"<<std::endl;
      exit(0);
    }
    _fc_output_num = output_blobs[0]->count();

    // output prob
    if(1 == output_blobs.size())
      {
  for (int k=0; k < _fc_output_num; ++k ) 
    {
      results.push_back( std::make_pair( k, output_blobs[0]->cpu_data()[k] ) );   
      //std::cout<<"<"<<k<<","<<output_blobs[0]->cpu_data()[k]<<"> ";
    }
      //std::cout<<std::endl;
      } 
    else
      {
  // blob data to cpu
  std::vector<float> label_blob_data;
  label_blob_data.clear();
  for (int j = 0; j < output_blobs.size(); ++j) {
    for (int k = 0; k < output_blobs[j]->count(); ++k) {
      label_blob_data.push_back(output_blobs[j]->cpu_data()[k]);
    }
  }
  
  //get label result init
  for (int k=0; k < _fc_output_num; ++k ) 
    {
      results.push_back( std::make_pair( k, label_blob_data[k] ) );    
    }
  for (int i = 1; i < img_dl.size(); ++i ) 
    {
      //get label result    
      for (int k=0; k < _fc_output_num; ++k ) 
        {
    results[k].second += label_blob_data[k+i*_fc_output_num];   //get mean
        }//for-k
    }//for-i
  
  //
  for (int k = 0; k < _fc_output_num; ++k ) 
    {
      results[k].second /= img_dl.size() ;  //get mean 
    }
      }//if-output-blob-size

    //sort label result
    sort(results.begin(), results.end(), predict_comp);   
    //std::cout<<"forward time:"<<rt.time()<<std::endl;
    int idx = results[0].first;

    // weights LR
    const std::vector<caffe::shared_ptr<caffe::Blob<float> > >& param_blobs = _net_dl->layer_by_name(weight_layer)->blobs();
    //std::cout<<param_blobs[0]->width()<<"x"<<param_blobs[0]->height()
    //     <<"x"<<param_blobs[0]->channels()<<"x"<<param_blobs[0]->num()<<"="<<param_blobs[0]->count()<<std::endl;

    std::vector<float> weights_LR;
    const float* ptr = param_blobs[0]->cpu_data() + param_blobs[0]->offset(idx);
    std::copy ( ptr, ptr+param_blobs[0]->channels(), std::back_inserter(weights_LR));
    cv::Mat weights = cv::Mat(1, param_blobs[0]->channels(),  CV_32FC1, &weights_LR[0]);
    //std::cout<<weights.cols<<"x"<<weights.rows<<std::endl;

    // activation last conv
    caffe::shared_ptr<caffe::Blob<float> > feature_blob;
    feature_blob = _net_dl->blob_by_name(activity_layer);
    //std::cout<<feature_blob->width()<<"x"<<feature_blob->height()
    //     <<"x"<<feature_blob->channels()<<"x"<<feature_blob->num()<<std::endl;


    cv::Mat activation = cv::Mat(feature_blob->channels(), feature_blob->width()*feature_blob->height(), CV_32FC1, (float*)feature_blob->cpu_data() );
    //std::cout<<activation.cols<<"x"<<activation.rows<<std::endl;

    cv::Mat map = weights*activation;
    //std::cout<<map.cols<<"x"<<map.rows<<std::endl;

    //                                                                                                                                                                            
    cv::Mat map_a = cv::Mat(feature_blob->width(), feature_blob->height(), CV_32FC1, map.data);
    //std::cout<<map_a.cols<<"x"<<map_a.rows<<std::endl;
    cv::resize(map_a, heatmap, cv::Size(256, 256), 0, 0, CV_INTER_LINEAR);
    //cv::imwrite("heatmap.jpg", heatmap);

    //
    cv::Mat map_c;
    double minVal, maxVal;
    cv::minMaxLoc(heatmap, &minVal, &maxVal);
    cv::threshold(heatmap, map_c, maxVal*.5, maxVal, CV_THRESH_BINARY);
    //cv::imwrite("heatmap-bw.jpg", map_c);
    cv::Mat map_d;
    map_c.convertTo(map_d, CV_8UC1);
    std::vector<std::vector<cv::Point>> contours ;  
    cv::findContours(map_d , contours ,   
        CV_RETR_EXTERNAL , CV_CHAIN_APPROX_NONE) ;  
    cv::Mat result(map_c.size() , CV_8U , cv::Scalar(255)) ;  
    cv::drawContours(result , contours ,  
        -1 , cv::Scalar(0) , 2) ; 
    for(int i = 0; i<contours.size(); i++){
      cv::Rect r = cv::boundingRect(cv::Mat(contours[i]));  
      //cv::rectangle(result, r,cv::Scalar(0),2); 
      rects.push_back(r);
    }
    //cv::imwrite("heatmap-c.jpg", result);
    //std::cout<<"detector runtime:"<<rt.time()<<std::endl;
    //


    // 
    return 0;
  }

}//namespace
