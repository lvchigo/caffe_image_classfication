mpwd=`pwd`
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../input/lib

################################Path################################
loadImgList='/home/chigo/working/caffe_img_classification/test/bin/imglist/img_test/list_test_6_train.txt'
#loadImgList='../imglist/list_test0313_10k_10_train.txt'
#loadImgList='train_ads6class.txt'


svFeat='train_feat_imagequalityblur2class_050626'
keyfile='../../keyfile/'
layerName='fc7'
################################DL_ImgLabel################################
#svMode:1-in73class,2-in6class,3-ads6class,4-imagequality,5-imagequality blur;
#Demo_online_classification -extract queryList.txt szFeat keyFilePath layerName binGPU deviceID svMode
../Demo_online_classification -extract $loadImgList $svFeat $keyfile $layerName 1 0 5



