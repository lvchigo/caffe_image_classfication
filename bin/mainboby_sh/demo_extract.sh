mpwd=`pwd`
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../src/API_caffe/v1.0.0/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../src/API_similardetect/lib

################################Path################################
loadImgList='/home/chigo/image/mainboby/mainboby_1_0_0_20160106/img_90class_train_head.txt'
#loadImgList='../imglist/list_test0313_10k_10_train.txt'
#loadImgList='train_ads6class.txt'


svFeat='train_feat_151127'
keyfile='../../keyfile/'
layerName='fc7'
################################DL_ImgLabel################################
#Demo_mainboby -extract queryList.txt szFeat keyFilePath layerName binGPU deviceID
../Demo_mainboby -extract $loadImgList $svFeat $keyfile $layerName 1 0



