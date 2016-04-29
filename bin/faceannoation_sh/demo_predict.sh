mpwd=`pwd`
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../src/API_caffe/v1.2.0/lib
export PYTHONPATH=$PYTHONPATH:$mpwd/../../src/API_caffe/v1.2.0/python/py-faster-rcnn-lib
################################Path################################
savepath='res_predict/'
rm -r $savepath
mkdir $savepath
################################Path################################
facepath='face/'
rm -r $savepath$facepath
mkdir $savepath$facepath
################################Path################################
nofacepath='noface/'
rm -r $savepath$nofacepath
mkdir $savepath$nofacepath
################################Path################################
roipath='roi/'
rm -r $savepath$roipath
mkdir $savepath$roipath
################################Path################################
#test
loadImgList='../imglist/list_test0313_10k_2.txt'
#loadImgList='Model_Res/res_predict_1w_0404/ChangeName.txt'
#loadImgList='/home/chigo/working/caffe_img_classification/test/bin/downloadimg_sh/shuying_20160420_test/imglist.txt'

################################DL_ImgLabel################################
#Demo_face_annoation -frcnn queryList.txt keyFilePath layerName binGPU deviceID
../Demo_face_annoation -frcnn $loadImgList ../../keyfile/ "fc7" 1 0

