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
loadImgList='../imglist/list_test0313_10k_100.txt'			#onlime_img_1w
#loadImgList='imglist/image_73class_test_1w_100.txt'			#onlime_img_1w
#loadImgList='imglist/image_voc2007_1w_100.txt'

################################DL_ImgLabel################################
#Demo_facedetect -frcnn queryList.txt keyFilePath layerName binGPU deviceID
../Demo_facedetect -frcnn $loadImgList ../../keyfile/ "fc7" 1 0

#Demo_facedetect -frcnn_addsample queryList.txt keyFilePath layerName binGPU deviceID
#../Demo_facedetect -frcnn_addsample $loadImgList ../../keyfile/ "fc7" 1 0

################################result 2 html################################
#./showDemo.py
#firefox showDemo.html
