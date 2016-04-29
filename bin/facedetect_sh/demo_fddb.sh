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
loadImgList='/home/chigo/working/face/fddb/FDDB-folds/FDDB-fold.txt'
queryPath='/home/chigo/working/face/fddb/originalPics/'

################################DL_ImgLabel################################
#Demo_facedetect -fddb queryList.txt queryPath keyFilePath layerName binGPU deviceID
../Demo_facedetect -fddb $loadImgList $queryPath ../../keyfile/ "fc7" 1 0

################################result 2 html################################
#./showDemo.py
#firefox showDemo.html
