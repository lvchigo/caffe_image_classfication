mpwd=`pwd`
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../src/API_caffe/v1.1.0/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../src/API_similardetect/lib

################################Path################################
savepath='res_Hypothese/'

rm -r $savepath
mkdir $savepath

################################Path################################
#test
path='/home/chigo/working/research/fast-rcnn-master/data/VOCdevkit2000/VOC2000/JPEGImages/'

find $path/ -name "*.jpg" >list_jpg.txt

################################DL_ImgLabel################################
#Demo_mainboby -get_bing_roi queryList.txt keyFilePath layerName binGPU deviceID
../Demo_mainboby_frcnn -get_bing_roi list_jpg.txt ../../keyfile/ "fc7" 1 0

rm list_jpg.txt

