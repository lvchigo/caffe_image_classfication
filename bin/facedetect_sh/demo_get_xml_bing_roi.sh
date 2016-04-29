mpwd=`pwd`
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../src/API_caffe/v1.1.0/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../src/API_proposal/BINGpp/lib
################################Path################################
savepath='res'
rm -r $savepath
mkdir $savepath
################################Path################################
#test
path='/home/chigo/working/caffe_img_classification/test/bin/facedetect_sh/data/face_mutipeople_20160316/img_head10/'
path_xml='/home/chigo/working/caffe_img_classification/test/bin/facedetect_sh/data/face_mutipeople_20160316/res/Annotations/'

find $path/ -name "*.jpg" >list_jpg.txt
################################DL_ImgLabel################################
#Demo_facedetect -get_xml_bing_roi queryList.txt loadXmlPath keyFilePath layerName binGPU deviceID
../Demo_facedetect -get_xml_bing_roi list_jpg.txt $path_xml ../../keyfile/ "fc7" 1 0

rm list_jpg.txt

