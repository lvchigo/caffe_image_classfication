mpwd=`pwd`
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../src/API_caffe/v1.1.0/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../src/API_similardetect/lib


################################Path################################
xmlSavePath='Annotations/'
rm -r $xmlSavePath
mkdir $xmlSavePath

################################Path################################
#test
loadImgList='/home/chigo/working/caffe_img_classification/test/bin/mainboby_frcnn_sh/xml/xml.txt'

################################DL_ImgLabel################################
#Demo_mainboby_frcnn -get_imagenetxml2xml queryList.txt xmlSavePath
../Demo_mainboby_frcnn -get_imagenetxml2xml $loadImgList $xmlSavePath

