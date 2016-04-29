mpwd=`pwd`
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../src/API_caffe/v1.2.0/lib
export PYTHONPATH=$PYTHONPATH:$mpwd/../../src/API_caffe/v1.2.0/python/py-faster-rcnn-lib
################################Path################################
path_img='data/face_mutipeople_20160316/img_head10/'
loadAnnotations='data/face_mutipeople_20160316/sv_point_head10.txt'

################################Path################################
SavePath='res/'
rm -r $SavePath
mkdir $SavePath
############################trainval.txt################################
rm $SavePath/trainval.txt
################################Path################################
xmlSavePath='Annotations/'
rm -r $SavePath$xmlSavePath
mkdir $SavePath$xmlSavePath
################################Path################################
imgSavePath='JPEGImages/'
rm -r $SavePath$imgSavePath
mkdir $SavePath$imgSavePath
################################Path################################
Unnormal_face_Path='unnormal_face/'
rm -r $SavePath$Unnormal_face_Path
mkdir $SavePath$Unnormal_face_Path
################################DL_ImgLabel################################
#Demo_facedetect -check_sencetime imgPath loadAnnotations xmlSavePath imgSavePath
../Demo_facedetect -check_sencetime $path_img $loadAnnotations $SavePath$xmlSavePath $SavePath$imgSavePath


