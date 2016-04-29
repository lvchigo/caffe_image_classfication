mpwd=`pwd`
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../src/API_caffe/v1.2.0/lib
export PYTHONPATH=$PYTHONPATH:$mpwd/../../src/API_caffe/v1.2.0/python/py-faster-rcnn-lib
################################Path################################
queryList='/home/chigo/image/dataset/WIDER_FACE/WIDER_val/imglist_wider_val.txt'
loadAnnotations='/home/chigo/image/dataset/WIDER_FACE/wider_face_split/wider_face_val.txt'

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
srcImgSavePath='JPEGImages/'
rm -r $SavePath$srcImgSavePath
mkdir $SavePath$srcImgSavePath
################################Path################################
imgSavePath='CheckImages/'
rm -r $SavePath$imgSavePath
mkdir $SavePath$imgSavePath
################################Path################################
Unnormal_face_Path='unnormal_face/'
rm -r $SavePath$Unnormal_face_Path
mkdir $SavePath$Unnormal_face_Path
################################DL_ImgLabel################################
#Demo_facedetect -Get_wider_FaceLabel queryList.txt loadAnnotations xmlSavePath srcImgSavePath imgSavePath
../Demo_facedetect -Get_wider_FaceLabel $queryList $loadAnnotations $SavePath$xmlSavePath $SavePath$srcImgSavePath $SavePath$imgSavePath


