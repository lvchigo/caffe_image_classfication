mpwd=`pwd`
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../src/API_caffe/v1.1.0/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../src/API_similardetect/lib
################################Path################################
path_xml='/home/chigo/image/mainboby/frcnn_DATA/VOC/Annotations_voc0712/'
path_img='/home/chigo/image/mainboby/frcnn_DATA/VOC/JPEGImages_voc0712/'

find $path_xml/ -name "*.xml" >list_xml.txt
################################Path################################
xmlSavePath='Annotations/'
rm -r $xmlSavePath
mkdir $xmlSavePath
################################Path################################
imgSavePath='JPEGImages/'
rm -r $imgSavePath
mkdir $imgSavePath
################################DL_ImgLabel################################
#Demo_mainboby_frcnn -get_voc2xml queryList.txt imgPath xmlSavePath imgSavePath maxNum
../Demo_mainboby_frcnn -get_voc2xml list_xml.txt $path_img $xmlSavePath $imgSavePath 5000

rm list_xml.txt

