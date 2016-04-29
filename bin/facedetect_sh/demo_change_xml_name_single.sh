mpwd=`pwd`
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../src/API_caffe/v1.2.0/lib
export PYTHONPATH=$PYTHONPATH:$mpwd/../../src/API_caffe/v1.2.0/python/py-faster-rcnn-lib
################################Path################################
path_xml='/home/chigo/image/noface_friend_5k/Annoation/'
path_img='/home/chigo/image/noface_friend_5k/img/'

find $path_xml/ -name "*.xml" >list_xml.txt
################################Path################################
savepath='res/'
rm -r $savepath
mkdir $savepath
################################Path################################
path_xml_save='Annotations/'
rm -r $savepath$path_xml_save
mkdir $savepath$path_xml_save
################################Path################################
path_img_save='JPEGImages/'
rm -r $savepath$path_img_save
mkdir $savepath$path_img_save
################################DL_ImgLabel################################
#Demo_facedetect -ch_xml_name queryList.txt imgPath xmlSavePath imgSavePath
../Demo_facedetect -ch_xml_name list_xml.txt $path_img $savepath$path_xml_save $savepath$path_img_save

rm list_xml.txt

