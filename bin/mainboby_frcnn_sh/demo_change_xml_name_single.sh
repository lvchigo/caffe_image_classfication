mpwd=`pwd`
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../src/API_caffe/v1.1.0/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../src/API_similardetect/lib
################################Path################################
path_xml='/home/chigo/working/research/Bing/Objectness-master/IN2016_Test/Annotations/'
path_img='/home/chigo/working/research/Bing/Objectness-master/IN2016_Test/JPEGImages/'

find $path_xml/ -name "*.xml" >list_xml.txt
################################Path################################
path_xml_save='Annotations/'
rm -r $path_xml_save
mkdir $path_xml_save
################################Path################################
path_img_save='JPEGImages/'
rm -r $path_img_save
mkdir $path_img_save
################################DL_ImgLabel################################
#Demo_mainboby_frcnn -ch_xml_name queryList.txt imgPath xmlSavePath imgSavePath
../Demo_mainboby_frcnn -ch_xml_name list_xml.txt $path_img $path_xml_save $path_img_save

rm list_xml.txt

