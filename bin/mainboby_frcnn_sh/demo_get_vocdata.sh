mpwd=`pwd`
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../src/API_caffe/v1.1.0/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../src/API_similardetect/lib
################################Path################################
path_list='/home/chigo/image/mainboby/frcnn_DATA/VOC/JPEGImages_voc0712_in/list_voc0712_in.txt'
path_xml='/home/chigo/image/mainboby/frcnn_DATA/VOC/Annotations_voc0712/'
path_img='/home/chigo/image/mainboby/frcnn_DATA/VOC/JPEGImages_voc0712/'
################################Path################################
path_xml_save='Annotations/'
rm -r $path_xml_save
mkdir $path_xml_save
################################Path################################
path_img_save='JPEGImages/'
rm -r $path_img_save
mkdir $path_img_save
################################DL_ImgLabel################################
#Demo_mainboby_frcnn -get_vocdata queryList.txt inImgPath inXmlPath imgSavePath xmlSavePath
../Demo_mainboby_frcnn -get_vocdata $path_list $path_img $path_xml $path_img_save $path_xml_save


