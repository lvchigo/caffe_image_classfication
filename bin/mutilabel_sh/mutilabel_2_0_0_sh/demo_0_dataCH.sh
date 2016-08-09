mpwd=`pwd`
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../../lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../../src/API_caffe/v1.2.0/lib
export PYTHONPATH=$PYTHONPATH:$mpwd/../../../src/API_caffe/v1.2.0/python/py-faster-rcnn-lib
################################Input Path################################
#voc2007
#in_Annotations='/home/chigo/image/mutilabel/img_src/VOC/Annotations_voc0712/' 
#in_Images='/home/chigo/image/mutilabel/img_src/VOC/JPEGImages_voc0712/'

#coco
in_Annotations='/home/chigo/working/caffe_img_classification/test/bin/mutilabel_sh/mutilabel_2_0_0_sh/coco_tmp/Annotations/' 
in_Images='/home/chigo/working/caffe_img_classification/test/bin/mutilabel_sh/mutilabel_2_0_0_sh/coco_tmp/JPEGImages/'

#oldinlabel
#in_Annotations='/home/chigo/image/mutilabel/IN_47class_300_20151230/Annotations/' 
#in_Images='/home/chigo/image/mutilabel/IN_47class_300_20151230/JPEGImages/'

#dict
#in_dict='dict/dict_voc.txt'
in_dict='dict/dict_coco.txt'
#in_dict='dict/dict_oldinlabel.txt'
out_dict='dict/dict_mutilabel.txt'
################################Path################################
savepath='res/'
rm -r $savepath
mkdir $savepath
################################Path################################
path_xml_save='Annotations/'
rm -r $savepath$path_xml_save
mkdir $savepath$path_xml_save
################################Path################################
#test
loadXMLList='imglist.txt'
find $in_Annotations -name "*.xml" >$savepath$loadXMLList

################################DL_ImgLabel################################
#inLabelClass:0-voc,1-coco,2-old in;
#Demo_mutilabel -dataCH2InMutiLabel loadXMLPath loadImagePath inDict outDict svXml inLabelClass
../../Demo_mutilabel -dataCH2InMutiLabel $savepath$loadXMLList $in_Images $in_dict $out_dict $savepath$path_xml_save 1

################################Path################################
rm $savepath$loadXMLList

