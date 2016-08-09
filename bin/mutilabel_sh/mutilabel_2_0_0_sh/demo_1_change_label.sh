mpwd=`pwd`
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../../lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../../src/API_caffe/v1.2.0/lib
export PYTHONPATH=$PYTHONPATH:$mpwd/../../../src/API_caffe/v1.2.0/python/py-faster-rcnn-lib
################################Input Path################################
#voc2007
#in_Annotations='/home/chigo/image/mutilabel/pretrain_data/res_voc0712/Annotations/' 
#in_Images='/home/chigo/image/mutilabel/pretrain_data/res_voc0712/JPEGImages/'

#coco
#in_Annotations='/home/chigo/working/caffe_img_classification/test/bin/mutilabel_sh/mutilabel_2_0_0_sh/coco_tmp/res_coco_tmp/Annotations/' 
#in_Images='/home/chigo/working/caffe_img_classification/test/bin/mutilabel_sh/mutilabel_2_0_0_sh/coco_tmp/JPEGImages/'

#oldinlabel
#in_Annotations='/home/chigo/image/mutilabel/pretrain_data/res_oldindata/Annotations/' 
#in_Images='/home/chigo/image/mutilabel/pretrain_data/res_oldindata/JPEGImages/'

#new add label
in_Annotations='/home/chigo/image/mutilabel/pretrain_data/new_add_tangbao/mutilabel_food/Annotations/'
in_Images='/home/chigo/image/mutilabel/pretrain_data/new_add_tangbao/mutilabel_food/JPEGImages/'
################################Path################################
savepath='res/'
rm -r $savepath
mkdir $savepath

Annoations='Annoations/'
rm -r $savepath$Annoations
mkdir $savepath$Annoations
################################Path################################
#test
loadXMLList='imglist.txt'
find $in_Annotations -name "*.xml" >$savepath$loadXMLList
################################DL_ImgLabel################################
#Demo_mutilabel -change_label loadXMLPath loadImagePath svPath
../../Demo_mutilabel -change_label $savepath$loadXMLList $in_Images $savepath

################################Path################################
rm $savepath$loadXMLList

