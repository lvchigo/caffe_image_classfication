mpwd=`pwd`
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../../lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../../src/API_caffe/v1.2.0/lib
export PYTHONPATH=$PYTHONPATH:$mpwd/../../../src/API_caffe/v1.2.0/python/py-faster-rcnn-lib
################################Input Path################################
#voc2007
in_Annotations='/home/chigo/image/voc/VOC2007/Annotations/'
#in_Annotations='/home/chigo/working/caffe_img_classification/test/bin/mutilabel_sh/mutilabel_2_0_0_sh/test/test_voc2007/'
################################Path################################
savepath='res/'
rm -r $savepath
mkdir $savepath
Annoations='Annotations/'
mkdir $savepath$Annoations
################################Path################################
#test
loadXMLList='imglist.txt'
find $in_Annotations -name "*.xml" >$savepath$loadXMLList
################################DL_ImgLabel################################
#Demo_mutilabel -change_voclabel4test loadXMLPath svPath
../../Demo_mutilabel -change_voclabel4test $savepath$loadXMLList $savepath$Annoations

################################Path################################
rm $savepath$loadXMLList

