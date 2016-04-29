mpwd=`pwd`
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../src/API_caffe/v1.2.0/lib
export PYTHONPATH=$PYTHONPATH:$mpwd/../../src/API_caffe/v1.2.0/python/py-faster-rcnn-lib
################################Path################################
savepath='res_predict/'
rm -r $savepath
mkdir $savepath
################################Path################################
facepath='face/'
rm -r $savepath$facepath
mkdir $savepath$facepath
################################Path################################
nofacepath='noface/'
rm -r $savepath$nofacepath
mkdir $savepath$nofacepath
################################Path################################
#test
loadAnnotations='Model_Res/sensetime_1w/test0313_1w-ret2.csv'
#loadAnnotations='Model_Res/scenetime_fddb/fddb-img-ret_head10.txt'
szQueryList='../imglist/list_test0313_10k.txt'
queryPath='/home/chigo/image/test/test0313_1w/'

################################DL_ImgLabel################################
#Demo_facedetect -sencetime_in loadAnnotations szQueryList queryPath
../Demo_facedetect -sencetime_in $loadAnnotations $szQueryList $queryPath

################################result 2 html################################
#./showDemo.py
#firefox showDemo.html
