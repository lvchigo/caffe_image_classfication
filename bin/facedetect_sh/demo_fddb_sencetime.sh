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
loadAnnotations='Model_Res/scenetime_fddb/fddb-img-ret.csv'
#loadAnnotations='Model_Res/scenetime_fddb/fddb-img-ret_head10.txt'
szQueryList='Model_Res/scenetime_fddb/FDDB-fold.txt'
queryPath='/home/chigo/working/face/fddb/originalPics/'

################################DL_ImgLabel################################
#Demo_facedetect -fddb_sencetime loadAnnotations szQueryList queryPath
../Demo_facedetect -fddb_sencetime $loadAnnotations $szQueryList $queryPath

################################result 2 html################################
#./showDemo.py
#firefox showDemo.html
