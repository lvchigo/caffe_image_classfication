mpwd=`pwd`
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../src/API_caffe/v1.2.0/lib
export PYTHONPATH=$PYTHONPATH:$mpwd/../../src/API_caffe/v1.2.0/python/py-faster-rcnn-lib
################################Path################################
#test
szQueryList='Model_Res/res_predict_1w_0404/imglist.txt'
queryPath='/home/chigo/image/test/test0313_1w/'

################################DL_ImgLabel################################
#Demo_facedetect -change_name szQueryList queryPath
../Demo_facedetect -change_name $szQueryList $queryPath

