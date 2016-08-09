mpwd=`pwd`
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../../lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../../src/API_caffe/v1.2.0/lib
export PYTHONPATH=$PYTHONPATH:$mpwd/../../../src/API_caffe/v1.2.0/python/py-faster-rcnn-lib
################################Input Path################################
#new add label 
in_Images='/home/chigo/image/test/list_test0313_100.txt'
#in_Images='/home/chigo/image/logo/list.txt'
keyfile='../../../keyfile/'
################################Path################################
savepath='res/'
rm -r $savepath
mkdir $savepath
imgpath='img/'
mkdir $savepath$imgpath
################################DL_ImgLabel################################
#Demo_mutilabel -test loadImagePath svPath keyfile MutiLabel_T binGPU deviceID
../../Demo_mutilabel -test $in_Images $savepath$imgpath $keyfile 0.6 1 0


