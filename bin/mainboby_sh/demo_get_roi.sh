mpwd=`pwd`
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../input/lib

savepath='res_Hypothese/'
################################Path################################
rm -r $savepath
mkdir $savepath

################################Path################################
#test
#loadImgList='../imglist/list_test0313_10k_100.txt'
#loadImgList='imglist/image_73class_test_1w_100.txt'
loadImgList='imglist/image_voc2007_1w_100.txt'

################################DL_ImgLabel################################
#Demo_mainboby -get_roi queryList.txt keyFilePath layerName binGPU deviceID
../Demo_mainboby -get_roi $loadImgList ../../keyfile/ "fc7" 1 0

################################result 2 html################################
#./showDemo.py
#firefox showDemo.html
