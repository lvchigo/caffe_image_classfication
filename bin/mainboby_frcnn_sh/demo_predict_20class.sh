mpwd=`pwd`
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../src/API_caffe/v1.1.0/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../src/API_similardetect/lib

savepath='res_predict/'
################################Path################################
rm -r $savepath
mkdir $savepath

################################mkdir################################
a=(aeroplane bicycle bird boat bottle bus car cat chair cow
   diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor)

for((i=0;i<20;i++))
 do
  b=${a[$i]}
  echo $b
  
  rm -r $savepath$b
  mkdir $savepath$b
done

################################Path################################
#test
#loadImgList='../imglist/list_test0313_10k.txt'			#onlime_img_1w
#loadImgList='imglist/image_73class_test_1w_100.txt'			#onlime_img_1w
loadImgList='imglist/image_voc2007_1w_100.txt'

################################DL_ImgLabel################################
#Demo_mainboby_frcnn -frcnn queryList.txt keyFilePath layerName binGPU deviceID
../Demo_mainboby_frcnn -frcnn $loadImgList ../../keyfile/ "fc7" 1 0

################################result 2 html################################
#./showDemo.py
#firefox showDemo.html
