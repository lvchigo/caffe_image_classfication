mpwd=`pwd`
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../../lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../../src/API_caffe/v1.2.0/lib
export PYTHONPATH=$PYTHONPATH:$mpwd/../../../src/API_caffe/v1.2.0/python/py-faster-rcnn-lib
################################Input Path################################
#new add label 
in_Images='/home/xiaogao/img/logo/logo_in_0726'
keyfile='../../../keyfile/'
################################Path################################
savepath='res/'
rm -r $savepath
mkdir $savepath

#Label
a=(electronics.cellphone
food.coffee
food.icecream
goods.bag
goods.clothes
goods.cosmetics
goods.hat
goods.other
goods.other_2
goods.shoe)

for((i=0;i<10;i++))
 do
  b=${a[$i]}
  echo $b

  find $in_Images/$b/ -name "*.jpg" >list_$b.txt

  mkdir $savepath/$b/
  xmlpath='Annoations/'
  mkdir $savepath/$b/$xmlpath
  imgpath='JPEGImages/'
  mkdir $savepath/$b/$imgpath
  checkpath='CheckImg/'
  mkdir $savepath/$b/$checkpath
  mixpath='mix/'
  mkdir $savepath/$b/$mixpath

  ################################DL_ImgLabel################################
  #Demo_mutilabel -resample loadImagePath svPath keyfile MutiLabel_T binGPU deviceID
  ../../Demo_mutilabel -resample list_$b.txt $savepath/$b/ $keyfile 0.8 1 0
done

rm list_*.txt


