mpwd=`pwd`
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../src/API_caffe/v1.1.0/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../src/API_similardetect/lib

path_img='/home/chigo/image/mainboby/frcnn_DATA/VOC/JPEGImages_voc0712_in/doing/'
savepath='get_big_img/'
################################Path################################
rm -r $savepath
mkdir $savepath

################################mkdir################################
a=(aeroplane
bicycle
bird
boat
bottle
bus
car
cat
chair
cow
diningtable
dog
horse
motorbike
person
pottedplant
sheep
sofa
train
tvmonitor)

for((i=0;i<20;i++))
 do
  b=${a[$i]}
  echo $b

  rm -r $savepath$b
  mkdir $savepath$b

  find $path_img/$b/ -name "*.jpg" >list_jpg.txt

  ################################DL_ImgLabel################################
  #Demo_mainboby_frcnn -get_big_img queryList.txt savepath ClassName MaxSingleClassNum
  ../Demo_mainboby_frcnn -get_big_img list_jpg.txt $savepath $b 3

  rm list_jpg.txt
done



