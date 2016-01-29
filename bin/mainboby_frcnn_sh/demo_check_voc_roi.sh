mpwd=`pwd`
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../src/API_caffe/v1.1.0/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../src/API_similardetect/lib
################################Path################################
path_xml='/home/chigo/image/mainboby/frcnn_DATA/VOC/Annotations_voc0712/'
path_img='/home/chigo/image/mainboby/frcnn_DATA/VOC/JPEGImages_voc0712/'

find $path_xml/ -name "*.xml" >list_xml.txt
################################Path################################
path_img_save='Check_voc_roi_img/'
rm -r $path_img_save
mkdir $path_img_save
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

  rm -r $path_img_save$b
  mkdir $path_img_save$b
done

################################DL_ImgLabel################################
#Demo_mainboby_frcnn -check_xml_roi_img queryList.txt imgPath imgSavePath　binVOC
../Demo_mainboby_frcnn -check_xml_roi_img list_xml.txt $path_img/ $path_img_save/ 1

rm list_xml.txt



