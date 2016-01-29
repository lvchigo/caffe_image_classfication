mpwd=`pwd`
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../src/API_caffe/v1.1.0/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../src/API_similardetect/lib
################################Path################################
path_xml='/home/chigo/image/mainboby/frcnn_DATA/IN_47class_300_20151230/Annotations/'
path_img='/home/chigo/image/mainboby/frcnn_DATA/IN_47class_300_20151230/JPEGImages/'
################################Path################################
path_xml_save='Annotations/'
rm -r $path_xml_save
mkdir $path_xml_save
################################Path################################
path_img_save='JPEGImages/'
rm -r $path_img_save
mkdir $path_img_save
################################mkdir################################
c=(food.cake
food.cook
food.icecream
goods.airplane
goods.bag
goods.bottle
goods.camera
goods.car
goods.clothes
goods.cosmetics
goods.flower
goods.glass
goods.manicure
goods.ship
goods.shoe
goods.train
goods.watch
people.friend
people.kid
people.self.female
people.self.male
pet.cat
pet.dog)

a=(pet.dog)

for((i=0;i<1;i++))
 do
  b=${a[$i]}
  echo $b

  find $path_xml/$b/ -name "*.xml" >list_xml_$b.txt
  
  rm -r $path_xml_save$b
  mkdir $path_xml_save$b
  rm -r $path_img_save$b
  mkdir $path_img_save$b

  ################################DL_ImgLabel################################
  #Demo_mainboby_frcnn -ch_xml_name queryList.txt imgPath xmlSavePath imgSavePath
  ../Demo_mainboby_frcnn -ch_xml_name list_xml_$b.txt $path_img/$b/ $path_xml_save$b $path_img_save$b

  rm list_xml_$b.txt
done



