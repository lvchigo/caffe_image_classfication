mpwd=`pwd`
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../src/API_caffe/v1.1.0/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../src/API_similardetect/lib
################################Path################################
path_img='/home/chigo/image/mainboby/frcnn_DATA/no_need_frcnn/'
################################Path################################
path_img_save='img_num/'
rm -r $path_img_save
mkdir $path_img_save
################################mkdir################################
a=(food.bread
food.candy
food.coffee
food.cookie
food.crab
food.dumpling
food.hamburger
food.pasta
food.pizza
food.rice
food.steak
food.sushi
goods.bangle
goods.bracelet
goods.drawbar
goods.guitar
goods.hat
goods.laptop
goods.lipstick.lipstick
goods.pendant
goods.phone
goods.puppet
goods.ring
people.street.street
pet.alpaca
pet.rabbit
scene.clothingshop
scene.courtyard
scene.desert
scene.forest
scene.scene.grasslands
scene.handdrawn.color
scene.handdrawn.whiteblack
scene.highway
scene.scene.house
scene.mountain
scene.sea
scene.scene.sky
scene.street
scene.scene.supermarket
scene.tallbuilding
other.text.text
other)

for((i=0;i<43;i++))
 do
  b=${a[$i]}
  echo $b

  find $path_img/$b/ -name "*.jpg" >list_img_$b.txt
  
  rm -r $path_img_save/$b
  mkdir $path_img_save/$b

  ################################DL_ImgLabel################################
  #Demo_mainboby_frcnn -get_img_num queryList.txt savepath MaxSingleClassNum
  ../Demo_mainboby_frcnn -get_img_num list_img_$b.txt $path_img_save/$b/ 500

  rm list_img_$b.txt
done



