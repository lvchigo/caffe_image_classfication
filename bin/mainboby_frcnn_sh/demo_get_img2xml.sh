mpwd=`pwd`
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../src/API_caffe/v1.1.0/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../src/API_similardetect/lib

################################Path################################
path_src='/home/chigo/image/mainboby/frcnn_DATA/IN_FullImg_Other/JPEGImages/'
path_xml_save='Annotations/'
path_img_save='JPEGImages/'

rm -r $path_xml_save
mkdir $path_xml_save

rm -r $path_img_save
mkdir $path_img_save
################################mkdir################################
a=(food.bread
food.candy
food.coffee
food.cookie
food.crab
food.diningtable.diningtable
food.dumpling
food.food.barbecue
food.food.cake
food.food.cook
food.food.food
food.food.fruit
food.food.hotpot
food.food.icecream
food.hamburger
food.pasta
food.pizza
food.rice
food.steak
food.sushi
goods.airplane.airplane
goods.bag.bag
goods.bangle
goods.bottle.bottle
goods.bracelet
goods.camera.camera
goods.car.bicycle
goods.car.bus
goods.car.car
goods.car.motorbike
goods.car.train
goods.chair.chair
goods.clothes.clothes
goods.drawbar
goods.goods.cosmetics
goods.goods.flower
goods.goods.glass
goods.goods.goods
goods.goods.jewelry
goods.goods.manicure
goods.goods.watch
goods.guitar
goods.hat
goods.laptop
goods.lipstick.lipstick
goods.pendant
goods.phone
goods.pottedplant.pottedplant
goods.puppet
goods.ring
goods.ship.ship
goods.shoe.shoe
goods.sofa.sofa
goods.tvmonitor.tvmonitor
other
other.2dcode.2dcode
other.sticker.sticker
other.text.text
people.eye.eye
people.friend.friend
people.hair.hair
people.kid.kid
people.lip.lip
people.people.people
people.self.female
people.self.male
people.street.street
pet.alpaca
pet.bird.bird
pet.cat.cat
pet.cow.cow
pet.dog.dog
pet.horse.horse
pet.pet.pet
pet.rabbit
pet.sheep.sheep
scene.clothingshop
scene.courtyard
scene.desert
scene.forest
scene.highway
scene.mountain
scene.scene.grasslands
scene.scene.house
scene.scene.scene
scene.scene.sky
scene.scene.supermarket
scene.sea
scene.street
scene.tallbuilding)

for((i=0;i<90;i++))
 do
  b=${a[$i]}
  echo $b

  find $path_src/$b/ -name "*.jpg" >list_$b.txt
  
  rm -r $path_xml_save$b
  mkdir $path_xml_save$b
  rm -r $path_img_save$b
  mkdir $path_img_save$b

  ################################DL_ImgLabel################################
  #Demo_mainboby_frcnn -get_img2xml queryList.txt xmlSavePath imgSavePath labelname
  ../Demo_mainboby_frcnn -get_img2xml list_$b.txt $path_xml_save$b $path_img_save$b $b

  rm list_$b.txt
done

