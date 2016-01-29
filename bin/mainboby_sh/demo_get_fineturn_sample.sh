mpwd=`pwd`
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../src/API_caffe/v1.0.0/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../src/API_similardetect/lib

################################Path################################
savepath='res_fineturn_sample/'

rm -r $savepath
mkdir $savepath

rm -r $savepath/train
mkdir $savepath/train

rm -r $savepath/val
mkdir $savepath/val

rm -r $savepath/test
mkdir $savepath/test

################################mkdir################################
a=(food.food.barbecue
food.food.bread
food.food.cake
food.food.candy
food.food.coffee
food.food.cook
food.food.cookie
food.food.crab
food.food.dumpling
food.food.food
food.food.fruit
food.food.hamburger
food.food.hotpot
food.food.icecream
food.food.pasta
food.food.pizza
food.food.rice
food.food.steak
food.food.sushi
goods.car.bicycle
goods.car.bus
goods.car.car
goods.car.motorbike
goods.car.train
goods.goods.airplane
goods.goods.bag
goods.goods.bangle
goods.goods.bottle
goods.goods.bracelet
goods.goods.camera
goods.goods.chair
goods.goods.clothes
goods.goods.cosmetics
goods.goods.diningtable
goods.goods.drawbar
goods.goods.flower
goods.goods.glass
goods.goods.goods
goods.goods.guitar
goods.goods.hat
goods.goods.jewelry
goods.goods.laptop
goods.goods.lipstick
goods.goods.manicure
goods.goods.pendant
goods.goods.phone
goods.goods.pottedplant
goods.goods.puppet
goods.goods.ring
goods.goods.ship
goods.goods.shoe
goods.goods.sofa
goods.goods.tvmonitor
goods.goods.watch
other.2dcode.2dcode
other.other.other
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
pet.pet.alpaca
pet.pet.bird
pet.pet.cat
pet.pet.cow
pet.pet.dog
pet.pet.horse
pet.pet.pet
pet.pet.rabbit
pet.pet.sheep
scene.scene.clothingshop
scene.scene.courtyard
scene.scene.desert
scene.scene.forest
scene.scene.grasslands
scene.scene.highway
scene.scene.house
scene.scene.mountain
scene.scene.scene
scene.scene.sea
scene.scene.sky
scene.scene.street
scene.scene.supermarket
scene.scene.tallbuilding)

################################Path################################
#test
#path='/home/chigo/working/research/fast-rcnn-master/data/VOCdevkit2000/VOC2000/JPEGImages/'
path='/home/chigo/image/mainboby/mainboby_1_0_0_20160106/img_test/'

for((i=0;i<90;i++))
 do
  b=${a[$i]}
  echo $b
  
  rm -r $savepath/train/$b
  mkdir $savepath/train/$b

  rm -r $savepath/val/$b
  mkdir $savepath/val/$b

  rm -r $savepath/test/$b
  mkdir $savepath/test/$b

  find $path/$b/ -name "*.jpg" >list_jpg.txt

  ################################DL_ImgLabel################################
  #Demo_mainboby -get_fineturn_sample queryList.txt savepath label
  ../Demo_mainboby -get_fineturn_sample list_jpg.txt $savepath $b

  rm list_jpg.txt
done



