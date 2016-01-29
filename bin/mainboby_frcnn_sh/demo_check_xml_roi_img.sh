mpwd=`pwd`
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../src/API_caffe/v1.1.0/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../src/API_similardetect/lib
################################Path################################
path_xml='/home/chigo/working/research/fast-rcnn-master/data/VOCdevkit2015/VOC2015/Annotations/'
path_img='/home/chigo/working/research/fast-rcnn-master/data/VOCdevkit2015/VOC2015/JPEGImages/'

find $path_xml/ -name "*.xml" >list_xml.txt
################################Path################################
path_img_save='Check_xml_roi_img/'
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
scene.handdrawn.color
scene.handdrawn.whiteblack
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

for((i=0;i<92;i++))
 do
  b=${a[$i]}
  echo $b

  rm -r $path_img_save$b
  mkdir $path_img_save$b
done

################################DL_ImgLabel################################
#Demo_mainboby_frcnn -check_xml_roi_img queryList.txt imgPath imgSavePath binVOC
../Demo_mainboby_frcnn -check_xml_roi_img list_xml.txt $path_img/ $path_img_save/ 0

rm list_xml.txt



