mpwd=`pwd`
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../src/API_caffe/v1.0.0/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../src/API_similardetect/lib

savepath='res_predict/'
################################Path################################
rm -r $savepath
mkdir $savepath

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

for((i=0;i<90;i++))
 do
  b=${a[$i]}
  echo $b
  
  rm -r $savepath$b
  mkdir $savepath$b
done

################################Path################################
#test
#loadImgList='../imglist/list_test0313_10k.txt'			#onlime_img_1w
loadImgList='../imglist/list_test0313_10k_100.txt'			#onlime_img_1w
#loadImgList='imglist/image_73class_test_1w_100.txt'			#onlime_img_1w
#loadImgList='imglist/image_voc2007_1w_100.txt'

################################DL_ImgLabel################################
#Demo_mainboby -predict queryList.txt keyFilePath layerName binGPU deviceID
../Demo_mainboby -predict $loadImgList ../../keyfile/ "fc7" 1 0

################################result 2 html################################
#./showDemo.py
#firefox showDemo.html
