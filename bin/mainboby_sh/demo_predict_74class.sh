mpwd=`pwd`
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../src/API_caffe/v1.0.0/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../src/API_similardetect/lib

savepath='res_predict/'
################################Path################################
rm -r $savepath
mkdir $savepath

################################mkdir################################
a=(food.barbecue
food.bread
food.cake
food.candy
food.coffee
food.cook
food.cookie
food.crab
food.dumpling
food.fruit
food.hamburger
food.hotpot
food.icecream
food.pasta
food.pizza
food.rice
food.steak
food.sushi
goods.airplane
goods.bag
goods.bangle
goods.bottle
goods.bracelet
goods.camera
goods.car
goods.clothes
goods.cosmetics
goods.drawbar
goods.flower
goods.glass
goods.guitar
goods.hat
goods.laptop
goods.lipstick
goods.manicure
goods.pendant
goods.phone
goods.puppet
goods.ring
goods.ship
goods.shoe
goods.train
goods.watch
people.eye
people.friend
people.hair
people.kid
people.lip
people.self.female
people.self.male
people.street
pet.alpaca
pet.cat
pet.dog
pet.rabbit
scene.clothingshop
scene.courtyard
scene.desert
scene.forest
scene.grasslands
scene.handdrawn.color
scene.handdrawn.whiteblack
scene.highway
scene.house
scene.mountain
scene.sea
scene.sky
scene.sticker
scene.street
scene.supermarket
scene.tallbuilding
scene.text
2dcode
other)

for((i=0;i<74;i++))
 do
  b=${a[$i]}
  echo $b
  
  rm -r $savepath$b
  mkdir $savepath$b
done

################################Path################################
#test
loadImgList='../imglist/list_test0313_10k_100.txt'			#onlime_img_1w
#loadImgList='imglist/image_73class_test_1w_100.txt'			#onlime_img_1w
#loadImgList='imglist/image_voc2007_1w_100.txt'

################################DL_ImgLabel################################
#Demo_mainboby -predict queryList.txt keyFilePath layerName binGPU deviceID
../Demo_mainboby -predict $loadImgList ../../keyfile/ "fc7" 1 0

################################result 2 html################################
#./showDemo.py
#firefox showDemo.html
