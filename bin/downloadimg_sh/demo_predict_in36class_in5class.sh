savepath='res/'

################################Path################################
rm -r $savepath
mkdir $savepath

################################Path################################
a=(other.other.other food.food.barbecue food.food.cake food.food.cook food.food.fruit 
food.food.hotpot food.food.icecream food.food.sushi food.food.food goods.bag.bag 
goods.clothes.clothes goods.goods.cosmetics goods.goods.flower goods.goods.glass goods.goods.hair 
goods.goods.jewelry goods.goods.manicure goods.goods.watch goods.shoe.shoe goods.goods.goods 
people.friend.friend people.kid.kid people.self.female people.self.male people.street.street 
people.people.people pet.cat.cat pet.dog.dog pet.pet.pet scene.scene.grasslands 
scene.scene.sea scene.scene.sky scene.scene.street scene.scene.scene other.sticker.sticker 
other.text.text)

for((i=0;i<36;i++))
 do
  b=${a[$i]}
  echo $b
  
  rm -r $savepath$b
  mkdir $savepath$b
done

################################Path################################
#test
#loadImgList='../imglist/list_test0313_10k_10.txt'			#onlime_img_1w
#loadImgList='../imglist/list_test0313_10k.txt'			#onlime_img_1w
#loadImgList='../imglist/list_ads_detect_online_1w_all.txt'	#onlime_ads_1w
#loadImgList='../imglist/list_ads_detect_online_1w_yes.txt'		#onlime_ads_yes
loadImgList='img_download.txt'

################################DL_ImgLabel################################
#cout << "\tDemo_ads -predict_ads queryList.txt result.txt keyFilePath layerName binGPU deviceID binLabel\n" << endl;

../Demo_downloadimg -predict_in36class $loadImgList result.txt ../../keyfile/ "fc7" 1 0 3
################################result 2 html################################
#./showDemo.py
#firefox showDemo.html
