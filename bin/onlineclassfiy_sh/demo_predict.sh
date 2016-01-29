mpwd=`pwd`
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../input/lib

savepath='res/'
################################Path################################
rm -r $savepath
mkdir $savepath

################################mkdir################################
a=(other.other.other food.food.barbecue food.food.cake food.food.cook food.food.fruit 
food.food.hotpot food.food.icecream food.food.sushi food.food.food goods.bag.bag 
goods.clothes.clothes goods.goods.cosmetics goods.goods.flower goods.goods.glass goods.goods.hair 
goods.goods.jewelry goods.goods.manicure goods.goods.watch goods.shoe.shoe goods.goods.goods 
people.friend.friend people.kid.kid people.self.female people.self.male people.street.street 
people.people.people pet.cat.cat pet.dog.dog pet.pet.pet scene.scene.grasslands 
scene.scene.sea scene.scene.sky scene.scene.street scene.scene.scene other.sticker.sticker 
other.text.text other.2dcode.2dcode)

for((i=0;i<37;i++))
 do
  b=${a[$i]}
  echo $b
  
  rm -r $savepath$b
  mkdir $savepath$b
done

################################mkdir################################
a=(ads.ads.ads ads.2dcode.2dcode ads.text.text ads.norm.norm)

for((i=0;i<4;i++))
 do
  b=${a[$i]}
  echo $b
  
  rm -r $savepath$b
  mkdir $savepath$b
done

################################mkdir################################
a=(quality.bad.bad quality.medial.medial quality.good.good)

for((i=0;i<3;i++))
 do
  b=${a[$i]}
  echo $b
  
  rm -r $savepath$b
  mkdir $savepath$b
done

################################mkdir################################
#blurstr='blur/'
#blurlabel1='1'
#blurlabel2='2'
#rm -r $blurstr
#mkdir $blurstr
#rm -r $blurstr$blurlabel1
#mkdir $blurstr$blurlabel1
#rm -r $blurstr$blurlabel2
#mkdir $blurstr$blurlabel2

################################mkdir################################
#adsRecallstr='adsRecall3class/'
#adsRecallstr1='0'
#adsRecallstr2='1'
#adsRecallstr3='2'
#rm -r $adsRecallstr
#mkdir $adsRecallstr
#rm -r $adsRecallstr$adsRecallstr1
#mkdir $adsRecallstr$adsRecallstr1
#rm -r $adsRecallstr$adsRecallstr2
#mkdir $adsRecallstr$adsRecallstr2
#rm -r $adsRecallstr$adsRecallstr3
#mkdir $adsRecallstr$adsRecallstr3

################################Path################################
#test
loadImgList='../imglist/list_test0313_10k_2.txt'
#loadImgList='../imglist/ads_add_1026_list.txt'			#onlime_img_1w
#loadImgList='../imglist/list_ads_detect_online_1w_all.txt'	#onlime_ads_1w
#loadImgList='/home/chigo/working/caffe_img_classification/test/bin/onlineclassfiy_sh/test/test.txt'	#onlime_ads_yes

################################DL_ImgLabel################################
#Demo_online_classification -predict queryList.txt keyFilePath layerName binGPU deviceID
../Demo_online_classification -predict $loadImgList ../../keyfile/ "fc7" 1 0
################################result 2 html################################
#./showDemo.py
#firefox showDemo.html
