mpwd=`pwd`
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../src/API_caffe/v1.1.0/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../src/API_proposal/BINGpp/lib
################################Path################################
path_src='/home/chigo/image/face/JPEGImages/'
path_xml_save='Annotations/'
path_img_save='JPEGImages/'

rm -r $path_xml_save
mkdir $path_xml_save

rm -r $path_img_save
mkdir $path_img_save
################################mkdir################################
a=(other
pet
puppet
sticker)

for((i=0;i<4;i++))
 do
  b=${a[$i]}
  echo $b

  find $path_src/$b/ -name "*.jpg" >list_$b.txt
  
  rm -r $path_xml_save$b
  mkdir $path_xml_save$b
  rm -r $path_img_save$b
  mkdir $path_img_save$b

  ################################DL_ImgLabel################################
  #Demo_facedetect -get_img2xml queryList.txt xmlSavePath imgSavePath labelname
  ../Demo_facedetect -get_img2xml list_$b.txt $path_xml_save$b $path_img_save$b $b

  rm list_$b.txt
done

