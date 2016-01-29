
#Info
loadfile='/home/chigo/working/caffe_img_classification/test/bin/downloadimg_sh/imagequality_bad_0.9.txt'
#loadfile='ads_text_2dcode_050611_ubuntu_10.csv'

path='/home/chigo/image/inquality/train/sv'

#rm && mkdir Path
rm -r $path
mkdir $path

#download img
#Demo_downloadimg -downloadidurlscore queryList.csv label savePath BinReSizeImg
../Demo_downloadimg -downloadidurlscore $loadfile "scene" $path 0

