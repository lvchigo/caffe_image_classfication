mpwd=`pwd`
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../input/lib

################################Path################################
path_src='/home/chigo/image/inquality/train_argus_add/people/people_good/male_good'
path_save='/home/chigo/image/inquality/train_argus_add/people/people_good/male_good_good_sv'

#merge file to one
rm -rf $path_save
mkdir $path_save

#find file
find $path_src/ -name "*.jpg" >list_tmp.txt

################################DL_ImgLabel################################
#Demo_online_classification -removesmallimg queryList.txt svPath size
../Demo_online_classification -removesmallimg list_tmp.txt $path_save 256





