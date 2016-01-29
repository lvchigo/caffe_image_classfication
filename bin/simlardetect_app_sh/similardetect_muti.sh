mpwd=`pwd`
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../lib

#Path
path_src='img/myself_photo/'
path_save='sv/'

#Label
a=(blur balloon boat game light mountain
palace people pet pool sea
sky sunshine text tree other)

#detect similar img
rm -rf keyfile
mkdir keyfile

#merge file to one
rm -rf $path_save
mkdir $path_save

for((i=0;i<16;i++))
 do
  b=${a[$i]}
  echo $b

  find $path_src/$b/ -name "*.jpg" >list_$b.txt
  find $path_src/$b/ -name "*.JPG" >>list_$b.txt

  rm -rf $path_save/$b/
  mkdir $path_save/$b/

  #Demo_simlardetect -simdetect ImageList.txt keyFilePath svPath
  ../Demo_simlardetect -simdetect list_$b.txt keyfile/ $path_save/$b/
  #../Demo_simlardetect -simdetect_filter list_$b.txt keyfile/ $path_save/$b/

  #rm -rf keyfile
  rm list_$b.txt
done

#rm -rf keyfile
rm list_*.txt




