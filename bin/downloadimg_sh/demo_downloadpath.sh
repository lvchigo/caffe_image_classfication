
#Info
loadfile='youzhiyongh_ubuntu.csv'
#loadfile='ads_text_2dcode_050611_ubuntu_10.csv'

path='youzhiyongh_ubuntu'

#rm && mkdir Path
rm -r $path
mkdir $path

#download img
#Demo_downloadimg -downloadpath queryList.csv savePath BinReSizeImg
../Demo_downloadimg -downloadpath $loadfile $path 0

