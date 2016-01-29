
#Info
startID='184160000'
downloadNum='5000'
path='downloadnum/'

#rm && mkdir Path
rm -r $path
mkdir $path

../Demo_downloadimg -downloadnum $startID $downloadNum $path 0
