mpwd=`pwd`
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../src/API_caffe/v1.1.0/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../src/API_similardetect/lib
################################Path################################
path_xml='/home/chigo/image/mainboby/frcnn_DATA/VOC/Annotations_voc0712/'
path_img='/home/chigo/image/mainboby/frcnn_DATA/VOC/JPEGImages_voc0712/'

################################Path################################
xmlSavePath='Annotations/'
rm -r $xmlSavePath
mkdir $xmlSavePath
################################Path################################
imgSavePath='JPEGImages/'
rm -r $imgSavePath
mkdir $imgSavePath

#creat list_id.txt by user
################################read file################################
for line in `cat list_id.txt`
do
    echo $line
    cp $path_xml/$line.xml $xmlSavePath/$line.xml
    cp $path_img/$line.jpg $imgSavePath/$line.jpg
done

#rm
#rm list_id.txt



