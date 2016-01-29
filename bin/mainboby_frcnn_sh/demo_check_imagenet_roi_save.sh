mpwd=`pwd`
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../src/API_caffe/v1.1.0/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../src/API_similardetect/lib
################################Path################################
path_xml='/home/xiaogao/job/research/fast-rcnn-master/data/VOCdevkit2020/VOC2020/Annotations/'
path_img='/home/xiaogao/job/research/fast-rcnn-master/data/VOCdevkit2020/VOC2020/JPEGImages/'

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
    cp $path_img/$line.JPEG $imgSavePath/$line.jpg
done

#rm
#rm list_id.txt

