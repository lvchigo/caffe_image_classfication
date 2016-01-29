mpwd=`pwd`
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mpwd/../../input/lib

################################Path################################
#loadImgList='../imglist/list_ads_5class_050702_train.txt'
loadImgList='../imglist/list_test0313_10k_10_train.txt'
#loadImgList='train_in73class.txt'

outPCAModel='PCA1000_imagequalityblur2class_050729.model'
outPCAFeat='train_pca1000feat_imagequalityblur2class_050729'
keyfile='../../keyfile/'
layerName='fc7'
################################DL_ImgLabel################################
#svMode:1-in73class,2-in6class,3-ads6class,4-imagequality,5-imagequality blur;
#Demo_online_classification -pcalearn queryList.txt outPCAModel outPCAFeat keyFilePath layerName binGPU deviceID svMode
../Demo_online_classification -pcalearn $loadImgList $outPCAModel $outPCAFeat $keyfile $layerName 1 0 5

