
#Info
#loadfile='ads_shuying_151109/test.csv'
#loadfile='ads_shuying_151109/query_result1.csv'
loadfile='ads_shuying_151109/bi_in1011-51.csv'

path='ads_shuying_151109/bi_in1011-51'

#rm && mkdir Path
rm -r $path
mkdir $path

#download img
#Demo_downloadimg -downloadurl queryList.csv savePath BinReSizeImg
#../Demo_downloadimg -downloadurl $loadfile $path 1

#Demo_downloadimg -ads_shuying queryList.csv savePath BinReSizeImg
../Demo_downloadimg -ads_shuying $loadfile $path 1

