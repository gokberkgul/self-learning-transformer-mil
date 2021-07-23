camelyon_path=$1

mkdir $camelyon_path
mkdir $camelyon_path'/training'
mkdir $camelyon_path'/testing'
normal_path=$camelyon_path'/training/normal'
tumor_path=$camelyon_path'/training/tumor'
test_path=$camelyon_path'/testing/images'
mkdir $normal_path
mkdir $tumor_path
mkdir $test_path

cd $camelyon_path && cd .. && cd $normal_path
pwd
for i in {1..160}
do
if [ $i -lt 10 ]
then
    echo "https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_00$i.tif"
elif [ $i -lt 100 ]
then
    echo "https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_0$i.tif"
else
    echo "https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_$i.tif"
fi
done

cd ../../.. && cd $tumor_path
pwd
for i in {1..111}
do
if [ $i -lt 10 ]
then
    echo "https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_00$i.tif"
elif [ $i -lt 100 ]
then
    echo "https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_0$i.tif"
else
    echo "https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_$i.tif"
fi
done

cd ../../.. && cd $test_path
pwd
for i in {1..130}
do
if [ $i -lt 10 ]
then
    echo "https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_00$i.tif"
elif [ $i -lt 100 ]
then
    echo "https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_0$i.tif"
else
    echo "https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_$i.tif"
fi
done