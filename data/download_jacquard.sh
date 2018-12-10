#! /bin/bash

for i in $(seq -f "%01g" 1 11)
do
    echo $i

    #wget https://jacquard.liris.cnrs.fr/data/Download/Jacquard_Dataset_$i.zip -P jacquard/
    unzip jacquard/Jacquard_Dataset_$i.zip -d ./jacquard/
    #wget http://pr.cs.cornell.edu/grasping/rect_data/temp/data$i.tar.gz -P cornell/
    #tar -xvzf cornell/data$i.tar.gz -C cornell/
    #rm cornell/data$i.tar.gz
    #mv cornell/$i/* cornell/
    #rmdir cornell/$i
done
