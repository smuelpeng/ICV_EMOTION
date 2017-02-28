#!/usr/bin/env sh
TOOLS=../caffe/caffe/build/tools
#nohup \
 $TOOLS/caffe train \
    --solver=$1 \
    --weights=./model/icv_iter_100000.caffemodel \
    --gpu 0,1 
    #--weights=./model/icv_iter_100000.caffemodel \
echo "done"
