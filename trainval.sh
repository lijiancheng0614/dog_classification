#!/usr/bin/env sh

now=$(date +"%Y%m%d_%H%M%S")
FILE_SOLVER=prototxt/$1/solver2.prototxt
DIR_MODEL=model/$1
FILE_INIT_MODEL=$DIR_MODEL/train_iter_15000.caffemodel
DIR_LOG=log/$1
mkdir -p $DIR_LOG
FILE_LOG=$DIR_LOG/trainval_$now.txt

caffe train --solver="$FILE_SOLVER" \
    --weights="$FILE_INIT_MODEL" \
    --gpu=$2 \
2>&1 | tee "$FILE_LOG" &
