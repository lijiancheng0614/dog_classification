#!/usr/bin/env sh

FILE_PROTOTXT=prototxt/$1/val.prototxt
FILE_MODEL=model/$1/train_iter_$2.caffemodel
DIR_LOG=log/$1
mkdir -p $DIR_LOG
FILE_LOG=$DIR_LOG/test_$2.txt

caffe test --model="$FILE_PROTOTXT" \
    --weights="$FILE_MODEL" \
    --iterations=165 \
    --gpu=$3 \
2>&1 | tee "$FILE_LOG" &
