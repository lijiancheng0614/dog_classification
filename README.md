# dog_classification

宠物狗种类识别

百度-西交大——大数据竞赛2017：http://js.baidu.com/

## Files tree

```
├── data
│   ├── train
│   ├── val
│   ├── test-01
│   ├── test-02
│   ├── data_train.txt
│   ├── label_name.txt
│   ├── list_train.txt
│   ├── list_test1.txt
│   └── list_test2.txt
├── model
├── prototxt
├── README.md
├── test.sh
└── train.sh
```

## Usage

### Data preparation

- train+val

    训练数据：8210条记录，实为8153张图片，100类宠物狗

    验证数据：10551张图片

    链接：http://pan.baidu.com/s/1slLOqBz

    密码：5axb

- test-01

    链接：http://pan.baidu.com/s/1gfaf9rt

    密码：fl5n

### Train

Run train script: `./train.sh $MODEL_NAME $GPU_ID`:

```bash
./train.sh inception_v3 0
```

### Test

Run test script: `./test.sh $MODEL_NAME $ITER $GPU_ID`:

```bash
./test.sh inception_v3 2500 0
```

### Submission

Run predict script: `python predict.py $TEST_IMAGE_DIR $MODEL_NAME $ITER $GPU_ID`:

```bash
python predict.py data/test-01/ inception_v3 15000 0
```

Or modify `protoxt/$MODEL_NAME/test.prototxt` and

run test script: `python test.py $TEST_LIST_PATH $MODEL_NAME $ITER $GPU_ID`:

```bash
python test.py data/list_test-01.txt prototxt/inception_v3/test.prototxt 'model/inception_v3/train_iter_15000.caffemodel 0
```
