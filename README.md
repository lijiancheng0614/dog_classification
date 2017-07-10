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
train.sh inception_v3 0
```

### Test

Run `test.sh`
