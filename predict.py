import sys
sys.path.insert(0, 'caffe/python')
import caffe
import os
import numpy as np
from PIL import Image

test_dir = sys.argv[1]
model_name = sys.argv[2]
model_iteration = sys.argv[3]
gpu_id = int(sys.argv[4])
output_file_path = 'submission.txt'
prototxt = 'prototxt/{}/deploy.prototxt'.format(model_name)
model = 'model/{}/train_iter_{}.caffemodel'.format(model_name, model_iteration)

caffe.set_device(gpu_id)
caffe.set_mode_gpu()

def test(image_path, net):
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2,1,0))
    image = caffe.io.load_image(image_path)
    net.blobs['data'].data[...] = transformer.preprocess('data', image)
    _ = net.forward()
    out = net.blobs['prob'].data[0].argmax()
    return out

net = caffe.Net(prototxt, model, caffe.TEST) 

fd = open(output_file_path, 'w')
l = os.listdir(test_dir)
for idx in l:
    print(idx)
    image_path = os.path.join(test_dir, idx)
    out = test(image_path, net)
    fd.write('{}\t{}\n'.format(out, idx.replace('.jpg', '')))

fd.close()
