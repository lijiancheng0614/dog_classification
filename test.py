import sys
sys.path.insert(0, 'caffe/python')
import caffe
import os
import numpy as np
from PIL import Image

prototxt = sys.argv[1]
model = sys.argv[2]
gpu_id = int(sys.argv[3])
output_file_path = 'submission.txt'

caffe.set_device(gpu_id)
caffe.set_mode_gpu()

def test(net):
    _ = net.forward()
    out = net.blobs['prob'].data[0].argmax()
    return out

net = caffe.Net(prototxt, model, caffe.TEST) 

fd = open(test_list_path)
l = [line.split()[0] for line in fd]
fd.close()
fd = open(output_file_path, 'w')
for idx in l:
    print(idx)
    out = test(net)
    fd.write('{}\t{}\n'.format(out, idx.replace('.jpg', '')))

fd.close()
