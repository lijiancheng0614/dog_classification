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

def test(image_path, net, new_height=330, new_width=330):
    im = Image.open(image_path)
    im = im.resize((new_width, new_height), Image.BILINEAR)
    im = np.array(im, dtype=np.float32)
    h, w = net.blobs['data'].data[0, 0].shape
    dh, dw = (new_height - h) / 2, (new_width - w) / 2
    im = im[dh : dh + h, dw : dw + w]
    im = im[:,:,::-1]
    im -= np.array((104, 117, 123))
    im = im.transpose((2,0,1))
    net.blobs['data'].data[...] = im
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
