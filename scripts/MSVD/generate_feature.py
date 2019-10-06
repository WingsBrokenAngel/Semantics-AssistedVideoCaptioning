# -*- encoding: utf-8 -*-
"""这是用来产生MSVD特征的脚本"""

import cv2
import numpy as np
import caffe
import math
import scipy.io as sio
import time
import random
import itertools
import os
from pprint import pprint
import pickle

batch_size = 32
seg_size = 32
path1 = '/home/chenhaoran/data/MSVD_frames'
path2 = './dict_youtube_mapping.pkl'
dims = [batch_size, 224, 224, 3]


def generate_feat(model_def_file, model_file, dmap):
    """
    首先要读入caffe的预训练模型
    然后读入每个视频的图像
    接着用caffe对这些图像进行处理得到输出特征
    最后将这些特征保存为mat格式的文件
    """
    # 读入caffe预训练模型
    gpu_id = 0
    caffe.set_device(gpu_id)
    caffe.set_mode_gpu()
    net = caffe.Net(model_file, model_def_file, caffe.TEST)
    eco_feats = np.zeros([1970, 1536], np.float32)
    image_mean = np.array([104, 117, 123], dtype=np.float64)
    image_mean = np.tile(image_mean[np.newaxis,np.newaxis,:], (224,224,1))
    # 读入视频列表
    vid_names = os.listdir(path1)
    # 对每个视频均匀读入seg_size张的图片，一个视频为一批进行处理
    for idx in range(0, len(vid_names)):
        # 读入该视频所有图片名，均匀选择seg_size张图片
        input_imgs = np.zeros(shape=dims, dtype=np.float64)
        name_list_full = []
        for idx2 in range(1):
            name = vid_names[idx+idx2]
            path2 = os.path.join(path1, name)
            frm_names = os.listdir(path2)
            frm_names = [f for f in frm_names if f[-4:]!='.npy']
            frm_len = len(frm_names)
            print(idx, name)
            delta = frm_len / seg_size
            idx_list = [int(i*delta) for i in range(seg_size)]
            print(frm_len, max(idx_list))
            name_list = [os.path.join(path2, frm_names[i]) for i in idx_list]
            name_list_full = name_list_full + name_list
        # 用cv2读入列表里的图像
        for idx2, img_path in enumerate(name_list_full):
            # print(img_path)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            input_imgs[idx2,:,:,:] = img - image_mean
        input_imgs = np.transpose(input_imgs, (0, 3, 1, 2))
        net.blobs['data'].data[...] = input_imgs
        output = net.forward()
        for idx2 in range(1):
            eco_feats[int(dmap[vid_names[idx+idx2]][3:])-1] = net.blobs['global_pool_gn02_reshape'].data[idx2]
    np.save('eco_32_feats.npy', eco_feats)



if __name__ == "__main__":
    # 模型文件
    model_def_file = "/home/chenhaoran/data1/eco/models/ECO_full_kinetics.caffemodel"
    model_file = "/home/chenhaoran/data1/eco/models_ECO_Full/kinetics/deploy.prototxt"
    # 产生特征
    dmap = pickle.load(open(path2, 'rb'))
    generate_feat(model_def_file, model_file, dmap)


