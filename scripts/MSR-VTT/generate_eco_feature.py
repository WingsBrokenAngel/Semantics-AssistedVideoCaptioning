# -*- encoding: utf-8 -*-
"""这是用来产生MSR-VTT特征的脚本"""

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
from glob import glob


batch_size = 32
seg_size = 32
path1 = '/home/chenhaoran/data1/msr-vtt/train-video-frame'
dims = [batch_size, 224, 224, 3]


def generate_feat(model_def_file, model_file):
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
    eco_feats = np.zeros([10000, 1536], np.float32)
    image_mean = np.array([104, 117, 123], dtype=np.float64)
    image_mean = np.tile(image_mean[np.newaxis,np.newaxis,:], (224,224,1))
    # 读入视频列表
    vid_names = glob(os.path.join(path1, '*'))
    # 对每个视频均匀读入seg_size张的图片，一个视频为一批进行处理
    for idx in range(0, len(vid_names)):
        # 读入该视频所有图片名，均匀选择seg_size张图片
        input_imgs = np.zeros(shape=dims, dtype=np.float64)
        vid_name = vid_names[idx]
        frm_names = glob(os.path.join(vid_name, '*'))
        if len(frm_names) < 16:
            continue
        frm_names = [f for f in frm_names if f[-4:]!='.npy']
        frm_len = len(frm_names)
        print(idx, vid_name)
        delta = frm_len / seg_size
        idx_list = [int(i*delta) for i in range(seg_size)]
        print(frm_len, max(idx_list))
        name_list = [frm_names[i] for i in idx_list]
        # 用cv2读入列表里的图像
        for idx2, img_path in enumerate(name_list):
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            input_imgs[idx2,:,:,:] = img - image_mean
        input_imgs = np.transpose(input_imgs, (0, 3, 1, 2))
        net.blobs['data'].data[...] = input_imgs
        output = net.forward()
        vid_idx = int(os.path.basename(vid_names[idx])[5:])
        eco_feats[vid_idx] = net.blobs['global_pool_gn02_reshape'].data[0]
    np.save('msrvtt_eco_32_feats', eco_feats)



if __name__ == "__main__":
    # 模型文件
    model_def_file = "/home/chenhaoran/data1/eco/models/ECO_full_kinetics.caffemodel"
    model_file = "/home/chenhaoran/data1/eco/models_ECO_Full/kinetics/deploy.prototxt"
    # 产生特征
    generate_feat(model_def_file, model_file)


