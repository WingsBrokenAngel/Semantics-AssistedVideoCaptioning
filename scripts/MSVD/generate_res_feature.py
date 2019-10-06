# -*- encoding: utf-8 -*-
"""这是用来产生MSVD特征的脚本"""

import cv2
import numpy as np
import tensorflow as tf
import tensornets as nets
from tensornets.utils import load_img
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


def generate_feat(inputx, model, out_feats, sess, dmap):
    """
    用model自带的imgload读入每个视频的图像
    接着用model对这些图像进行处理得到输出特征
    最后将这些特征保存为npy格式的文件
    """
    # 读入视频列表
    vid_names = os.listdir(path1)
    res_feats = np.zeros([1970, seg_size, 2048], np.float32)
    # 对每个视频均匀读入seg_size张的图片，一个视频为一批进行处理
    for idx in range(0, len(vid_names)):
        # 读入该视频所有图片名，均匀选择seg_size张图片
        input_imgs = np.zeros(shape=dims, dtype=np.float32)

        name = vid_names[idx]
        vidpath = os.path.join(path1, name)
        frm_names = os.listdir(vidpath)
        frm_names = [f for f in frm_names if f[-4:]!='.npy']
        frm_len = len(frm_names)
        delta = frm_len / seg_size
        idx_list = [int(i*delta) for i in range(seg_size)]
        print(idx, name, frm_len, max(idx_list))
        name_list = [os.path.join(vidpath, frm_names[i]) for i in idx_list]
        # 用load_img读入列表里的图像, model.preprocess进行预处理
        for idx2, img_path in enumerate(name_list):
            img = load_img(img_path, target_size=256, crop_size=224)
            input_imgs[idx2,:,:,:] = model.preprocess(img)

        feats = sess.run(out_feats, {inputx: input_imgs})

        res_feats[int(dmap[vid_names[idx]][3:])-1] = feats
    np.save('ResNeXt101c64_32frm_feats.npy', res_feats)



if __name__ == "__main__":
    # 模型文件
    inputx = tf.placeholder(tf.float32, [None, 224, 224, 3])
    model = nets.ResNeXt101c64(inputx, is_training=False)
    out_feats = model.get_outputs()[-3]
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    sess.run(model.pretrained())
    # 产生特征
    dmap = pickle.load(open(path2, 'rb'))
    generate_feat(inputx, model, out_feats, sess, dmap)


