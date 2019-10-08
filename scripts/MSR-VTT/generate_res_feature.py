# -*- encoding: utf-8 -*-
"""这是用来产生MSR-VTT特征的脚本"""

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
import glob

batch_size = 32
seg_size = 32
train_path = '/PATH/TO/VIDEO/FRAME'
dims = [batch_size, 224, 224, 3]
global flags


def generate_feat(inputx, model, out_feats, sess):
    """
    用model自带的imgload读入每个视频的图像
    接着用model对这些图像进行处理得到输出特征
    最后将这些特征保存为npy格式的文件
    """
    # 读入视频列表
    global flags
    seg_name = os.path.basename(flags.input)
    vid_names = glob.glob(os.path.join(flags.input, '*'))
    res_feats = np.zeros([10000, seg_size, 2048], np.float32)
    # 对每个视频均匀读入seg_size张的图片，一个视频为一批进行处理
    for idx, vid_n in enumerate(vid_names):
        # 读入该视频所有图片名，均匀选择seg_size张图片
        input_imgs = np.zeros(shape=dims, dtype=np.float32)
        vid_idx = int(os.path.basename(vid_n)[5:])
        fpaths = glob.glob(os.path.join(vid_n, '*'))
        frm_len = len(fpaths)
        if frm_len == 0:
            continue
        delta = frm_len / seg_size
        idx_list = [int(i*delta) for i in range(seg_size)]
        print(idx, vid_n, frm_len, max(idx_list))
        # 用load_img读入列表里的图像, model.preprocess进行预处理
        for idx2, idx3 in enumerate(idx_list):
            img_path = fpaths[idx3]
            img = load_img(img_path, target_size=256, crop_size=224)
            input_imgs[idx2,:,:,:] = model.preprocess(img)

        feats = sess.run(out_feats, {inputx: input_imgs})

        res_feats[vid_idx] = feats
        print(idx, 'video has been processed.')
    np.save(flags.output, res_feats)



if __name__ == "__main__":
    global flags
    tf.app.flags.DEFINE_string('input', '', 'input path')
    tf.app.flags.DEFINE_string('output', '', 'output path')
    flags = tf.app.flags.FLAGS
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
    generate_feat(inputx, model, out_feats, sess)


