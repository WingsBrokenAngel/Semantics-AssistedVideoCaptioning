# -*- coding: utf-8 -*-
# Author: Haoran Chen
# Date: 2019-4-28

import tensorflow as tf
from tensorflow import placeholder, glorot_normal_initializer, zeros_initializer
from tensorflow.nn import dropout
import numpy as np
from tag_net import TagNet
from pprint import pprint


n_y = 300
MSVD_PATH = None
MSRVTT_PATH = None
MSVD_GT_PATH = None
MSRVTT_GT_PATH = None
max_epochs = 1000
lr = 0.0002
batch_size = 128
keep_prob = 0.5
batch_size = 64
PATIENCE = 3



def main():
    (train_x, train_y), (test_x, test_y) = load_split_data()


    new_indices = []
    for idx in range(train_x.shape[0]):
        if train_x[idx].sum() > 1e-6:
            new_indices.append(idx)
    indices = np.array(new_indices)
    train_num, test_num = len(indices), test_x.shape[0]

    tag_net = TagNet()
    with tag_net.graph.as_default():
        global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(tag_net.cost, global_step)
        saver = tf.train.Saver(max_to_keep=100)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(graph=tag_net.graph, config=config)
    with tag_net.graph.as_default():
        sess.run(tf.global_variables_initializer())
    cost_list, acc_list = [], []
    bad_cnt = 0
    for eidx in range(max_epochs):
        residual = train_num % batch_size
        np.random.shuffle(indices)
        costs, accs = 0., 0.
        wanted_ops = [tag_net.cost, tag_net.acc, tag_net.pred, tag_net.pred_mask, 
        tag_net.tmp, tag_net.acc_mask, train_op]

        for idx in range(train_num // batch_size):
            ret = train_step(train_x, train_y, indices, idx, batch_size, tag_net, wanted_ops, sess)
            costs += ret[0]
            accs += ret[1]

        if residual:
            ret = train_step(train_x, train_y, indices, train_num//batch_size, residual, tag_net, wanted_ops, sess)
            costs += ret[0]
            accs += ret[1]

        avg_cost = costs / train_num
        avg_acc = accs / train_num
        print('train %d epoch: acc %.4f, cost %.4f' % (eidx, avg_acc, avg_cost))
        if eidx in [99, 199, 399, 799, 999]:
            p = saver.save(sess, './saves/msrvtt_tag_model_%d_resnext_eco.ckpt' % (eidx + 1))
            print(p, 'has been saved.')

        residual, costs, accs = test_num % batch_size, 0.0, 0.0
        for idx in range(test_num // batch_size):
            ret = test_step(test_x, test_y, idx, batch_size, tag_net, wanted_ops[:-1], sess)
            costs, accs = costs + ret[0], accs + ret[1]
        if residual:
            ret = test_step(test_x, test_y, idx, residual, tag_net, wanted_ops[:-1], sess)
            costs, accs = costs + ret[0], accs + ret[1]

        avg_cost, avg_acc = costs / test_num, accs / test_num
        print('test %d epoch: acc %.4f, cost %.4f' % (eidx, avg_acc, avg_cost))
        cost_list.append(avg_cost)
        acc_list.append(avg_acc)


def train_step(train_x, train_y, indices, idx, bs, tag_net, wanted_ops, sess):
    begin_idx = idx * batch_size
    end_idx = begin_idx + bs
    batchx = train_x[indices[begin_idx:end_idx]]
    batchy = train_y[indices[begin_idx:end_idx]]
    feed_dict = {tag_net.y: batchy, tag_net.z: batchx, tag_net.keep_prob: 0.5}
    res = sess.run(wanted_ops, feed_dict)
    # pprint(res)
    return res[0] * bs, res[1] * bs


def test_step(test_x, test_y, idx, bs, tag_net, wanted_ops, sess):
    begin_idx = idx * batch_size
    end_idx = begin_idx + bs
    batchx, batchy = test_x[begin_idx:end_idx], test_y[begin_idx:end_idx]
    feed_dict = {tag_net.y: batchy, tag_net.z: batchx, tag_net.keep_prob: 1.0}
    res = sess.run(wanted_ops, feed_dict)
    return res[0] * bs, res[1] * bs


def load_split_data():
    # load features for MSVD and MSR-VTT
    msvd = np.load(MSVD_PATH)
    msrvtt = np.load(MSRVTT_PATH)
    # load tag ground true for MSVD and MSR-VTT
    msvd_gt = np.load(MSVD_GT_PATH)
    msrvtt_gt = np.load(MSRVTT_GT_PATH)

    train_x = np.concatenate([msvd, msrvtt[:7010]], 0)
    train_y = np.concatenate([msvd_gt, msrvtt_gt[:7010]], 0)

    test_x = msrvtt[7010:]
    test_y = msrvtt_gt[7010:]
    return (train_x, train_y), (test_x, test_y)


if __name__ == "__main__":
    tf.app.flags.DEFINE_string("msvd", "", "path to MSVD features")
    tf.app.flags.DEFINE_string("msvd_tag", "", "path to MSVD tags")
    tf.app.flags.DEFINE_string("msrvtt", "", "path to MSRVTT features")
    tf.app.flags.DEFINE_string("msrvtt_tag", "", "path to MSRVTT tags")
    flags = tf.app.flags.FLAGS
    MSVD_PATH, MSVD_GT_PATH = flags.msvd, flags.msvd_tag
    MSRVTT_PATH, MSRVTT_GT_PATH = flags.msrvtt, flags.msrvtt_tag
    main()
