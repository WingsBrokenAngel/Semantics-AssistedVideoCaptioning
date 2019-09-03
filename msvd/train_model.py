# -*- coding: utf-8 -*-
# Author: Haoran Chen
# Date: 2019-04-03
# Date: 2019-04-10
# Date: 2019-04-23

import sys
sys.path.append('..')
import tensorflow as tf
import pickle
import numpy as np
from numpy.random import multinomial, shuffle
from scn import SemanticLSTM
from pprint import pprint
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor


np.random.seed(123)
options = {}
options['n_steps'] = 20         # max sentence length
n_steps = options['n_steps']
options['n_x'] = 300            # word embedding size
embed_size = options['n_x']
options['n_f'] = 512
options['n_y'] = 300            # tag feature
options['n_z'] = 3584           # video feature
options['batch_size'] = 64       # batch size
bs = options['batch_size']
options['n_h'] = 512            # output size
options['n_v'] = 12594          # vocabulary size
epoch = 50
TRAIN_SIZE = 48774 

flags, features, tag_feat, idx2word = None, None, None, None

YOUTUBE_CORPUS = "./data/youtube_corpus.pkl"
RES_ECO_FEAT = "./data/msvd_resnext_eco_32.npy"
TAG_FEAT = "./tagging/msvd_semantic_tag_e100.npy"
REF = './data/youtube_references.pkl'
WEIGHT = {'Bleu_1':1., 'Bleu_2': 1.11, 'Bleu_3': 1.25, 'Bleu_4': 1.43, 
          'CIDEr': 0.83, 'METEOR': 2., 'ROUGE_L':0.}

def main():
    # 加载数据
    global features, tag_feat, idx2word
    with open(YOUTUBE_CORPUS, "rb") as fo:
        youtube_corpus = pickle.load(fo)
    word_embed_array = youtube_corpus[5] # 12594 * 300, 词嵌入矩阵
    options['embeddings'] = word_embed_array
    # 2 * 48774  训练集标签   2 * 4290     验证集标签   2 * 27763   测试集标签
    train_data = youtube_corpus[0]
    idx2word = youtube_corpus[4]
    train_gt_sents = [[idx2word[w] for w in sent] for sent in train_data[0]]
    features = np.load(RES_ECO_FEAT) # 1970 * 3584   输入的图像特征
    tag_feat = np.load(TAG_FEAT) # 1970 * 300        标注的特征
    # 构建模型和优化器
    model = SemanticLSTM(options)
    best_score, save_path = 0., None
    with model.graph.as_default():
        global_step = tf.train.get_or_create_global_step()
        lr = tf.train.exponential_decay(0.0004, global_step, 8000000, 0.5, True)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(model.train_loss, global_step)
        # 保存模型的对象
        saver = tf.train.Saver(max_to_keep=30)
        # 训练模型
        train_idx = np.arange(TRAIN_SIZE, dtype=np.int32)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config, graph=model.graph)
    
        sess.run(tf.global_variables_initializer())
        print('argmax:', sess.run(model.if_argmax))
        for idx1 in range(epoch):
            train_part(train_data,model, sess, train_op, idx1, train_gt_sents)
            scores = cal_metrics(model, sess, 'val')
            sum_s = np.sum([scores[key]*WEIGHT[key] for key in scores])
            if best_score < sum_s:
                best_score = sum_s
                save_path = saver.save(sess, './saves/%s-best.ckpt' % flags.name)
                print('Epoch %d: the best model has been saved as %s.\n' % 
                      (idx1, save_path), flush=True)
        saver.restore(sess, save_path)
        cal_metrics(model, sess, 'test')


def print_sents(indices, gt_sents, preds, phase):
    for idx1, idx2 in enumerate(indices):
        cap1, cap2 = gt_sents[idx2], preds[idx1]
        print(phase, idx2, ' '.join(cap1), '\t\t', ' '.join(cap2))
        if idx1 >= 9:
            break
    print('\n', flush=True)


def train_step(data, model, sess, indices, train_op, eidx):
    mask, tags, batch_feat, captions = get_batch(bs, data, indices)
    sample_prob = eidx // 10 * 0.1
    wanted_ops = [train_op, model.loss, model.sents]
    feed_dict = {model.words: captions, model.y: tags, 
                 model.z: batch_feat, model.mask: mask, 
                 model.batch_size: len(indices), 
                 model.keep_prob: 0.5, model.sample_prob: sample_prob, 
                 model.n_steps: captions.shape[0]}
    _, loss1, sents = sess.run(wanted_ops, feed_dict)
    return loss1, sents


def get_batch(bs, all_data, indices):
    '''
    Args:
    bs: type int, batch size
    all_data: 2-d array, 2*sentences number
    tag_feat: 2-d array, video number * tag feature dimension
    features: 2-d array, video number * video feature dimension
    sidx: int, start index
    eidx: int, end index

    Returns:
    mask: 2-d float32 array, max steps * batch size
    tags: 2-d float32 array, batch size * tag dimension, tag features for this batch
    batch_feat: 2-d float32 array, batch size * video feature dimension
    captions: 2-d int32 array, max_step * batch_size
    '''
    max_len = max([len(all_data[0][idx]) for idx in indices])
    mask = np.zeros(shape=(max_len, len(indices)), dtype=np.float32)
    captions = np.zeros(shape=(max_len, len(indices)), dtype=np.int32)
    tags, batch_feat = [], []
    for idx1, idx2 in enumerate(indices):
        sent = all_data[0][idx2]
        captions[:len(sent), idx1] = sent    # 句长*迷你批尺寸
        mask[:len(sent), idx1] = 1.
        vid_idx = all_data[1][idx2]
        tags.append(tag_feat[vid_idx])
        batch_feat.append(features[vid_idx])
    tags = np.stack(tags, axis=0)
    batch_feat = np.stack(batch_feat, axis=0)
    return mask, tags, batch_feat, captions


def train_part(train_data, model, sess, train_op, idx1, train_gt_sents):
    train_idx = np.arange(TRAIN_SIZE)
    np.random.shuffle(train_idx)
    train_loss = 0
    sent_list = []
    for idx2 in range(TRAIN_SIZE//bs):
        indices = train_idx[idx2 * bs:(idx2 + 1) * bs]
        loss1, sents = train_step(train_data, model, sess, indices, train_op, idx1)
        train_loss += loss1
        sent_list.append(sents.T)
    res_num = TRAIN_SIZE % bs
    if res_num:
        indices = train_idx[TRAIN_SIZE // bs * bs:TRAIN_SIZE]
        loss1, sents = train_step(train_data, model, sess, indices, train_op, idx1)
        train_loss += loss1
        sent_list.append(sents.T)

    print('Epoch {:3d}: Train Loss {:.5f}'.format(idx1, train_loss / TRAIN_SIZE), flush=True)
    if idx1 % 10 == 9:
        train_preds = []
        for sents in sent_list:
            for sent in sents:
                tmp = []
                for w in sent:
                    tmp.append(idx2word[w])
                    if w == 0:
                        break
                train_preds.append(tmp)
        print_sents(train_idx, train_gt_sents, train_preds, 'train')



def cal_metrics(model, sess, phase):
    sent_dict, sent_list = {}, []
    if phase == "val":
        ref = pickle.load(open(REF, 'rb'))[1]
        idx2cap = {idx+1200: elem for idx, elem in enumerate(ref)}
        idx_start, idx_end = 1200, 1300
    elif phase == "test":
        ref = pickle.load(open(REF, 'rb'))[2]
        idx2cap = {idx+1300: elem for idx, elem in enumerate(ref)}
        idx_start, idx_end = 1300, 1970
    for idx in range(idx_start, idx_end):
        tag, vid, mask = tag_feat[idx], features[idx], np.ones([n_steps, 1])
        tag, vid = np.expand_dims(tag, 0), np.expand_dims(vid, 0)
        wanted_ops = model.sents_sample
        feed_dict = {model.mask: mask, model.y: tag, model.z: vid, 
                     model.batch_size: 1, model.n_steps: n_steps}
        sent = sess.run(wanted_ops, feed_dict)
        sent_dict[idx] = []
        for x in np.squeeze(sent):
            if x == 0:
                break
            sent_dict[idx].append(idx2word[x])
        sent_dict[idx] = [' '.join(sent_dict[idx])]
        sent_list.append(sent_dict[idx][0])
    scores = score(idx2cap, sent_dict)
    pprint(scores)
    with open(options['flags'].name+'_output.log', 'w') as fo:
        for sent in sent_list:
            fo.write(sent+'\n')
    return scores

def score(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(),"METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores

    

if __name__ == "__main__":
    # global flags
    tf.app.flags.DEFINE_integer('argmax', 1, 
                                '1 for argmax and 0 for multinomial sample')
    tf.app.flags.DEFINE_string('name', '1', 'name of model')
    tf.app.flags.DEFINE_string('corpus', None, 'Path to corpus file')
    tf.app.flags.DEFINE_string('reseco', None, 'Path to ResNeXt & ECO feature files')
    tf.app.flags.DEFINE_string('tag', None, 'Path to Tag feature files')
    tf.app.flags.DEFINE_string('ref', None, 'Path to reference files')
    flags = tf.app.flags.FLAGS
    if flags.corpus:
        YOUTUBE_CORPUS = flags.corpus
    if flags.reseco:
        RES_ECO_FEAT = flags.reseco
    if flags.tag:
        TAG_FEAT = flags.tag
    if flags.ref:
        REF = flags.ref
    options['flags'] = flags
    main()
