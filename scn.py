# -*- coding: utf-8 -*-
# Author: Chen Haoran
# Date: 2019-3-24
# Date: 2019-4-23

import tensorflow as tf
from tensorflow import placeholder, Variable, get_variable, variable_scope
from tensorflow.nn import dropout, bidirectional_dynamic_rnn, embedding_lookup, softmax
from tensorflow.layers import Dense
from tensorflow import glorot_normal_initializer
from tensorflow.losses import Reduction, sparse_softmax_cross_entropy
import numpy as np
import os
import sys
from pprint import pprint



class SemanticLSTM():   

    def __init__(self, options):
        '''
        n_x是词嵌入维度
        n_h是隐藏层单元数
        n_f是LSTM输入维度
        n_v是词汇量
        n_y是标签维数
        n_z是视频特征维数
        '''

        self.options = options
        n_x, n_h, n_f = options['n_x'], options['n_h'], options['n_f']
        n_y, n_z, n_v = options['n_y'], options['n_z'], options['n_v']
        self.graph = tf.Graph()
        with self.graph.as_default(), variable_scope("scn", reuse=tf.AUTO_REUSE):
            tf.set_random_seed(123)
            self._init_weights()
            self.words = placeholder(tf.int32, [None, None], name="word")
            self.mask = placeholder(tf.float32, [None, None], name="mask")  # shape (n_steps, batch_size)
            self.y = placeholder(tf.float32, [None, n_y], name="tag_feats")  
            self.z = placeholder(tf.float32, [None, n_z], name="vid_feats")
            self.batch_size = placeholder(tf.int32, [], name='batch_size')
            self.sample_prob = placeholder(tf.float32, [], name="sample_prob")
            self.keep_prob = placeholder(tf.float32, [], name="keep_prob")
            self.n_steps = placeholder(tf.int32, [], name="n_steps")
            self.if_argmax = Variable(bool(self.options['flags'].argmax), 
                                      trainable=False, name='if_argmax')
            self.embeddings = Variable(self.options['embeddings'], 
                                       trainable=False, name="embeddings")
            self.whid = get_variable('whid', [n_x, n_h], tf.float32, glorot_normal_initializer())
            self.bhid = get_variable('bhid', [n_v], tf.float32, tf.zeros_initializer())   
            self.output_w = tf.transpose(tf.matmul(self.embeddings, self.whid)) # hidden state size * vocabulary size
            self.c0 = get_variable('C0', [n_z, n_x], tf.float32, glorot_normal_initializer())

            self._train_part(n_x, n_h, self.n_steps)
            self._val_test()

    def _train_part(self, n_x, n_h, n_steps):
        n_v = self.options['n_v']
        # n_steps, batch_size, embed_dim
        words_embed = embedding_lookup(self.embeddings, self.words) 
        words_embed = dropout(words_embed, self.keep_prob, (1, self.batch_size, n_x))
        word_sliced = tf.slice(words_embed, [0, 0, 0], [n_steps-1, -1, -1])
        # print('word sliced:', word_sliced, word_sliced.get_shape())
        # convert video feature from None * n_z to None * n_x
        vid_feat_proj = tf.expand_dims(tf.matmul(self.z, self.c0), 0)
        # use video feature as the input for the first step
        self.state_below = tf.concat([vid_feat_proj, word_sliced], 0)
        # n_steps * batch_size * n_dims
        self.h_list = self._decoder_layer() 
        self.h_list_reshape = tf.reshape(self.h_list, [-1, n_h])
        # (n_steps*batch_size, n_vocabulary)
        logits = tf.matmul(self.h_list_reshape, self.output_w) + self.bhid 
        sents = tf.reshape(logits, [-1, self.batch_size, n_v])
        # (n_steps, batch_size)
        self.sents = tf.argmax(sents, -1)
        # (n_steps*batch_size, )
        w_reshape = tf.reshape(self.words, [-1], name="w_reshape")
        # (n_steps, batch_size)
        weighted_mask = self.mask / (tf.reduce_sum(self.mask, 0, True)**0.7)
        # (n_steps*batch_size)
        mask_word = tf.reshape(weighted_mask, shape=[-1], name="mask_word")
        mask_indices = tf.where(mask_word > 0)
        '''
        self.prediction = tf.gather_nd(logits, mask_indices, name="prediction")
        w_reshape = tf.gather_nd(w_reshape, mask_indices)
        mask_word = tf.gather_nd(mask_word, mask_indices)
        self.loss = sparse_softmax_cross_entropy(logits=self.prediction+1e-8, 
                                                 labels=w_reshape, 
                                                 weights=mask_word, 
                                                 reduction=Reduction.SUM)
        '''
        self.loss = sparse_softmax_cross_entropy(logits=logits, 
                                                 labels=w_reshape,
                                                 weights=mask_word, 
                                                 reduction=Reduction.NONE)
        self.loss = tf.reduce_sum(tf.gather(self.loss, mask_indices))
        self.train_loss = self.loss / tf.cast(self.batch_size, tf.float32)

    def _decoder_layer(self):
        """state below: size of n_steps * batch_size * n_x
        y: size of batch_size * n_y
        z: size of batch_size * n_z
        """
        n_steps = self.n_steps
        n_x = self.options['n_x']
        n_h = self.options['n_h']
        n_f = self.options['n_f']
        batch_size = self.batch_size
        # batch_size * n_f
        y = dropout(self.y, self.keep_prob)
        z = dropout(self.z, self.keep_prob)
        tmp2_i = tf.matmul(y, self.Wb_i)
        tmp2_f = tf.matmul(y, self.Wb_f)
        tmp2_o = tf.matmul(y, self.Wb_o)
        tmp2_c = tf.matmul(y, self.Wb_c)
        # batch_size * n_f
        tmp3_i = tf.matmul(z, self.Ca_i)
        tmp3_f = tf.matmul(z, self.Ca_f)
        tmp3_o = tf.matmul(z, self.Ca_o)
        tmp3_c = tf.matmul(z, self.Ca_c)
        # batch_size * n_f
        tmp4_i = tf.matmul(y, self.Cb_i)
        tmp4_f = tf.matmul(y, self.Cb_f)
        tmp4_o = tf.matmul(y, self.Cb_o)
        tmp4_c = tf.matmul(y, self.Cb_c)

        def _state_below(tmp1, tmp2, tmp3, tmp4, Wc, Cc, b):
            state_below = tf.matmul(tmp1*tmp2, Wc) + tf.matmul(tmp3*tmp4, Cc) + b
            return state_below

        def _step(a, elems):
            def _preactivate(a, y, w1, w2, w3, x):
                p = tf.matmul(tf.matmul(a, w1) * tf.matmul(y, w2), w3) + x
                return p

            def _get_word_embed(h):
                word_logit = tf.matmul(h, self.output_w) + self.bhid
                word_chosen1 = tf.argmax(word_logit, 1)
                word_chosen2 = tf.multinomial(word_logit, 1)
                word_chosen2 = tf.squeeze(word_chosen2)
                word_chosen = tf.cond(self.if_argmax, lambda: word_chosen1, lambda: word_chosen2)
                return embedding_lookup(self.embeddings, word_chosen)

            word_embed1, step = elems
            word_embed = tf.cond((tf.random_uniform([])>=self.sample_prob) | (step==0), 
                                 lambda: word_embed1, 
                                 lambda: _get_word_embed(a[0]))
            # batch_size = tf.shape(word_embed)[0]
            word_embed = tf.reshape(word_embed, (batch_size, n_x))
            # batch_size * n_f
            tmp1_i, tmp1_f = tf.matmul(word_embed, self.Wa_i), tf.matmul(word_embed, self.Wa_f)
            tmp1_o, tmp1_c = tf.matmul(word_embed, self.Wa_o), tf.matmul(word_embed, self.Wa_c)
            # batch_size * n_h
            input_i = _state_below(tmp1_i, tmp2_i, tmp3_i, tmp4_i, self.Wc_i, self.Cc_i, self.b_i)
            input_f = _state_below(tmp1_f, tmp2_f, tmp3_f, tmp4_f, self.Wc_f, self.Cc_f, self.b_f)
            input_o = _state_below(tmp1_o, tmp2_o, tmp3_o, tmp4_o, self.Wc_o, self.Cc_o, self.b_o)
            input_c = _state_below(tmp1_c, tmp2_c, tmp3_c, tmp4_c, self.Wc_c, self.Cc_c, self.b_c)
            # batch_size * n_h
            preact_i = _preactivate(a[0], y, self.Ua_i, self.Ub_i, self.Uc_i, input_i)
            preact_f = _preactivate(a[0], y, self.Ua_f, self.Ub_f, self.Uc_f, input_f)
            preact_o = _preactivate(a[0], y, self.Ua_o, self.Ub_o, self.Uc_o, input_o)
            preact_c = _preactivate(a[0], y, self.Ua_c, self.Ub_c, self.Uc_c, input_c)

            i = tf.sigmoid(preact_i)
            f = tf.sigmoid(preact_f)
            o = tf.sigmoid(preact_o)
            c = tf.tanh(preact_c)
            c = f * a[1] + i * c
            h = o * tf.tanh(c)
            return h, c

        # self.mask: shape n_steps * batch_size
        # self.state_below: shape n_steps * batch_size * n_x
        seqs = [self.state_below, tf.range(n_steps)]
        # print(seqs)
        initializer = (tf.zeros([batch_size, n_h], tf.float32), tf.zeros([batch_size, n_h], tf.float32))
        h_list, c_list = tf.scan(_step, seqs, initializer)
        h_list = dropout(h_list, self.keep_prob, [1, self.batch_size, n_h])
        return h_list

    def _val_test(self):
        # project word embeddings, tag embeddings and video feature to n_f space
        n_steps, n_x, n_v = self.n_steps, self.options['n_x'], self.options['n_v']
        n_h, n_f = self.options['n_h'], self.options['n_f']
        batch_size = self.batch_size
        # project tag to n_f space
        tmp2_i = tf.matmul(self.y, self.Wb_i)
        tmp2_f = tf.matmul(self.y, self.Wb_f)
        tmp2_o = tf.matmul(self.y, self.Wb_o)
        tmp2_c = tf.matmul(self.y, self.Wb_c)
        # project video to n_f space
        tmp3_i = tf.matmul(self.z, self.Ca_i)
        tmp3_f = tf.matmul(self.z, self.Ca_f)
        tmp3_o = tf.matmul(self.z, self.Ca_o)
        tmp3_c = tf.matmul(self.z, self.Ca_c)
        # project tag to n_f space in order to combine with video
        tmp4_i = tf.matmul(self.y, self.Cb_i)
        tmp4_f = tf.matmul(self.y, self.Cb_f)
        tmp4_o = tf.matmul(self.y, self.Cb_o)
        tmp4_c = tf.matmul(self.y, self.Cb_c)
        # merge word embedding and tag features
        def _step(a, elems):
            def _state_below(tmp1, tmp2, tmp3, tmp4, Wc, Cc, b):
                state_below = tf.matmul(tmp1*tmp2, Wc) + tf.matmul(tmp3*tmp4, Cc) + b
                return state_below

            def _preactivate(a, y, w1, w2, w3, x):
                p = tf.matmul(tf.matmul(a, w1) * tf.matmul(y, w2), w3) + x
                return p

            def _get_word_embed(h):
                word_prob = tf.matmul(h, self.output_w) + self.bhid
                word_idx = tf.argmax(word_prob, 1)
                word_embed = embedding_lookup(self.embeddings, word_idx)
                return word_embed

            step = elems[0]
            word_embed = tf.cond(step>0, lambda :_get_word_embed(a[0]), 
                            lambda :tf.matmul(self.z, self.c0))
            # [batch size, n_f]
            tmp1_i = tf.matmul(word_embed, self.Wa_i)
            tmp1_f = tf.matmul(word_embed, self.Wa_f)
            tmp1_o = tf.matmul(word_embed, self.Wa_o)
            tmp1_c = tf.matmul(word_embed, self.Wa_c)

            i_input = _state_below(tmp1_i, tmp2_i, tmp3_i, tmp4_i, self.Wc_i, self.Cc_i, self.b_i)
            f_input = _state_below(tmp1_f, tmp2_f, tmp3_f, tmp4_f, self.Wc_f, self.Cc_f, self.b_f)
            o_input = _state_below(tmp1_o, tmp2_o, tmp3_o, tmp4_o, self.Wc_o, self.Cc_o, self.b_o)
            c_input = _state_below(tmp1_c, tmp2_c, tmp3_c, tmp4_c, self.Wc_c, self.Cc_c, self.b_c)

            preact_i = _preactivate(a[0], self.y, self.Ua_i, self.Ub_i, self.Uc_i, i_input)
            preact_f = _preactivate(a[0], self.y, self.Ua_f, self.Ub_f, self.Uc_f, f_input)
            preact_o = _preactivate(a[0], self.y, self.Ua_o, self.Ub_o, self.Uc_o, o_input)
            preact_c = _preactivate(a[0], self.y, self.Ua_c, self.Ub_c, self.Uc_c, c_input)

            i, f = tf.sigmoid(preact_i), tf.sigmoid(preact_f)
            o, c = tf.sigmoid(preact_o), tf.tanh(preact_c)

            c = f * a[1] + i * c
            h = o * tf.tanh(c)
            return h, c

        seqs = [tf.range(n_steps)]
        initializer = (tf.zeros([batch_size, n_h], tf.float32), tf.zeros([batch_size, n_h], tf.float32))
        h_list, c_list = tf.scan(_step, seqs, initializer)
        tmp = tf.reshape(h_list, [-1, n_h])
        logits = tf.matmul(tmp, self.output_w) + self.bhid
        self.test_logits = tf.reshape(logits, [-1, batch_size, n_v])
        self.sents_sample = tf.argmax(self.test_logits, -1)

        
    def _init_weights(self):
        n_x = self.options['n_x']
        n_h = self.options['n_h']
        n_f = self.options['n_f']
        n_y = self.options['n_y']
        n_z = self.options['n_z']
        self.Wa_i = self._get_variable('Wa_i', (n_x, n_f))
        self.Wa_f = self._get_variable("Wa_f", (n_x, n_f))
        self.Wa_o = self._get_variable("Wa_o", (n_x, n_f))
        self.Wa_c = self._get_variable("Wa_c", (n_x, n_f))

        self.Wb_i = self._get_variable("Wb_i", (n_y, n_f))
        self.Wb_f = self._get_variable("Wb_f", (n_y, n_f))
        self.Wb_o = self._get_variable("Wb_o", (n_y, n_f))
        self.Wb_c = self._get_variable("Wb_c", (n_y, n_f))

        self.Wc_i = self._get_variable("Wc_i", (n_f, n_h))
        self.Wc_f = self._get_variable("Wc_f", (n_f, n_h))
        self.Wc_o = self._get_variable("Wc_o", (n_f, n_h))
        self.Wc_c = self._get_variable("Wc_c", (n_f, n_h))

        self.Ua_i = self._get_variable("Ua_i", (n_h, n_f))
        self.Ua_f = self._get_variable("Ua_f", (n_h, n_f))
        self.Ua_o = self._get_variable("Ua_o", (n_h, n_f))
        self.Ua_c = self._get_variable("Ua_c", (n_h, n_f))

        self.Ub_i = self._get_variable("Ub_i", (n_y, n_f))
        self.Ub_f = self._get_variable("Ub_f", (n_y, n_f))
        self.Ub_o = self._get_variable("Ub_o", (n_y, n_f))
        self.Ub_c = self._get_variable("Ub_c", (n_y, n_f))

        self.Uc_i = self._get_variable("Uc_i", (n_f, n_h))
        self.Uc_f = self._get_variable("Uc_f", (n_f, n_h))
        self.Uc_o = self._get_variable("Uc_o", (n_f, n_h))
        self.Uc_c = self._get_variable("Uc_c", (n_f, n_h))

        self.Ca_i = self._get_variable("Ca_i", (n_z, n_f))
        self.Ca_f = self._get_variable("Ca_f", (n_z, n_f))
        self.Ca_o = self._get_variable("Ca_o", (n_z, n_f))
        self.Ca_c = self._get_variable("Ca_c", (n_z, n_f))

        self.Cb_i = self._get_variable("Cb_i", (n_y, n_f))
        self.Cb_f = self._get_variable("Cb_f", (n_y, n_f))
        self.Cb_o = self._get_variable("Cb_o", (n_y, n_f))
        self.Cb_c = self._get_variable("Cb_c", (n_y, n_f))

        self.Cc_i = self._get_variable("Cc_i", (n_f, n_h))
        self.Cc_f = self._get_variable("Cc_f", (n_f, n_h))
        self.Cc_o = self._get_variable("Cc_o", (n_f, n_h))
        self.Cc_c = self._get_variable("Cc_c", (n_f, n_h))

        self.b_i = self._get_bias("b_i", (n_h,))
        self.b_f = self._get_bias("b_f", (n_h,))
        self.b_o = self._get_bias("b_o", (n_h,))
        self.b_c = self._get_bias("b_c", (n_h,))

    def _get_variable(self, name, shape):
        return tf.get_variable(name, shape, tf.float32, 
                               tf.glorot_normal_initializer(), 
                               collections=["SCN", 
                               tf.GraphKeys.LOCAL_VARIABLES, 
                               tf.GraphKeys.GLOBAL_VARIABLES])

    def _get_bias(self, name, shape):
        return tf.get_variable(name, shape, tf.float32, 
                               tf.zeros_initializer(), 
                               collections=["SCN", 
                               tf.GraphKeys.LOCAL_VARIABLES, 
                               tf.GraphKeys.GLOBAL_VARIABLES])


if __name__ == "__main__":
    options = {}
    options['n_x'] = 5
    options['n_f'] = 7
    options['n_h'] = 11
    options['n_y'] = 13
    options['n_z'] = 17
    options['n_steps'] = 19
    options['batch_size'] = 23
    n_steps = 19
    batch_size = 23
    model = SemanticLSTM(options)

    np.random.seed(123)

    input_x = np.random.randn(n_steps, batch_size, options['n_x'])
    input_y = np.random.randn(batch_size, options['n_y'])
    input_z = np.random.randn(batch_size, options['n_z'])
    mask = np.ones(shape=(n_steps, batch_size, options['n_h']), dtype=np.float32)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with model.graph.as_default(), tf.Session(config=config, graph=model.graph) as sess:
        pprint(tf.global_variables())
        print('\n\n\n\n')
        sess.run(tf.global_variables_initializer())
        h_list = sess.run(model.h_list, feed_dict={model.state_below: input_x, model.y: input_y, 
                                                   model.z: input_z, model.mask: mask})

    pprint(h_list)

    with model.graph.as_default():
        x = tf.get_collection('SCN')
        pprint(x)
