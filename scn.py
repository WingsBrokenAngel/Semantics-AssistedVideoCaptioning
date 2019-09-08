# -*- coding: utf-8 -*-
# Author: Chen Haoran
# Date: 2019-3-24
# Date: 2019-4-23

import tensorflow as tf
from tensorflow import placeholder, Variable, get_variable, variable_scope
from tensorflow import reshape, matmul, transpose, expand_dims, concat
from tensorflow.nn import (dropout, bidirectional_dynamic_rnn, 
                           embedding_lookup, top_k, softmax, log_softmax)
from tensorflow.layers import Dense
from tensorflow import glorot_normal_initializer
from tensorflow.losses import Reduction, sparse_softmax_cross_entropy
from tensorflow.nn.rnn_cell import LSTMCell, DropoutWrapper
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
        n_z是总视频特征维数
        n_z1 是ECO视频特征维数
        n_z2 是ResNeXt视频特征维数
        n_att 是attention LSTM内部单元个数
        '''

        self.options = options
        n_x, n_h = options['n_x'], options['n_h']
        n_f, n_v = options['n_f'], options['n_v']
        n_y, n_z = options['n_y'], options['n_z']

        self.graph = tf.Graph()
        with self.graph.as_default(), variable_scope("scn", reuse=tf.AUTO_REUSE):
            tf.set_random_seed(123)
            self._init_weights()
            # self.words shape: (n_steps, batch_size)
            self.words = placeholder(tf.int32, [None, None], name="word")
            # self.mask shape: (n_steps, batch_size)
            self.mask = placeholder(tf.float32, [None, None], name="mask")
            # self.y shape: (batch_size, n_y)  
            self.y = placeholder(tf.float32, [None, n_y], name="tag_feats")
            # self.z shape: (batch_size, eco_dims+res_dims)  
            self.z = placeholder(tf.float32, [None, n_z], name="eco_res_feats")
            self.sample_prob = placeholder(tf.float32, [], name="sample_prob")
            self.keep_prob = placeholder(tf.float32, [], name="keep_prob")
            self.if_argmax = tf.constant(bool(options['flags'].argmax), dtype=tf.bool, name='if_argmax')
            self.embeddings = Variable(
                self.options['embeddings'], trainable=False, name="embeddings")
            self.whid = get_variable(
                'whid', [n_x, n_h], tf.float32, glorot_normal_initializer())
            self.bhid = get_variable(
                'bhid', [n_v], tf.float32, tf.zeros_initializer())

            # self.output_w shape: (hidden_state_size, vocabulary_size)
            self.output_w = transpose(matmul(self.embeddings, self.whid)) 
            self.c0 = get_variable(
                'C0', [n_z, n_x], tf.float32, glorot_normal_initializer())
            # revise the mask as the input for scan
            self._construct_model()


    def _construct_model(self):
        n_steps = tf.shape(self.words)[0]
        n_x, n_h, n_v = self.options['n_x'], self.options['n_h'], self.options['n_v']
        batch_size = tf.shape(self.y)[0]
        words_embed = embedding_lookup(self.embeddings, self.words) 
        # n_steps, batch_size, embed_dim
        words_embed = dropout(words_embed, self.keep_prob, (1, batch_size, n_x))
        # convert video feature from None * n_z to None * n_x
        # vid_feat_proj shape: (1, batch_size, embed_dim)
        vid_feat_proj = expand_dims(matmul(self.z, self.c0), 0)
        # use video feature as the input for the first step
        # state_below shape: (n_steps, batch_size, embed_dim)
        state_below = concat([vid_feat_proj, words_embed[:-1]], 0)

        # h_list shape: (n_steps, batch_size, h_dims)
        self.h_list = self._decoder_layer(state_below)
        self.h_list_reshape = reshape(self.h_list, [-1, n_h])
        # logits shape: (n_steps*batch_size, n_vocabulary)
        self.logits = matmul(self.h_list_reshape, self.output_w) + self.bhid  
        logits_reshaped = reshape(self.logits, [-1, batch_size, n_v])
        self.sents = tf.argmax(logits_reshaped, -1)
        # w_reshape shape: (n_steps, batch_size)
        weighted_mask = self.mask / (tf.reduce_sum(self.mask, 0, keepdims=True)**0.7)
        self.loss = sparse_softmax_cross_entropy(logits=logits_reshaped+1e-8, 
                                                 labels=self.words, 
                                                 weights=weighted_mask, 
                                                 reduction=tf.losses.Reduction.SUM)
        self.train_loss = self.loss / tf.cast(batch_size, tf.float32)

        test_h_list = self._test_layer(matmul(self.z, self.c0))
        test_h_list = reshape(test_h_list, (-1, n_h))
        test_logits = matmul(test_h_list, self.output_w) + self.bhid
        test_logits = reshape(test_logits, (-1, batch_size, n_v))
        self.test_sents = tf.argmax(test_logits, -1)


    def _decoder_layer(self, state_below):
        """state below: size of (n_steps, batch_size, n_x)
        """
        n_x = self.options['n_x']
        n_h = self.options['n_h']
        n_f = self.options['n_f']
        # batch_size * n_f
        y = dropout(self.y, self.keep_prob)
        tmp2_i = matmul(y, self.Wb_i)
        tmp2_f = matmul(y, self.Wb_f)
        tmp2_o = matmul(y, self.Wb_o)
        tmp2_c = matmul(y, self.Wb_c)
        # batch_size * n_f
        z = dropout(self.z, self.keep_prob)
        tmp3_i = matmul(z, self.Ca_i)
        tmp3_f = matmul(z, self.Ca_f)
        tmp3_o = matmul(z, self.Ca_o)
        tmp3_c = matmul(z, self.Ca_c)
        # batch_size * n_f
        tmp4_i = matmul(y, self.Cb_i)
        tmp4_f = matmul(y, self.Cb_f)
        tmp4_o = matmul(y, self.Cb_o)
        tmp4_c = matmul(y, self.Cb_c)

        def _state_below(tmp1, tmp2, tmp3, tmp4, Wc, Cc, b):
            # print('tmp1:', tmp1, 'tmp2:', tmp2, 'tmp3:', tmp3, 'tmp4:', tmp4)
            state_b = matmul(tmp1 * tmp2, Wc) + matmul(tmp3 * tmp4, Cc) + b
            return state_b

        def _step(a, b):
            print('in decoder layer', a, b)
            word_embed1, step = b[0], b[1]
            def _preactivate(a, y, w1, w2, w3, x):
                p = matmul(matmul(a, w1) * matmul(y, w2), w3) + x
                return p

            def _get_word_embed(h):
                word_logit = matmul(h, self.output_w) + self.bhid
                word_chosen1 = tf.argmax(word_logit, 1)
                word_chosen2 = tf.multinomial(word_logit, 1)
                word_chosen2 = tf.squeeze(word_chosen2)
                word_chosen = tf.cond(
                    self.if_argmax, lambda: word_chosen1, lambda: word_chosen2)
                return embedding_lookup(self.embeddings, word_chosen)

            word_embed = tf.cond((tf.random_uniform([])>=self.sample_prob) | tf.equal(step,0), 
                                 lambda: word_embed1, 
                                 lambda: _get_word_embed(a[0]))
            # batch_size = tf.shape(word_embed)[0]
            word_embed = tf.reshape(word_embed, (-1, n_x))

            # batch_size * n_f
            tmp1_i = matmul(word_embed, self.Wa_i)
            tmp1_f = matmul(word_embed, self.Wa_f)
            tmp1_o = matmul(word_embed, self.Wa_o) 
            tmp1_c = matmul(word_embed, self.Wa_c)
            # batch_size * n_h
            input_i = _state_below(
                tmp1_i, tmp2_i, tmp3_i, tmp4_i, self.Wc_i, self.Cc_i, self.b_i)
            input_f = _state_below(
                tmp1_f, tmp2_f, tmp3_f, tmp4_f, self.Wc_f, self.Cc_f, self.b_f)
            input_o = _state_below(
                tmp1_o, tmp2_o, tmp3_o, tmp4_o, self.Wc_o, self.Cc_o, self.b_o)
            input_c = _state_below(
                tmp1_c, tmp2_c, tmp3_c, tmp4_c, self.Wc_c, self.Cc_c, self.b_c)
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
        # state_below: shape n_steps * batch_size * n_x
        n_steps = tf.shape(state_below)[0]
        steps = tf.range(n_steps, dtype=tf.int32)
        elems = [state_below, steps]
        batch_size = tf.shape(state_below)[1]
        init_t = (tf.zeros([batch_size, n_h]), tf.zeros([batch_size, n_h]))
        h_list, c_list = tf.scan(_step, elems, init_t)
        h_list = dropout(h_list, self.keep_prob, [1, batch_size, n_h])
        return h_list


    def _test_layer(self, vid_embed):
        """vid_embed: size of (batch_size, n_x)
        """
        n_x = self.options['n_x']
        n_h = self.options['n_h']
        n_f = self.options['n_f']
        # batch_size * n_f
        y, z = self.y, self.z
        tmp2_i = matmul(y, self.Wb_i)
        tmp2_f = matmul(y, self.Wb_f)
        tmp2_o = matmul(y, self.Wb_o)
        tmp2_c = matmul(y, self.Wb_c)
        # batch_size * n_f
        tmp3_i = matmul(z, self.Ca_i)
        tmp3_f = matmul(z, self.Ca_f)
        tmp3_o = matmul(z, self.Ca_o)
        tmp3_c = matmul(z, self.Ca_c)
        # batch_size * n_f
        tmp4_i = matmul(y, self.Cb_i)
        tmp4_f = matmul(y, self.Cb_f)
        tmp4_o = matmul(y, self.Cb_o)
        tmp4_c = matmul(y, self.Cb_c)

        def _state_below(tmp1, tmp2, tmp3, tmp4, Wc, Cc, b):
            # print('tmp1:', tmp1, 'tmp2:', tmp2, 'tmp3:', tmp3, 'tmp4:', tmp4)
            state_b = matmul(tmp1 * tmp2, Wc) + matmul(tmp3 * tmp4, Cc) + b
            return state_b

        def _step(a, b):
            step = b[0]
            print('in test layer', a, b)
            def _preactivate(a, y, w1, w2, w3, x):
                p = matmul(matmul(a, w1) * matmul(y, w2), w3) + x
                return p

            def _get_word_embed(h):
                word_logit = matmul(h, self.output_w) + self.bhid
                word_chosen = tf.argmax(word_logit, 1)
                return embedding_lookup(self.embeddings, word_chosen)

            word_embed = tf.cond(tf.equal(step, 0), 
                                 lambda: vid_embed, 
                                 lambda: _get_word_embed(a[0]))
            # batch_size = tf.shape(word_embed)[0]
            word_embed = tf.reshape(word_embed, (-1, n_x))
            # batch_size * n_f
            tmp1_i = matmul(word_embed, self.Wa_i)
            tmp1_f = matmul(word_embed, self.Wa_f)
            tmp1_o = matmul(word_embed, self.Wa_o)
            tmp1_c = matmul(word_embed, self.Wa_c)
            # batch_size * n_h
            input_i = _state_below(
                tmp1_i, tmp2_i, tmp3_i, tmp4_i, self.Wc_i, self.Cc_i, self.b_i)
            input_f = _state_below(
                tmp1_f, tmp2_f, tmp3_f, tmp4_f, self.Wc_f, self.Cc_f, self.b_f)
            input_o = _state_below(
                tmp1_o, tmp2_o, tmp3_o, tmp4_o, self.Wc_o, self.Cc_o, self.b_o)
            input_c = _state_below(
                tmp1_c, tmp2_c, tmp3_c, tmp4_c, self.Wc_c, self.Cc_c, self.b_c)
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
        # state_below: shape n_steps * batch_size * n_x
        steps = tf.range(self.options['n_steps'], dtype=tf.int32)
        elems = [steps]
        batch_size = tf.shape(y)[0]
        init_t = (tf.zeros([batch_size, n_h]), tf.zeros([batch_size, n_h]))
        h_list, c_list = tf.scan(_step, elems, init_t)
        return h_list


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


class F:
    def __init__(self):
        self.argmax = True


if __name__ == "__main__":
    options = {}
    options['embeddings'] = np.random.randn(8000, 5).astype(np.float32)
    options['b_w'] = 3
    options['n_x'] = 5
    options['n_f'] = 7
    options['n_h'] = 11
    options['n_y'] = 13
    options['n_z'] = 3584
    options['n_z1'] = 1536
    options['n_z2'] = 2048
    options['n_s'] = 19
    options['n_v'] = 8000
    options['n_steps'] = 19
    options['flags'] = F()
    n_steps = 19
    batch_size = 23

    model = SemanticLSTM(options)
    np.random.seed(123)

    words = (np.abs(np.random.randn(n_steps, batch_size))*1000).astype(np.int32)
    y = np.random.randn(batch_size, options['n_y'])
    z1 = np.random.randn(batch_size, options['n_z1'])
    z2 = np.random.randn(batch_size, options['n_s'], options['n_z2'])
    mask = np.ones(shape=(n_steps, batch_size), dtype=np.float32)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with model.graph.as_default(), tf.Session(config=config, graph=model.graph) as sess:
        pprint(tf.global_variables())
        print('\n\n\n\n')
        sess.run(tf.global_variables_initializer())
        h_list = sess.run(model.h_list, feed_dict={model.words: words,
                                                   model.y: y, 
                                                   model.z1: z1, 
                                                   model.z2: z2,
                                                   model.mask: mask, 
                                                   model.batch_size: batch_size, 
                                                   model.sample_prob: 0.5, 
                                                   model.keep_prob: 0.5, 
                                                   model.n_steps: options['n_steps'], 
                                                })

    pprint(h_list)

    with model.graph.as_default():
        x = tf.get_collection('SCN')
        pprint(x)
