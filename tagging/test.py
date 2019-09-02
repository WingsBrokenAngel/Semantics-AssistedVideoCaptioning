# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from tag_net import TagNet



def main():
    tagnet = TagNet()
    with tagnet.graph.as_default():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config, graph=tagnet.graph)
        saver = tf.train.Saver()
        saver.restore(sess, './saves/msrvtt_tag_model_1000_resnext_eco.ckpt')

        data = np.load('../data/msrvtt_resnext_eco_feats.npy')
        
        batch_size = 10
        semantic = np.zeros([10000, 300], np.float32)
        for idx in range(1000):
            wanted_ops = [tagnet.pred]
            feed_dict = {tagnet.z: data[idx*batch_size:(idx+1)*batch_size], 
                         tagnet.keep_prob: 1.0}

            res = sess.run(wanted_ops, feed_dict)   
            semantic[idx*batch_size:(idx+1)*batch_size] = res[0]

        np.save('msrvtt_e1000_tag_feats', semantic)


if __name__ == "__main__":
    main()
