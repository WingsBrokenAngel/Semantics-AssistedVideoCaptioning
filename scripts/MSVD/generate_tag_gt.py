# -*- coding: utf-8 -*-
# Author: Haoran Chen
# Date: 2019-4-28

import numpy as np
import pickle
from pprint import pprint



def main(tag_gt, word2idx_tag, idx2word, ref):
    for idx, sent in enumerate(ref[0]):
        vid_idx = ref[1][idx]
        for w in sent:
            if idx2word[w] in word2idx_tag:
                tag_gt[vid_idx, word2idx_tag[idx2word[w]]] = 1


if __name__ == "__main__":
    tag_gt = np.zeros([1970, 300], np.int32)

    with open('youtube_corpus.pkl', 'rb') as fo:
        data = pickle.load(fo)
        train, val, test = data[0], data[1], data[2]
        word2idx, idx2word = data[3], data[4]

    with open('tag_idx_word.pkl', 'rb') as fo:
        idx2word_tag, word2idx_tag = pickle.load(fo)

    main(tag_gt, word2idx_tag, idx2word, data[0])
    main(tag_gt, word2idx_tag, idx2word, data[1])
    main(tag_gt, word2idx_tag, idx2word, data[2])

    np.save('msvd_tag_gt', tag_gt)