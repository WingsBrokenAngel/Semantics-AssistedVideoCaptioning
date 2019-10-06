# -*- coding: utf-8 -*-
# Author: Haoran Chen
# Date: 2019-4-28

import numpy as np
import zipfile as zf
import json
from nltk.tokenize import TreebankWordTokenizer
import pickle
from pprint import pprint


def main(tag_gt, word2idx, zipname):
    with zf.ZipFile(zipname) as myzip:
        namelist = myzip.namelist()
        print('namelist:', namelist)
        datainfo = myzip.open(namelist[-1], 'r')
        info_dict = json.load(datainfo)
        sentences = info_dict['sentences']
        tokenizer = TreebankWordTokenizer()
        for sentence in sentences:
            video_id = sentence['video_id']
            video_idx = int(video_id[5:])
            caption = sentence['caption']
            words = tokenizer.tokenize(caption)
            for word in words:
                if word in word2idx:
                    tag_gt[video_idx, word2idx[word]] = 1


if __name__ == "__main__":
    tag_gt = np.zeros([10000, 300], np.int32)

    with open('youtube_corpus.pkl', 'rb') as fo:
        data = pickle.load(fo)
        train, val, test = data[0], data[1], data[2]
        word2idx, idx2word = data[3], data[4]
        n_v = len(idx2word)
        counter = np.zeros([n_v], np.int32)

        for sent in train[0]:
            for w in sent:
                counter[w] += 1

        for sent in val[0]:
            for w in sent:
                counter[w] += 1

        indices = np.argsort(counter)[::-1]

        counter_sorted = np.sort(counter)[::-1]

        word_sorted = []
        for i in indices:
            word_sorted.append(idx2word[i])

        selected = [4,5,8,13,16,17,20,21,23,24,26,27,28,29,30,31,32,34,35,36,38,41,42] + \
                    list(range(44,51)) + list(range(52,56)) + [57,59,62,63,66,70,71,72,74] + \
                    [76,77,78,80,82,83,84,85,87,88,90,91,93,94,95,96,97,98,99,100,101] + \
                    [102,103,106,108,109,110,111,112,113,114,115,116,117,120,121,122,123] + \
                    [124,125,126,127,128,129,130,131,133,136,138,139,140,141,142] + \
                    [143,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160] +\
                    list(range(162,173)) + [175,176,177,178,179,180,181,183,185,186,188,189] +\
                    list(range(190,203)) + list(range(204,215)) + list(range(216,224)) + [226,227,228,230] +\
                    [231,232,233,234,236] + list(range(238,253)) + list(range(253,258)) + [259,260,261] +\
                    [263,265] + list(range(268,278)) + [279,282,283,284,286,288,289,290] +\
                    list(range(291,309)) + list(range(310,313)) + [315,317,318,319,320,321,322] + [326,327] +\
                    [328,329,331,332,334,335,336,338,339,340] + list(range(341,363)) +[364,366,367] +\
                    list(range(368,374)) + [376,377,378,379,380,381,382,383,384]

        key_words = []
        for i in selected:
            key_words.append(word_sorted[i])

        idx2word_tag, word2idx_tag = {}, {}

        for idx, word in enumerate(key_words):
            idx2word_tag[idx] = word
            word2idx_tag[word] = idx

    with open("tag_idx_word.pkl", "wb") as fo:
        pickle.dump([idx2word_tag, word2idx_tag], fo, -1)

    main(tag_gt, word2idx_tag, 'train_val_annotation.zip')
    main(tag_gt, word2idx_tag, 'test_videodatainfo.json.zip')

    pprint(tag_gt[0])
    np.save('msr_vtt_tag_gt', tag_gt)