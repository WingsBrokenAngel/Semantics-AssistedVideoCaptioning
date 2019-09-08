#!/bin/bash
CUDA_VISIBLE_DEVICES=1 \
TF_XLA_FLAGS=--tf_xla_cpu_global_jit \
XLA_FLAGS=--xla_hlo_profile \
python train_model.py \
    --argmax=0 \
    --name=nf_1024_nh_1024_sample_4 \
    --corpus=../data/msrvtt_corpus.pkl \
    --reseco=../data/msrvtt_resnext_eco.npy \
    --tag=../tagging/msrvtt_e800_tag_feats.npy \
    --ref=../data/msrvtt_ref.pkl 
