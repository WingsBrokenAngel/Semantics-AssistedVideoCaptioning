#!/bin/bash
CUDA_VISIBLE_DEVICES=3 TF_XLA_FLAGS=--tf_xla_cpu_global_jit XLA_FLAGS=--xla_hlo_profile python train_model.py --argmax=0 --name=test \
    --corpus=/data2/chenhaoran/VideoCaptioner3/SCN/data/msrvtt_corpus.pkl \
    --reseco=/data2/chenhaoran/VideoCaptioner3/SCN/data/msrvtt_resnext_eco.npy \
    --tag=/data2/chenhaoran/VideoCaptioner3/SCN/tagging/msrvtt_e800_tag_feats.npy \
    --ref=/data2/chenhaoran/VideoCaptioner3/SCN/data/msrvtt_ref.pkl
