CUDA_VISIBLE_DEVICES=5 \
TF_XLA_FLAGS=--tf_xla_cpu_global_jit \
XLA_FLAGS=--xla_hlo_profile \
python train_model.py \
    --argmax=0 \
    --name=nf_512_nh_512_sample_4 \
    --corpus=../data/msvd_corpus.pkl \
    --ecores=../data/msvd_resnext_eco.npy \
    --tag=../tagging/msvd_semantic_tag_e1000.npy \
    --ref=../data/msvd_ref.pkl \
    --test=./saves/nf_512_nh_512_sample_4-best.ckpt
