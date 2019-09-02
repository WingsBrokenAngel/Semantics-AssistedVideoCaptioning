CUDA_VISIBLE_DEVICES=1 \
TF_XLA_FLAGS=--tf_xla_cpu_global_jit \
XLA_FLAGS=--xla_hlo_profile \
python train_model.py \
    --argmax=0 \
    --name=test \
    --corpus=./data/msvd_corpus.pkl \
    --reseco=./data/msvd_resnext_eco.npy \
    --tag=./tagging/msvd_semantic_tag_e1000.npy \
    --ref=./data/msvd_ref.pkl
