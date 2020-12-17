#!/bin/bash

data_dir=$1
raw_dir=$data_dir/raw
max_len=512
bpe_dir=$data_dir/bpe
bin_dir=$data_dir/len-${max_len}.bin/

if [ ! -d $bpe_dir ]; then
    mkdir $bpe_dir
fi

if [ ! -d $bin_dir ]; then
    mkdir $bin_dir
fi

raw_data=("train" "training" "valid" "test" "dev")
for data_name in ${raw_data[*]}; do
    python preprocess_cnn_dm.py \
        --fin $raw_dir/$data_name.article \
        --fout $bpe_dir/$data_name.source-target.bpe.source\
        --max_len ${max_len}

    python preprocess_cnn_dm.py \
        --fin $raw_dir/$data_name.summary \
        --fout $bpe_dir/$data_name.source-target.bpe.target\
        --max_len ${max_len}
done

fairseq-preprocess \
    --user-dir ../../fastseq/models/prophetnet_fs/ \
    --task translation_prophetnet \
    --source-lang source \
    --target-lang target \
    --srcdict $data_dir/dict.source.txt \
    --tgtdict $data_dir/dict.target.txt \
    --trainpref $bpe_dir/train.source-target.bpe \
    --validpref $bpe_dir/valid.source-target.bpe \
    --testpref $bpe_dir/test.source-target.bpe \
    --destdir $bin_dir/ \
    --workers 60
