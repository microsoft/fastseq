# Scaling Neural Machine Translation (Ott et al., 2018)
https://arxiv.org/abs/1806.00187

## Speedup by using FastSeq

- Speed on single NVIDIA-V100-16GB

  |     BatchSize    |      256       |      512       |      1024      |
  |:----------------:|:--------------:|:--------------:|:--------------:|
  | fairseq-0.10.2    |  144.5 samples/s  |      OOM       |      OOM       |
  | above + fastseq  | 364.1 samples/s  |  402.1 samples/s |  422.8 samples/s |

### Training a new model on WMT'16 En-De

##### 1. Preprocess the dataset with a joined dictionary (optional)
```bash
TOK=tok
BIN=bin
rm -rf $TOK $BIN
mkdir -p $TOK $BIN
# train
wget -O $TOK/train.bpe.source https://fastseq.blob.core.windows.net/data/tasks/wmt16_en_de_bpe32k/tok/train.bpe.source
wget -O $TOK/train.bpe.target https://fastseq.blob.core.windows.net/data/tasks/wmt16_en_de_bpe32k/tok/train.bpe.target
# val
wget -O $TOK/val.bpe.source https://fastseq.blob.core.windows.net/data/tasks/wmt16_en_de_bpe32k/tok/val.bpe.source
wget -O $TOK/val.bpe.target https://fastseq.blob.core.windows.net/data/tasks/wmt16_en_de_bpe32k/tok/val.bpe.target
# test
wget -O $TOK/test.bpe.source https://fastseq.blob.core.windows.net/data/tasks/wmt16_en_de_bpe32k/tok/test.bpe.source
wget -O $TOK/test.bpe.target https://fastseq.blob.core.windows.net/data/tasks/wmt16_en_de_bpe32k/tok/test.bpe.target
fairseq-preprocess \
    --source-lang source --target-lang target \
    --validpref $TOK/val.bpe \
    --trainpref $TOK/train.bpe \
    --testpref $TOK/test.bpe \
    --destdir $BIN/ \
    --nwordssrc 32768 --nwordstgt 32768 \
    --joined-dictionary \
    --workers 20
```

Or you can download the preprocessed data directly
```bash
TOK=tok
BIN=bin
rm -rf $TOK $BIN
mkdir -p $TOK $BIN
wget -O $BIN/dict.source.txt https://fastseq.blob.core.windows.net/data/tasks/wmt16_en_de_bpe32k/bin/dict.source.txt
wget -O $BIN/dict.target.txt https://fastseq.blob.core.windows.net/data/tasks/wmt16_en_de_bpe32k/bin/dict.target.txt
wget -O $BIN/test.source-target.source.bin https://fastseq.blob.core.windows.net/data/tasks/wmt16_en_de_bpe32k/bin/test.source-target.source.bin
wget -O $BIN/test.source-target.source.idx https://fastseq.blob.core.windows.net/data/tasks/wmt16_en_de_bpe32k/bin/test.source-target.source.idx
wget -O $BIN/test.source-target.target.bin https://fastseq.blob.core.windows.net/data/tasks/wmt16_en_de_bpe32k/bin/test.source-target.target.bin
wget -O $BIN/test.source-target.target.idx https://fastseq.blob.core.windows.net/data/tasks/wmt16_en_de_bpe32k/bin/test.source-target.target.idx
```

##### 2. Train a model (optional)
```bash
fairseq-train \
    bin/ \
    --arch transformer_vaswani_wmt_en_de_big \
    --share-all-embeddings \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr 0.0005 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --warmup-init-lr 1e-07 \
    --dropout 0.3 \
    --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-tokens 3584 \
    --fp16
```

Or you can download model directly
```bash
wget -O checkpoints/checkpoint_last.pt https://fastseq.blob.core.windows.net/data/models/wmt16.en.de.32k/model.pt
```
Note this is a test model for speed evaluation, not good enough for real translation.


### Setting

```bash
$ fastseq-generate-for-fairseq \
    bin \
    --path checkpoints/checkpoint_last.pt \
    --batch-size BATCH_SIZE \
    --beam 4 \
    --lenpen 0.6 \
    --remove-bpe \
    --gen-subset test \
    --postprocess-workers 5
```
To get baseline speed number which doesn't use FastSeq optimizations, replace `fastseq-generate-for-fairseq` by `fairseq-generate` and remove argument `--postprocess-workers 5` since it is only provided by fastseq.
