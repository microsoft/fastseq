# BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension

https://arxiv.org/pdf/1910.13461.pdf

## Introduction

BART is sequence-to-sequence model trained with denoising as pretraining objective. We show that this pretraining objective is more generic and show that we can match [RoBERTa](../roberta) results on SQuAD and GLUE and gain state-of-the-art results on summarization (XSum, CNN dataset), long form generative question answering (ELI5) and dialog response genration (ConvAI2). See the associated paper for more details.

## Speedup BART (Fairseq version) by using FastSeq

- CNN daily mail validation data, NVIDIA-V100-16GB

  |     BatchSize    |       32      |        64       |      128       |
  |:----------------:|:-------------:|:---------------:|:--------------:|
  | fairseq-0.9.0    | 2.7 samples/s |       OOM       |      OOM       |
  | above + fastseq  | 9.0 samples/s | 12.5 samples/s  | 14.5 samples/s |

### Model

Model | Description | # params | Download
---|---|---|---
`bart.base` | BART model with 6 encoder and decoder layers | 140M | [bart.base.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/bart.base.tar.gz)
`bart.large` | BART model with 12 encoder and decoder layers | 400M | [bart.large.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz)
`bart.large.mnli` | `bart.large` finetuned on `MNLI` | 400M | [bart.large.mnli.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/bart.large.mnli.tar.gz)
`bart.large.cnn` | `bart.large` finetuned on `CNN-DM` | 400M | [bart.large.cnn.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/bart.large.cnn.tar.gz)
`bart.large.xsum` | `bart.large` finetuned on `Xsum` | 400M | [bart.large.xsum.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/bart.large.xsum.tar.gz)

`bart.large.cnn` is used in speed benchmark.

### Task
[CNN/DM](https://github.com/harvardnlp/sent-summary) validation data

### Setting

```bash
$ fastseq-generate-for-fairseq \
      cnn_dm.1k/len-1024.bin \
      --path bart.large.cnn/model.pt \
      --fp16 \
      --task translation \
      --batch-size BATCH_SIZE \
      --gen-subset valid \
      --truncate-source  \
      --bpe gpt2 \
      --beam 4 \
      --num-workers 4 \
      --min-len 55 \
      --max-len-b 140 \
      --no-repeat-ngram-size 3 \
      --lenpen 2.0
```

To get the baseline fairseq's speed number, replace `fastseq-generate-for-fairseq` by `fairseq-generate`.

## Speedup BART (Huggingface Transformers version) by using FastSeq

- CNN daily mail validation data, NVIDIA-V100-16GB

  |      BatchSize      |       32      |       64       |       128      |
  |:-------------------:|:-------------:|:--------------:|:--------------:|
  | transformers-3.0.2  | 3.4 samples/s |      OOM       |      OOM       |
  |  above + fastseq    | 5.2 samples/s | 6.2 samples/s  | 6.4 samples/s  |


### Model
`facebook/bart-large-cnn` from model hub.

### Task
[CNN/DM](https://github.com/harvardnlp/sent-summary) validation data

### Setting

```bash
$ fastseq-generate-for-transformers \
    facebook/bart-large-cnn \
    cnn_dm.1k/val.source \
    out.summary \
    --reference_path cnn_dm/val.target \
    --device cuda \
    --bs 128 \
    --fp16 \
    --score_path out.score \
    --task summarization
```

To get the baseline transformers' speed number, we can either add option `--without_fastseq_opt` or use [tool](https://github.com/huggingface/transformers/tree/master/examples/seq2seq) provided in Transformers GitHub repository.

