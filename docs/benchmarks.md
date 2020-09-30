# Benchmarks

All the following benchmarking experiments run on NVIDIA-V100-16GB with [the docker](../docker/Dockerfile). The results can be reproduced by using the command below:

```bash
$ cd benchmarks && bash run_all_benchmarks.sh
```

## ProphetNet

- CNN daily mail validation data, NVIDIA-V100-16GB

  |       BatchSize      |       32      |        64       |      128       |
  |:--------------------:|:-------------:|:---------------:|:--------------:|
  |      prophetnet      | 2.7 samples/s |  3.1 samples/s  |      OOM       |
  | prophetnet + fastseq | 5.5 samples/s |  8.4 samples/s  | 10.3 samples/s |

with setting:

```bash
$ fastseq-generate-for-fairseq \
      cnn_dm_bert.1k/len-1024.bin \
      --path prophetnet/model.pt \
      --fp16 \
      --task translation_prophetnet \
      --batch-size BATCH_SIZE \
      --beam 4 \
      --num-workers 4 \
      --min-len 55 \
      --max-len-b 140 \
      --no-repeat-ngram-size 3 \
      --lenpen 2.0 \
      --remove-bpe \
      --gen-subset valid \
```

## BART from Fairseq

- CNN daily mail validation data, NVIDIA-V100-16GB

  |     BatchSize    |       32      |        64       |      128       |
  |:----------------:|:-------------:|:---------------:|:--------------:|
  | fairseq-0.9.0    | 2.7 samples/s |       OOM       |      OOM       |
  | above + fastseq  | 9.0 samples/s | 12.5 samples/s  | 14.5 samples/s |

with setting:

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

## BART from Transformers

- CNN daily mail validation data, NVIDIA-V100-16GB

  |      BatchSize      |       32      |       64       |       128      |
  |:-------------------:|:-------------:|:--------------:|:--------------:|
  | transformers-3.0.2  | 3.4 samples/s |      OOM       |      OOM       |
  |  above + fastseq    | 5.2 samples/s | 6.2 samples/s  | 6.4 samples/s  |
  | transformers-2.11.0 | 2.5 samples/s |      OOM       |      OOM       |
  |  above + fastseq    | 4.4 samples/s | 5.3 samples/s  | >5.3 samples/s |

(numbers for 2.11.0 needs to be updated based on docker env.)

with setting:

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

## WMT from Fairseq
- [WMT16 En-De](https://github.com/pytorch/fairseq/tree/master/examples/scaling_nmt) model

  |     BatchSize    |      256       |      512       |      1024      |
  |:----------------:|:--------------:|:--------------:|:--------------:|
  | fairseq-0.9.0    |  84 samples/s  |      OOM       |      OOM       |
  | above + fastseq  | 129 samples/s  |  131 samples/s |  135 samples/s |


with setting:

```bash
$ fastseq-generate-for-fairseq \
      wmt14.en-fr.joined-dict.newstest2014/ \
      --path wmt14.en-fr.joined-dict.transformer/model.pt \
      --beam 4 \
      --lenpen 0.6 \
      --remove-bpe \
      --batch-size 32
```

To get the fairseq's speed number, replace `fastseq-generate-for-fairseq` by `fairseq-generate`.
