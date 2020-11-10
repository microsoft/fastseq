# ProphetNet

A pre-trained language model for sequence-to-sequence learning with a novel self-supervised objective called future n-gram prediction.
- [Paper](https://arxiv.org/pdf/2001.04063)
- [Open Source](https://github.com/microsoft/ProphetNet)

## Speedup by using FastSeq

- CNN daily mail validation data, NVIDIA-V100-16GB

  |       BatchSize      |       32      |        64       |      128       |
  |:--------------------:|:-------------:|:---------------:|:--------------:|
  |      prophetnet      | 2.7 samples/s |  3.1 samples/s  |      OOM       |
  |   above + fastseq    | 5.5 samples/s |  8.4 samples/s  | 10.3 samples/s |


### Model
ProphetNet-large-160GB (fine-tuned on CNN/Daily Mail with 9 epochs) [link](https://drive.google.com/file/d/14v0HMc7obh_5aPFSFWzcr_nZCrK49Sey/view)

### Task
[CNN/DM](https://github.com/harvardnlp/sent-summary) validation data

### Setting

```bash
$ fastseq-generate-for-fairseq \
      cnn_dm_bert.1k/len-512.bin \
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
To get baseline speed number which doesn't use FastSeq optimizations, replace `fastseq-generate-for-fairseq` by `fairseq-generate`.

### Code Example
Refer to [file](../../tests/models/test_prophetnet_fs.py).
