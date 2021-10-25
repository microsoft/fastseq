# ProphetNet

A pre-trained language model for sequence-to-sequence learning with a novel self-supervised objective called future n-gram prediction.
- [Paper](https://arxiv.org/pdf/2001.04063)
- [Open Source](https://github.com/microsoft/ProphetNet)

## Speedup by using FastSeq

- CNN daily mail validation data, NVIDIA-V100-16GB

  |       BatchSize      |       32      |        64       |      128       |
  |:--------------------:|:-------------:|:---------------:|:--------------:|
  |      prophetnet (fs 0.9.0)      | 2.4 samples/s |  2.8 samples/s  |      OOM       |
  |   above + fastseq    | ??? samples/s |  9.1 samples/s  | 11.9 samples/s |


### Model
ProphetNet-large-160GB (fine-tuned on CNN/Daily Mail with 9 epochs) [link](https://drive.google.com/file/d/14v0HMc7obh_5aPFSFWzcr_nZCrK49Sey/view)

### Task
[CNN/DM](https://github.com/harvardnlp/sent-summary) validation data

### Setting

```bash
$ fastseq-generate-for-fairseq \
      cnn_dm_bert/len-512.bin \
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

### Generate the binary data

```bash
bash generate_binary_data_for_prophetnet.sh INPUT_DATA_DIR
```
