# T5
The T5 model was presented in [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683.pdf) by Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu.

[Model](https://huggingface.co/transformers/model_doc/t5.html) based on Transformers was used. The original code can be found [here](https://github.com/google-research/text-to-text-transfer-transformer).

## Speedup by using FastSeq

- Speed on single NVIDIA-V100-16GB

  |       BatchSize      |        64       |      128       |
  |:--------------------:|:---------------:|:--------------:|
  |   ransformers_v3.0.2 |  4.8 samples/s  |      OOM       |
  |   above + fastseq    |  7.0 samples/s  | 7.5 samples/s  |


### Model
`t5-base` from Huggingface Transformers model hub.

### Task - WMT16 English-Romanian Translation

download with this command:
```bash
wget https://cdn-datasets.huggingface.co/translation/wmt_en_ro.tar.gz
tar -xzvf wmt_en_ro.tar.gz
export ENRO_DIR=${PWD}/wmt_en_ro
```
this should make a directory called `wmt_en_ro/` with 6 files.

### Setting

```bash
$ fastseq-generate-for-transformers \
    t5-base \
    wmt_en_ro/val.source \
    out.summary \
    --reference_path cnn_dm/val.target \
    --device cuda \
    --bs BATCH_SIZE \
    --fp16 \
    --score_path out.score \
    --task translation_en_to_ro \
    --no_repeat_ngram_size 3
```
To get the baseline transformers' speed number, we can either add option `--without_fastseq_opt` or use [tool](https://github.com/huggingface/transformers/tree/master/examples/seq2seq) provided in Transformers GitHub repository.
