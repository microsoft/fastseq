# T5
The T5 model was presented in [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683.pdf) by Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu.

[Model](https://huggingface.co/transformers/model_doc/t5.html) based on Transformers was used. The original code can be found [here](https://github.com/google-research/text-to-text-transfer-transformer).

## Speedup by using FastSeq

- Speed on single NVIDIA-V100-16GB

  |       BatchSize      |        64       |      128       |
  |:--------------------:|:---------------:|:--------------:|
  |   transformers_v4.12.0 |  8.7 samples/s  |      OOM       |
  |   above + fastseq    |  19.5 samples/s | 31.3 samples/s  |


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
    --postprocess_workers 3
```
Baseline speed number is obtained by running [Transformers v4.12.0 code](https://github.com/huggingface/transformers/blob/b0892fa0e8df02d683e05e625b3903209bff362d/examples/seq2seq/run_eval.py).

### Code Example
Refer to [file](../../tests/optimizer/transformers/test_t5_optimizer.py).
