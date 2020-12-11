# DistilBART

http://arxiv.org/abs/2010.13002

More info can be found [here](https://github.com/huggingface/transformers/blob/master/examples/seq2seq/README.md#distilbart).

## Speedup DistilBART (Huggingface Transformers version) by using FastSeq

- Speed on single NVIDIA-V100-16GB

  |      BatchSize      |       64       |       128      |
  |:-------------------:|:--------------:|:--------------:|
  | transformers-3.0.2  | 4.3 samples/s  |      OOM       |
  |  above + fastseq    | 16.5 samples/s  | 18.3 samples/s  |


### Model
`sshleifer/distilbart-cnn-12-6` from model hub.

### Task
[CNN/DM](https://github.com/harvardnlp/sent-summary) validation data

### Setting

```bash
$ fastseq-generate-for-transformers \
    sshleifer/distilbart-cnn-12-6 \
    cnn_dm.1k/val.source \
    out.summary \
    --reference_path cnn_dm.1k/val.target \
    --device cuda \
    --bs BATCH_SIZE \
    --fp16 \
    --score_path out.score \
    --task summarization
```

To get the baseline transformers' speed number, we can either add option `--without_fastseq_opt` or use [tool](https://github.com/huggingface/transformers/tree/master/examples/seq2seq) provided in Transformers GitHub repository.

