# GPT2

OpenAI GPT-2 model was proposed in [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) by Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei and Ilya Sutskever. It’s a causal (unidirectional) transformer pretrained using language modeling on a very large corpus of ~40 GB of text data.

## Speedup by using FastSeq

- Speed on single NVIDIA-V100-16GB

  |       BatchSize       |        32       |      64       |      128       |
  |:---------------------:|:---------------:|:--------------:|:--------------:|
  |   transformers_v4.12.0 |   0.9 samples/s |   0.9 samples/s       |   OOM  |
  |   above + fastseq     |  6.0 samples/s | 7.1 samples/s |   OOM  |


### Model
The `gpt2` model weight comes from Huggingface Transformers model hub.

### Task

[CNN/DM](https://github.com/harvardnlp/sent-summary) validation data

### Setting

```bash
$ fastseq-generate-for-transformers \
    gpt2 \
    cnn_dm/raw/val.source \
    out.summary \
    --reference_path cnn_dm/raw/val.target \
    --device cuda \
    --bs 64 \
    --fp16 \
    --score_path out.score \
    --task summarization \
    --no_repeat_ngram_size 3 \
    --max_tokenizer_length 512 \
    --max_gen_length 711 \
    --postprocess_workers 3 \
    --beam 4 \
    --causal_lm
```
Baseline speed number is obtained using the above comment with "--without_fastseq_opt" flag.

### Code Example
Refer to [file](../../tests/optimizer/transformers/test_gpt2_optimizer.py).
