# Unilm

- Paper
	- [UnilmV1](https://arxiv.org/abs/1905.03197)
	- [UnilmV2](https://arxiv.org/abs/2002.12804)
- [Open Source](https://github.com/microsoft/unilm)

## Speedup by using FastSeq

- CNN daily mail validation data, NVIDIA-V100-16GB

  |       BatchSize       |        64       |      128       |
  |:---------------------:|:---------------:|:--------------:|
  |   transformers_v4.11.3 |   1.7 samples/s |      OOM       |
  |   above + fastseq     |  13.8 samples/s | 16.4 samples/s |

### Model
cnndm-unilm-base-cased (fine-tuned on CNN/Daily Mail) [link](https://unilm.blob.core.windows.net/ckpt/cnndm.unilm1-base-cased.bin)

### Task
[CNN/DM](https://github.com/harvardnlp/sent-summary) validation data

### Train / Finetune
- Refer to offical repo
- By Fastseq
```python
from fastseq.models.unilm_hf.tokenization_unilm import UnilmTokenizer
from fastseq.models.unilm_hf.modeling_unilm import UnilmForSeq2Seq

model = UnilmForSeq2Seq.from_pretrained('cnndm-unilm-base-cased')
tokenizer = UnilmTokenizer.from_pretrained('cnndm-unilm-base-cased')

# prepare input for model training
input_ids, attention_mask, token_type_ids, position_ids = preprocess(batch_data, tokenizer)

# model output & loss for backward
output = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids)
loss = loss_fn(output, label)
```

### Setting
```bash
$ fastseq-generate-for-transformer \
      cnndm-unilm-base-cased \
      path/to/cnndm_rawtext_input \
      path/to/cnndm_summary_output \
      --fp16 \
      --bs 128 \
      --max_length 608 \
      --max_gen_length 160 \
      --no_repeat_ngram_size 3
```
To get baseline speed number which doesn't use FastSeq optimizations, refer to [Unilm Offical Repo](https://github.com/microsoft/unilm/tree/master/s2s-ft)`.

### Code Example

Refer to [file](../../tests/models/test_unilm_hf.py).
