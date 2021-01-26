# Unilm

- Paper
	- [UnilmV1](https://arxiv.org/abs/1905.03197)
	- [UnilmV2](https://arxiv.org/abs/2002.12804)
- [Open Source](https://github.com/microsoft/unilm)

## Speedup by using FastSeq

- CNN daily mail validation data, NVIDIA-2080TI

  |       BatchSize      |       32      |        64       |      128       |
  |:--------------------:|:-------------:|:---------------:|:--------------:|
  |      unilm           |  - samples/s  |   1.2 samples/s  |      OOM       |
  |   above + fastseq    |  - samples/s  |   8.4 samples/s  | 8.4 samples/s |

### Model
cnndm-unilm-base-cased (fine-tuned on CNN/Daily Mail) [link](https://unilm.blob.core.windows.net/ckpt/cnndm.unilm1-base-cased.bin)

### Task
[CNN/DM](https://github.com/harvardnlp/sent-summary) validation data

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
