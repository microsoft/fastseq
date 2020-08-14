<h1 align="Center"> <p> FastSeq </p> </h1>

# Introduction

FastSeq provides efficient implementations of the popular sequence models with high performance for text generation, summarization, and translation tasks. It can automatically optimize the performance of the pupular NLP toolkits (e.g. [FairSeq](https://github.com/pytorch/fairseq)) by simply `import fastseq`.

# Benchmark

## Run [bart.large.cnn](https://dl.fbaipublicfiles.com/fairseq/models/bart.large.cnn.tar.gz) 
  - on NVIDIA-V100-16GB

    |         BatchSize        |       32      |       64       |       128      |
    |:------------------------:|:-------------:|:--------------:|:--------------:|
    |       fairseq-0.9.0      | 4.2 samples/s |       OOM      |       OOM      |
    | fairseq-0.9.0 + fastseq  | 9.5 samples/s | 12.8 samples/s | 13.9 samples/s |
  
  - on NVIDIA-V100-32GB

    |         BatchSize        |       32      |       64       |       128      |
    |:------------------------:|:-------------:|:--------------:|:--------------:|
    |       fairseq-0.9.0      | 4.3 samples/s |  4.5 samples/s |       OOM      |
    | fairseq-0.9.0 + fastseq  | 10.3 samples/s| 13.5 samples/s | 14.5 samples/s |

where:

- `fairseq-0.9.0` refers to [the v0.9.0 branch](https://github.com/pytorch/fairseq/tree/v0.9.0)
  of fairseq

- `fairseq-0.9.0 + fastseq` runs `fastseq` on top of `fairseq-0.9.0`

- Parameters: `beam_size=4`, `lenpen=2.0`, `max_len_b=140`, `min_len=55`, `no_repeat_ngram_size=3`

- More details can be found at [tests/optimizer/fairseq/benchmark_fairseq_optimizer.py](https://github.com/microsoft/fastseq/blob/main/tests/optimizer/fairseq/benchmark_fairseq_optimizer.py)

## Run `fastseq-generate` + `fairseq-0.9.0` 

- BART model on NVIDIA-V100-16GB

  |     BatchSize    |       32      |       64       |      128       |
  |:----------------:|:-------------:|:--------------:|:--------------:|
  | fairseq-generate | 3.5 samples/s |      OOM       |      OOM       |
  | fastseq-generate | 10.5 samples/s | 14.7 samples/s | 17.3 samples/s |
  |wo model&data load| 11.3 samples/s | 16.7 samples/s | 20.2 samples/s |
Model load takes 20+s which is relative large when E2E time is only several minutes.

- BART model on NVIDIA-V100-32GB

  |     BatchSize    |       32      |       64      |      128      |
  |:----------------:|:-------------:|:-------------:|:-------------:|
  | fairseq-generate | 3.8 samples/s | 4.0 samples/s |      OOM      |
  | fastseq-generate | 6.4 samples/s | 7.1 samples/s | 7.5 samples/s |
Above table not updated yet due to lack access to 32GB V100.

with the command:

```bash
$ fastseq-generate \
      DATA_BIN_PATH \
      --path MODEL_PATH \
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

- [transformer_vaswani_wmt_en_fr_big](https://github.com/pytorch/fairseq/tree/master/examples/scaling_nmt) model

  |     BatchSize    |       32       |       64       |      128       |      256       |      512       |      1024      |
  |:----------------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|
  | fairseq-generate | 22.8 samples/s | 36.1 samples/s | 38.0 samples/s | 41.0 samples/s | 38.6 samples/s |      OOM       |
  | fastseq-generate | 31.2 samples/s | 50.6 samples/s | 54.1 samples/s | 59.3 samples/s | 63.1 samples/s | 54.6 samples/s |


with the command:

```bash
$ fastseq-generate \
      wmt14.en-fr.joined-dict.newstest2014/ \
      --path wmt14.en-fr.joined-dict.transformer/model.pt \
      --beam 4 \
      --lenpen 0.6 \
      --remove-bpe \
      --batch-size 32
```


# Models

## Supported models in [fairseq](https://github.com/pytorch/fairseq)

- [x] [BART](https://arxiv.org/pdf/1910.13461.pdf)
- [x] [Scaling Neural Machine Translation (Ott et al., 2018)](https://github.com/pytorch/fairseq/blob/master/examples/scaling_nmt/README.md)
- [x] [Mixture Models for Diverse Machine Translation: Tricks of the Trade (Shen et al., 2019)](https://github.com/pytorch/fairseq/blob/master/examples/translation_moe/README.md)
- [x] [Pay Less Attention with Lightweight and Dynamic Convolutions (Wu et al., 2019)](https://github.com/pytorch/fairseq/blob/master/examples/pay_less_attention_paper/README.md)


## Supported models in [huggingFace-transformer](https://github.com/huggingface/transformers)

- [ ] [BART](https://huggingface.co/transformers/model_doc/bart.html)
- [ ] [GPT-2](https://huggingface.co/transformers/model_doc/gpt2.html)
- [ ] [UniLM-V1](https://github.com/microsoft/unilm)
- [ ] [UniLM-V2](https://github.com/microsoft/unilm)
- [ ] [ProphetNet](https://github.com/microsoft/ProphetNet)
- [ ] [T5](https://huggingface.co/transformers/model_doc/t5.html)

# Installation

## Requirements

- Python version >= 3.6
- [torch](http://pytorch.org/) >= 1.4.0
- [fairseq](https://github.com/pytorch/fairseq) >= 0.9.0
- [requets](https://pypi.org/project/requests/) >= 2.24.0
- [absl-py](https://pypi.org/project/absl-py/) >= 0.9.0

## Python package

`fastseq` Python package can be directly installed with pip using

```bash
$ pip install fastseq
```

## Install from the source

```bash
$ git clone https://github.com/microsoft/fastseq
$ cd fastseq
$ pip install --editable ./
```

# Usage

## Example

Only one line of code change is needed to use the optimizations provided by `FastSeq`.

```Python
# import fastseq at the beginning of your program
import fastseq
import torch

# Download bart.large.cnn
bart = torch.hub.load('pytorch/fairseq', 'bart.large.cnn')

bart.cuda()  # use GPU
bart.eval()  # disable dropout for evaluation
bart.half()

slines = ['FastSeq provides efficient implementations of the popular sequence models. Please visit https://github.com/microsoft/fastseq for more details.']

hypotheses = bart.sample(
    slines, beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3)

print(hypotheses)
```

## Run tests

```bash
# run a single test.
$ python tests/optimizer/fairseq/test_fairseq_optimizer.py

# run benchmark.
$ python tests/optimizer/fairseq/benchmark_fairseq_optimizer.py

# run all the tests.
$ python -m unittest discover -s tests/ -p '*.py'
```

## Build

```bash
# build package
$ python setup.py sdist bdist_wheel
```

# Code Style

## Python coding style

Changes to Python code should conform to [PEP 8](https://www.python.org/dev/peps/pep-0008/). `yapf` can be used to help format the python code, and use `pylint` to check your Python changes.

```bash
# format the code by yapf
$ yapf --style pep8 -i -r PYTHON_FILE/PACKAGE

# run pylint check
$ pylint --rcfile=.pylintrc  PYTHON_FILE/PACKAGE
```

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
