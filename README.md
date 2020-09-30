<h1 align="Center"> <p> FastSeq </p> </h1>

## Introduction

FastSeq provides efficient implementations of the popular sequence models (e.g. [ProphetNet](https://github.com/microsoft/ProphetNet)) with high performance for text generation, summarization, and translation tasks. Meanwhile, it can automatically optimize the inference speed of the pupular NLP toolkits (e.g. [FairSeq](https://github.com/pytorch/fairseq) and [HuggingFace-Transformers](https://github.com/huggingface/transformers)) without accuracy loss or code changes by simply `import fastseq`.

## Benchmarks

### FastSeq v.s. FairSeq

- ProphetNet
  |       BatchSize      |       32      |        64       |      128       |
  |:--------------------:|:-------------:|:---------------:|:--------------:|
  |      prophetnet      | 2.7 samples/s |  3.1 samples/s  |      OOM       |
  | prophetnet + fastseq | 5.5 samples/s |  8.4 samples/s  | 10.3 samples/s |

- BART

  |     BatchSize    |       32      |        64       |      128       |
  |:----------------:|:-------------:|:---------------:|:--------------:|
  | fairseq-0.9.0    | 2.7 samples/s |       OOM       |      OOM       |
  | above + fastseq  | 9.0 samples/s | 12.5 samples/s  | 14.5 samples/s |

- [WMT16 En-De](https://github.com/pytorch/fairseq/tree/master/examples/scaling_nmt)

  |     BatchSize    |      256       |      512       |      1024      |
  |:----------------:|:--------------:|:--------------:|:--------------:|
  | fairseq-0.9.0    |  84 samples/s  |      OOM       |      OOM       |
  | above + fastseq  | 129 samples/s  |  131 samples/s |  135 samples/s |

### FastSeq v.s. Transformers

- BART
  |      BatchSize      |       32      |       64       |       128      |
  |:-------------------:|:-------------:|:--------------:|:--------------:|
  | transformers-3.0.2  | 3.4 samples/s |      OOM       |      OOM       |
  |  above + fastseq    | 5.2 samples/s | 6.2 samples/s  | 6.4 samples/s  |
  | transformers-2.11.0 | 2.5 samples/s |      OOM       |      OOM       |
  |  above + fastseq    | 4.4 samples/s | 5.3 samples/s  | >5.3 samples/s |

More details about the benmarks can be found [here](docs/benchmarks.md).

## Supported Models

### Supported models in [FairSeq](https://github.com/pytorch/fairseq)

- [x] [ProphetNet](https://github.com/microsoft/ProphetNet)
- [x] [BART](https://arxiv.org/pdf/1910.13461.pdf)
- [x] [Scaling Neural Machine Translation (Ott et al., 2018)](https://github.com/pytorch/fairseq/blob/master/examples/scaling_nmt/README.md)
- [x] [Mixture Models for Diverse Machine Translation: Tricks of the Trade (Shen et al., 2019)](https://github.com/pytorch/fairseq/blob/master/examples/translation_moe/README.md)
- [x] [Pay Less Attention with Lightweight and Dynamic Convolutions (Wu et al., 2019)](https://github.com/pytorch/fairseq/blob/master/examples/pay_less_attention_paper/README.md)


### Supported models in [HuggingFace-Transformers](https://github.com/huggingface/transformers)

- [x] [BART](https://huggingface.co/transformers/model_doc/bart.html)
- [x] [T5](https://huggingface.co/transformers/model_doc/t5.html)
- [ ] [GPT-2](https://huggingface.co/transformers/model_doc/gpt2.html)
- [ ] [UniLM-V1](https://github.com/microsoft/unilm)
- [ ] [UniLM-V2](https://github.com/microsoft/unilm)
- [ ] [ProphetNet](https://github.com/microsoft/ProphetNet)

## Installation

### Requirements

- Python version >= 3.6
- [torch](http://pytorch.org/) >= 1.4.0
- [fairseq](https://github.com/pytorch/fairseq) >= 0.9.0
- [transformers](https://github.com/huggingface/transformers) >= 3.0.2
- [requets](https://pypi.org/project/requests/) >= 2.24.0
- [absl-py](https://pypi.org/project/absl-py/) >= 0.9.0
- [rouge-score](https://pypi.org/project/rouge-score/)

If you use fairseq or transformers, you only need to install one of them. If you use both, you need to install both.

### Python package

`fastseq` Python package can be directly installed with pip using

```bash
$ pip install fastseq
```

### Install from the source

```bash
$ git clone https://github.com/microsoft/fastseq
$ cd fastseq
$ pip install --editable ./
```

## Usage

### Example

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

### Command line tool for fairseq models
Example

```bash
$ fastseq-generate-for-fairseq \
    cnn_dnn/bin \
    --path bart.large.cnn/model.pt \
    --fp16 \
    --task translation \
    --batch-size 128 \
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

### Command line tool for transformers models
Example

```bash
$ fastseq-generate-for-transformers \
    facebook/bart-large-cnn \
    cnn_dm/val.source \
    out.summary \
    --reference_path cnn_dm/val.target \
    --device cuda \
    --bs 128 \
    --fp16 \
    --score_path out.score \
    --task summarization
```

### Run tests

```bash
# run a single test.
$ python tests/optimizer/fairseq/test_fairseq_optimizer.py

# run benchmark.
$ python tests/optimizer/fairseq/benchmark_fairseq_optimizer.py

# run all the tests.
$ python -m unittest discover -s tests/ -p '*.py'

# run all the benchmarks.
$ cd benchmarks && bash run_all_benchmarks.sh
```

## Code Style

### Python coding style

Changes to Python code should conform to [PEP 8](https://www.python.org/dev/peps/pep-0008/). `yapf` can be used to help format the python code, and use `pylint` to check your Python changes.

```bash
# format the code by yapf
$ yapf --style pep8 -i -r PYTHON_FILE/PACKAGE

# run pylint check
$ pylint --rcfile=.pylintrc  PYTHON_FILE/PACKAGE
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
