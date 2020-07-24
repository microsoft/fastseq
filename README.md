<h1 align="Center"> <p> FastSeq </p> </h1>

# Introduction

FastSeq provides efficient implementations of the popular sequence models with fast performance for text generation, summarization, and translation tasks. It can automatically optimize the performance of the pupular NLP toolkits (e.g. [FairSeq](https://github.com/pytorch/fairseq)) by simply `import fastseq`.

# Benchmark

- Run [bart.large.cnn](https://dl.fbaipublicfiles.com/fairseq/models/bart.large.cnn.tar.gz) on NVIDIA-V100-16GB

|         BatchSize        |       32      |       64       |       128      |
|:------------------------:|:-------------:|:--------------:|:--------------:|
|       FairSeq-0.9.0      | 4.2 samples/s |       OOM      |       OOM      |
| FairSeq-0.9.0 + FastSeq  | 9.5 samples/s | 12.8 samples/s | 13.9 samples/s |
where:

- `FairSeq-lasest` refers to [the master branch](https://github.com/pytorch/fairseq)
  of FairSeq

- `FairSeq-latest + FastSeq` runs `FastSeq` on top of `FairSeq-latest`

- Parameters: `beam_size=4`, `lenpen=2.0`, `max_len_b=140`, `min_len=55`, `no_repeat_ngram_size=3`

- More details can be found at [tests/optimiser/fairseq/benchmark_fairseq_optimiser.py](tests/optimiser/fairseq/benchmark_fairseq_optimiser.py)

# Requirements and installation

- Python version >= 3.6
- [torch](http://pytorch.org/) >= 1.4.0
- [fairseq](https://github.com/pytorch/fairseq) >= 0.9.0

```bash
git clone https://github.com/microsoft/fastseq
cd fastseq
pip install --editable ./
```

# Usage

## Example

Only one line of code change is needed to use the optimizations provided by `FastSeq`.

```Python
# import fastseq at the beginning of your program
import fastseq
import torch

# Download BART already finetuned for MNLI
bart = torch.hub.load('pytorch/fairseq', 'bart.large.mnli')
bart.eval()  # disable dropout for evaluation

# Encode a pair of sentences and make a prediction
tokens = bart.encode('FastSeq optimizes FairSeq.', 'FastSeq accelerates FairSeq.')
bart.predict('mnli', tokens).argmax()  # 2: entailment
```

## Run tests

```bash
# run a single test.
python tests/optimiser/fairseq/test_fairseq_optimiser.py

# run benchmark.
python tests/optimiser/fairseq/benchmark_fairseq_optimiser.py

# run all the tests.
python -m unittest discover -s tests/ -p '*.py'
```

## Build

```bash
# build package
python setup.py sdist bdist_wheel
```

# Code Style

## Python coding style

Changes to Python code should conform to [PEP 8](https://www.python.org/dev/peps/pep-0008/). `yapf` can be used to help format the python code, and use `pylint` to check your Python changes.

```bash
# format the code by yapf
yapf --style pep8 -i -r PYTHON_FILE/PACKAGE

# run pylint check
pylint --rcfile=.pylintrc  PYTHON_FILE/PACKAGE
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
