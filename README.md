<h1 align="Center"> <p> FastSeq </p> </h1>

## Introduction

FastSeq provides efficient implementation of popular sequence models (e.g. [Bart](https://arxiv.org/pdf/1910.13461.pdf), [ProphetNet](https://github.com/microsoft/ProphetNet)) for text generation, summarization, translation tasks etc. It automatically optimizes inference speed based on pupular NLP toolkits (e.g. [FairSeq](https://github.com/pytorch/fairseq) and [HuggingFace-Transformers](https://github.com/huggingface/transformers)) without accuracy loss. All these can be easily done (no need to change any code/model/data if using our command line tool, or simply add one-line code `import fastseq` if using source code).

## Speed Gain
Below shows the generation speed gain by using FastSeq.

| Model            | W/O FastSeq (in samples/s) | W/ FastSeq (in samples/s) | Speedup |
|------------------|:--------------------------:|:-------------------------:|:-----:|
| [ProphetNet](examples/prophetnet/README.md)       | 2.7                        | 10.3                      | 3.8x  |
| [Bart (`fs`)](examples/bart/README.md)              | 2.7                        | 13.3                      | 5x  |
| [Bart (`hf`)](examples/bart/README.md#speedup-bart-huggingface-transformers-version-by-using-fastseq)              | 3.4                        | 9.9                       | 2.9x  |
| [DistilBart (`hf`)](examples/distilbart/README.md)    | 4.0                        | 11.9                       | 3x  |
| [T5 (`hf`)](examples/t5/README.md)                  | 4.8                        | 11.0                       | 2.3x  |
| [WMT16 En-De (`fs`)](examples/wmt/README.md)        | 84.0                       | 124.0                     | 1.5x  |

- All the following benchmarking experiments run on NVIDIA-V100-16GB with [docker](docker/Dockerfile). Highest speed recorded for each model by tuning batch size. For parameter setting details, click link of corresponding model.
- `fs` stands for [Fairseq](https://github.com/pytorch/fairseq) 0.9.0 version, `hf` stands for [Huggingface Transformers](https://github.com/huggingface/transformers) 3.0.2 version.
- Optimizations were automatically applied to all generation/sequence models in Fairseq & Huggingface Transformers. Above only lists a subset of them.

- ## How it works?
We developped a wide range of speedup techniques, including improving beam search efficiency, reducing memory footprint, speeding up calculation for key operations etc, IO speedup etc. To seamlessly connect with community, they were applied to existing models from Fairseq and Huggingface Transformers in the backend, while keeping model interface and usage same as before.

## Installation

### Requirements

- Python version >= 3.6
- [torch](http://pytorch.org/) >= 1.4.0
- [fairseq](https://github.com/pytorch/fairseq) == 0.9.0
- [transformers](https://github.com/huggingface/transformers) == 3.0.2
- [requets](https://pypi.org/project/requests/) >= 2.24.0
- [absl-py](https://pypi.org/project/absl-py/) >= 0.9.0
- [rouge-score](https://pypi.org/project/rouge-score/) >= 0.0.4

If you use fairseq or transformers, you only need to install one of them. If you use both, you need to install both.

### Install from PIP package

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

### Use source code for speedup

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

### Use command line tool to speedup fairseq models
Example usage for bart model on cnn daily mail task.

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
Both model file and task data file are the same as original Fairseq version.

### Use command line tool to speedup transformers models
Example usage for bart model on cnn daily mail task.

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
Both model file and task data file are the same as original Transformers version.

### Run tests

```bash
# run a single test.
$ python tests/optimizer/fairseq/test_fairseq_optimizer.py

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
