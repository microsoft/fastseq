<h1 align="Center">
  <p> FastSeq
</h1>

FairSeq provides fast performance and efficient implementations of the popular sequence
models for text generation, summarization, and translation tasks. It can also
automatically and significantly optimize the performance of the pupular NLP
toolkits (e.g. [FairSeq](https://github.com/pytorch/fairseq)) by `import fastseq`.

# Requirements and Installation
* Python version >= 3.6
* [torch]((http://pytorch.org/)) >= 1.4.0
* [fairseq](https://github.com/pytorch/fairseq) >= 0.9.0
* [absl-py](https://github.com/abseil/abseil-py) >= 0.9.0

```bash
git clone https://github.com/microsoft/fastseq
cd fastseq
pip install --editable ./
```

# Run

## Run tests
```bash
python tests/optimiser/fairseq/test_fairseq_optimiser.py
python tests/optimiser/fairseq/benchmark_fairseq_optimiser.py
# run all the tests.
python -m unittest discover -s tests/ -p '*.py'
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
