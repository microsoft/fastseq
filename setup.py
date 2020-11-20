# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

FASTSEQ_VERSION = '0.0.4'
MIN_FAIRSEQ_VERSION = '0.9.0'
MAX_FAIRSEQ_VERSION = '0.9.0'
MIN_TRANSFORMERS_VERSION = '3.0.2'
MAX_TRANSFORMER_VERSION = '3.0.2'

def get_fastseq_version():
    return FASTSEQ_VERSION

extras = {}

extras["gitpython"] = ["gitpython>=3.1.7"]
extras["editdistance"] = ["editdistance>=0.5.3"]

extensions = [
    CUDAExtension('ngram_repeat_block_cuda', [
        'fastseq/clib/cuda/ngram_repeat_block_cuda.cpp',
        'fastseq/clib/cuda/ngram_repeat_block_cuda_kernel.cu',]),
]

setup(
    name="fastseq",
    version=get_fastseq_version(),
    author="Microsft AdsBrain Team",
    author_email="fastseq@microsoft.com",
    description="Efficient implementations of sequence models with fast performance",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="NLP NLG deep learning transformer sequence pytorch tensorflow BERT GPT GPT-2 Microsoft",
    license="MIT",
    url="https://github.com/microsoft/fastseq",
    packages=find_packages(where=".", exclude=["benchmarks", "tests", "__py_cache__"]),
    setup_requires=[
        'cython',
        'numpy',
        'setuptools>=18.0',
    ],
    install_requires=[
        "absl-py",
        "filelock",
        "numpy",
        "requests",
        "rouge-score>=0.0.4",
        "packaging",
        "torch>=1.4.0",
        "fairseq >= {}, <= {}".format(MIN_FAIRSEQ_VERSION, MAX_FAIRSEQ_VERSION),
        "transformers >= {}, <= {}".format(
            MIN_TRANSFORMERS_VERSION, MAX_TRANSFORMER_VERSION),
        "pytorch-transformers==1.0.0",
    ],
    extras_require=extras,
    python_requires=">=3.6.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    ext_modules=extensions,
    entry_points={
        'console_scripts': [
            'fastseq-generate-for-fairseq = fastseq_cli.generate:cli_main',
            'fastseq-generate-for-transformers = fastseq_cli.transformers_generate:run_generate',
            'fastseq-eval-lm-for-fairseq = fastseq_cli.eval_lm:cli_main',
        ],
    },
    cmdclass={
        'build_ext': BuildExtension
    },
)
