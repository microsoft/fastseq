from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

extras = {}

extras["torch"] = ["torch>=1.4.0"]
extras["fairseq"] = ["fairseq>=0.9.0"]
extras["transformers"] = ["transformers>=3.0.2"]

extensions = [
       CUDAExtension('ngrb_cuda', [
                   'fastseq/clib/cuda/ngrb_cuda.cpp',
                   'fastseq/clib/cuda/ngrb_cuda_kernel.cu',
               ]),
        ]

setup(
    name="fastseq",
    version="0.0.3",
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
        "numpy",
        "requests",
        "packaging",
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
