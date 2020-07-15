from setuptools import find_packages, setup

extras = {}

extras["torch"] = ["torch>=1.4.0"]
extras["fairseq"] = ["fairseq>=0.9.0"]

setup(
    name="fastseq",
    version="0.1",
    author="Microsft AdsBrain Team",
    author_email="fhu@microsoft.com",
    description="Accelerator for State-of-the-art Natural Language Processing Libraries",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="NLP NLG deep learning transformer pytorch tensorflow BERT GPT GPT-2 Microsoft",
    license="Apache",
    url="https://github.com/feihugis/adsbrain-generation",
    #package_dir={"": "src"},
    packages=find_packages(where=".", exclude=["tests"]),
    setup_requires=[
        'cython',
        'numpy',
        'setuptools>=18.0',
    ],
    install_requires=[
        "numpy",
        "packaging",
        "yapf >= 0.30.0",
    ],
    extras_require=extras,
    python_requires=">=3.6.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
