#!/bin/bash
FASTSEQ_TEST_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ENV_PATH=/tmp/
python3 -m venv ${ENV_PATH}/testing_env
source ${ENV_PATH}/testing_env/bin/activate
pip install gitpython
pip install absl-py
pip install packaging
pip install unittest-xml-reporting
pip install lxml
cd ${FASTSEQ_TEST_PATH}/../
pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
rm -rf build/
rm ngram_repeat_block_cuda*.so
which pip
which python
pip install --editable .
USE_EL_ATTN=1 python tests/run_fairseq_tests.py
#USE_EL_ATTN=0 python tests/run_fairseq_tests.py
deactivate
