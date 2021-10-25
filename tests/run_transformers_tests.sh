#!/bin/bash
FASTSEQ_TEST_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ENV_PATH=/tmp/
python3 -m venv ${ENV_PATH}/testing_env
source ${ENV_PATH}/testing_env/bin/activate
pip install gitpython
pip install absl-py
pip install packaging
pip install pytest
pip install timeout-decorator
pip install torch torchvision
pip install unittest-xml-reporting
pip install lxml
cd ${FASTSEQ_TEST_PATH}/../
rm -rf build/
rm ngram_repeat_block_cuda*.so
pip install --editable .
# python tests/run_transformers_tests.py
deactivate
