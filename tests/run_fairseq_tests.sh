#!/bin/bash
FASTSEQ_TEST_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ENV_PATH=/tmp/
python3 -m venv ${ENV_PATH}/testing_env
source ${ENV_PATH}/testing_env/bin/activate
pip install gitpython
pip install absl-py
pip install packaging
cd ${FASTSEQ_TEST_PATH}/../
pip install --editable .
pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
cd tests
python run_fairseq_tests.py 
deactivate
