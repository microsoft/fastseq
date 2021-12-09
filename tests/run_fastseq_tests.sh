#!/bin/bash
rm -rf build/
rm ngram_repeat_block_cuda*.so
pip install --editable .
for fastseq_py_test_file in $(find tests/ -name "test_*.py")
do 
  if [[ $fastseq_py_test_file == *"test_fairseq_optimizer"* ]]; then
      echo "Running $fastseq_py_test_file with EL Attention"
      PYTHONIOENCODING=utf8 USE_EL_ATTN=1 python3 $fastseq_py_test_file
  fi
  echo "Running $fastseq_py_test_file"
  PYTHONIOENCODING=utf8 python3 $fastseq_py_test_file
done
