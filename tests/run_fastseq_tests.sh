#!/bin/bash

for fastseq_py_test_file in $(find tests/ -name "test_*.py")
do 
  echo "Skipping $fastseq_py_test_file"
  # if [[ $fastseq_py_test_file == *"test_generate_fs_cli"* ]]; then
  #       echo "Running $fastseq_py_test_file"
  #       python $fastseq_py_test_file
  # fi
  # if [[ $fastseq_py_test_file == *"test_fairseq_optimizer"* ]]; then
  #     echo "Running $fastseq_py_test_file with EL Attention"
  #     USE_EL_ATTN=1 python $fastseq_py_test_file
  # fi
  # echo "Running $fastseq_py_test_file"
  # python $fastseq_py_test_file
done
