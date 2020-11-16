#!/bin/bash

for fastseq_py_test_file in $(find tests/ -name "test_api_*.py")
do
  echo "run $fastseq_py_test_file"
  python $fastseq_py_test_file
done
