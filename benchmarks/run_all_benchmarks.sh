#!/bin/bash
# Run at current folder. Check ./perf for outputs.
set -eE -o functrace
failure() {
  local lineno=$1
  local msg=$2
  echo "Failed at line $lineno: $msg"
}
trap 'failure ${LINENO} "$BASH_COMMAND"' ERR

# clean cache if you want to start from a clean environment
#rm -rf ~/.cache/fastseq-cache

for f in `ls ./models/`; do
    echo "Run $f ..."
    bash ./models/$f
done
