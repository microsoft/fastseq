#!/bin/bash
# Run at current folder. Check ./perf for outputs.
# clean cache if you want to start from a clean environment
#rm -rf ~/.cache/fastseq-cache

for f in `ls ./models/`; do
    echo "Run $f ..."
    bash ./models/$f
    ret=$?
    echo "`date`	$f	$ret" >> perf.summary
done
