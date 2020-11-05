#!/bin/bash
# Run it at its parent folder, and check result at ../perf.
# USAGE -./benchmark.sh
#   [fairseq|fairseq+fastseq|transformers|transformers+fastseq]
#   <model>
#   <task>
#   <split> # train/val/test (text) or train/valid/test (binary)
#   <batch-sizes>
source utils.sh

# TASK - cnn dm val 1k set
./benchmark.sh fairseq bart.large.cnn cnn_dm.1k/len-1024.bin valid 32          # each loop 7 minutes
./benchmark.sh fairseq+fastseq bart.large.cnn cnn_dm.1k/len-1024.bin valid 32/64/128  # each loop 5 minutes

## TASK - cnn dm val full
#./benchmark.sh fairseq bart.large.cnn cnn_dm/len-1024.bin valid 32          # each loop 2 hours
#./benchmark.sh fairseq+fastseq bart.large.cnn cnn_dm/len-1024.bin valid 32/64/128  # each loop 1.5 hours

# Accuracy
grep "bart.large.cnn cnn_dm.1k/len-1024.bin valid " perf | awk '{if($8!="NA"){c+=1;s+=$8}}END{print s/c}' | bash range.sh 10.4 10.6
# Speed on V100 16GB 250W
grep -E "fairseq_v0.9.0 bart.large.cnn cnn_dm.1k/len-1024.bin valid 32 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 2.3 2.8
grep -E "fairseq_v0.9.0\+fastseq_v.* bart.large.cnn cnn_dm.1k/len-1024.bin valid 32 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 7.9 100
grep -E "fairseq_v0.9.0\+fastseq_v.* bart.large.cnn cnn_dm.1k/len-1024.bin valid 64 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 10.7 100
grep -E "fairseq_v0.9.0\+fastseq_v.* bart.large.cnn cnn_dm.1k/len-1024.bin valid 128 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 12.5 100

## Accuracy
#grep "bart.large.cnn cnn_dm/len-1024.bin valid " perf | awk '{if($8!="NA"){c+=1;s+=$8}}END{print s/c}' | bash range.sh 17.9 18
## Speed on V100 16GB 250W
#grep -E "fairseq_v0.9.0 bart.large.cnn cnn_dm/len-1024.bin valid 32 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 2.2 2.4
#grep -E "fairseq_v0.9.0\+fastseq_v.* bart.large.cnn cnn_dm/len-1024.bin valid 32 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 6 100
#grep -E "fairseq_v0.9.0\+fastseq_v.* bart.large.cnn cnn_dm/len-1024.bin valid 64 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 8.7 100
#grep -E "fairseq_v0.9.0\+fastseq_v.* bart.large.cnn cnn_dm/len-1024.bin valid 128 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 10.8 100
