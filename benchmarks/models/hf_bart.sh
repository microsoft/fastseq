#!/bin/bash
# Run it at its parent folder, and check result at ../perf.
# USAGE - ./benchmark.sh
#   [fairseq|fairseq+fastseq|transformers|transformers+fastseq]
#   <model>
#   <task>
#   <split> # train/val/test (text) or train/valid/test (binary)
#   <batch-sizes>
source utils.sh

# MODEL - bart large cnn from transformer
# TASK - cnn dm val 1k set
./benchmark.sh transformers facebook/bart-large-cnn cnn_dm.1k/raw val 32 --task summarization    # each loop 5 minutes
./benchmark.sh transformers+fastseq facebook/bart-large-cnn cnn_dm.1k/raw val 32/64/128 --task summarization    # each loop 8 minutes
## TASK - cnn dm val full set
#./benchmark.sh transformers facebook/bart-large-cnn cnn_dm/raw val 32 --task summarization    # each loop 2 hours
#./benchmark.sh transformers+fastseq facebook/bart-large-cnn cnn_dm/raw val 32/64/128 --task summarization    # each loop 2 hours

# Accuracy
grep "facebook/bart-large-cnn cnn_dm.1k/raw val " perf | awk '{print $9}' | awk -F'|' '{if($1!="NA"){c+=1;s+=$1}}END{print s/c}' | bash range.sh 34.8 35
# Speed on V100 16GB 250W
grep -E "transformers_v3.0.2 facebook/bart-large-cnn cnn_dm.1k/raw val 32 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 3.2 3.4
grep -E "transformers_v3.0.2\+fastseq_v.* facebook/bart-large-cnn cnn_dm.1k/raw val 32 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 7.3 100
grep -E "transformers_v3.0.2\+fastseq_v.* facebook/bart-large-cnn cnn_dm.1k/raw val 64 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 9.6 100
grep -E "transformers_v3.0.2\+fastseq_v.* facebook/bart-large-cnn cnn_dm.1k/raw val 128 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 9.9 100

## Accuracy
#grep "facebook/bart-large-cnn cnn_dm/raw val " perf | awk '{print $9}' | awk -F'|' '{if($1!="NA"){c+=1;s+=$1}}END{print s/c}' | bash range.sh 44.78 44.82
## Speed on V100 16GB 250W
#grep -E "transformers_v3.0.2 facebook/bart-large-cnn cnn_dm/raw val 32 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 2.2 2.4
#grep -E "transformers_v3.0.2\+fastseq_v.* facebook/bart-large-cnn cnn_dm/raw val 32 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 3.9 100
#grep -E "transformers_v3.0.2\+fastseq_v.* facebook/bart-large-cnn cnn_dm/raw val 64 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 4.5 100
#grep -E "transformers_v3.0.2\+fastseq_v.* facebook/bart-large-cnn cnn_dm/raw val 128 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 4.9 100

