#!/bin/bash
# Run it at its parent folder, and check result at ../perf. 
# USAGE - ./benchmark.sh 
#   [fairseq|fairseq+fastseq|transformers|transformers+fastseq]
#   <model>
#   <task>
#   <split> # train/val/test (text) or train/valid/test (binary)
#   <batch-sizes>
source utils.sh

# clean cache if you want to start from a clean environment
#rm -rf ~/.cache/fastseq-cache
export LOOP=3   # repeat every generation X times

# MODEL - distibart cnn
# TASK - cnn dm val 1k set
./benchmars.sh transformers sshleifer/distilbart-cnn-12-6 cnn_dm.1k/raw val 64 --task summarization  # each loop takes 7 minutes
./benchmark.sh transformers+fastseq sshleifer/distilbart-cnn-12-6 cnn_dm.1k/raw val 64/128 --task summarization  # each loop takes 7 minutes
# Accuracy
grep "sshleifer/distilbart-cnn-12-6 cnn_dm.1k/raw val " perf | awk '{print $9}' | awk -F'|' '{if($1!="NA"){c+=1;s+=$1}}END{print s/c}' | bash range.sh 35.1 35.3
# Speed on V100 16GB 250W
grep -E "transformers_v3.0.2 sshleifer/distilbart-cnn-12-6 cnn_dm.1k/raw val 64 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 3.9 4.2
grep -E "transformers_v3.0.2\+fastseq_v.* sshleifer/distilbart-cnn-12-6 cnn_dm.1k/raw val 64 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 6.5 100
# todo: bigger bs doesn't increase speed
grep -E "transformers_v3.0.2\+fastseq_v.* sshleifer/distilbart-cnn-12-6 cnn_dm.1k/raw val 128 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 6.5 100

## TASK - cnn dm val full set
#./benchmark.sh transformers sshleifer/distilbart-cnn-12-6 cnn_dm/raw val 64 --task summarization  # each loop takes 2.5 hours
#./benchmark.sh transformers+fastseq sshleifer/distilbart-cnn-12-6 cnn_dm/raw val 64/128 --task summarization  # each loop takes 2.5 hours
## Accuracy
#grep "sshleifer/distilbart-cnn-12-6 cnn_dm/raw val " perf | awk '{print $9}' | awk -F'|' '{if($1!="NA"){c+=1;s+=$1}}END{print s/c}' | bash range.sh 45 45.1
## Speed on V100 16GB 250W
#grep -E "transformers_v3.0.2 sshleifer/distilbart-cnn-12-6 cnn_dm/raw val 64 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 2.95 3.05
#grep -E "transformers_v3.0.2\+fastseq_v.* sshleifer/distilbart-cnn-12-6 cnn_dm/raw val 64 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 5.2 100
## todo: bigger bs doesn't increase speed
#grep -E "transformers_v3.0.2\+fastseq_v.* sshleifer/distilbart-cnn-12-6 cnn_dm/raw val 128 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 5.2 100
