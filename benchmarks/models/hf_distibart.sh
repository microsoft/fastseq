#!/bin/bash
# Run it at its parent folder, and check result at ../perf.
# USAGE - ./benchmark.sh
#   [fairseq|fairseq+fastseq|transformers|transformers+fastseq]
#   <model>
#   <task>
#   <split> # train/val/test (text) or train/valid/test (binary)
#   <batch-sizes>
source utils.sh

# MODEL - distibart cnn
# TASK - cnn dm val 1k set
#./benchmark.sh transformers hf.sshleifer.distilbart-cnn-12-6.tar.gz cnn_dm.1k/raw val 64 --task summarization  # each loop takes 7 minutes
#./benchmark.sh transformers+fastseq hf.sshleifer.distilbart-cnn-12-6.tar.gz cnn_dm.1k/raw val 64/128 --task summarization  # each loop takes 7 minutes
## TASK - cnn dm val full set
./benchmark.sh transformers hf.sshleifer.distilbart-cnn-12-6.tar.gz cnn_dm/raw val 64 --task summarization  # each loop takes 2.5 hours
./benchmark.sh transformers+fastseq hf.sshleifer.distilbart-cnn-12-6.tar.gz cnn_dm/raw val 64/128 --task summarization  # each loop takes 2.5 hours

# Accuracy
#grep "hf.sshleifer.distilbart-cnn-12-6.tar.gz cnn_dm.1k/raw val " perf | awk '{print $9}' | awk -F'|' '{if($1!="NA"){c+=1;s+=$1}}END{print s/c}' | bash range.sh 35.1 35.3
## Speed on V100 16GB 250W
#grep -E "transformers_v3.0.2 hf.sshleifer.distilbart-cnn-12-6.tar.gz cnn_dm.1k/raw val 64 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 4.0 6.0
#grep -E "transformers_v3.0.2\+fastseq_v.* hf.sshleifer.distilbart-cnn-12-6.tar.gz cnn_dm.1k/raw val 64 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 16.4 100
## todo: bigger bs doesn't increase speed
#grep -E "transformers_v3.0.2\+fastseq_v.* hf.sshleifer.distilbart-cnn-12-6.tar.gz cnn_dm.1k/raw val 128 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 18.4 100

# Accuracy
grep "hf.sshleifer.distilbart-cnn-12-6.tar.gz cnn_dm/raw val " perf | awk '{print $9}' | awk -F'|' '{if($1!="NA"){c+=1;s+=$1}}END{print s/c}' | bash range.sh 45 45.1
# Speed on V100 16GB 250W
grep -E "transformers_v3.0.2 hf.sshleifer.distilbart-cnn-12-6.tar.gz cnn_dm/raw val 64 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 2.95 3.05
grep -E "transformers_v3.0.2\+fastseq_v.* hf.sshleifer.distilbart-cnn-12-6.tar.gz cnn_dm/raw val 64 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 16.5 100
# todo: bigger bs doesn't increase speed
grep -E "transformers_v3.0.2\+fastseq_v.* hf.sshleifer.distilbart-cnn-12-6.tar.gz cnn_dm/raw val 128 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 18.3 100
