#!/bin/bash
# Run it at its parent folder, and check result at ../perf. 
# USAGE - ./benchmark.sh 
#   [fairseq|fairseq+fastseq|transformers|transformers+fastseq]
#   <model>
#   <task>
#   <split> # train/val/test (text) or train/valid/test (binary)
#   <batch-sizes>
source utils.sh

# MODEL - mbart
./benchmark.sh transformers facebook/mbart-large-en-ro wmt_en_ro/raw val 64 --task translation    # each bs takes 5 minutes
./benchmark.sh transformers+fastseq facebook/mbart-large-en-ro wmt_en_ro/raw val 64 --task translation    # each bs takes 5 minutes
# Accuracy
grep "facebook/mbart-large-en-ro wmt_en_ro/raw val " perf | awk '{if($8!="NA"){c+=1;s+=$8}}END{print s/c}' | bash range.sh 27.79 27.95
# Speed on V100 16GB 250W
grep -E "transformers_v3.0.2 facebook/mbart-large-en-ro wmt_en_ro/raw val 64 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 5.8 6.2
grep -E "transformers_v3.0.2\+fastseq_v.* facebook/mbart-large-en-ro wmt_en_ro/raw val 64 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 6.0 100
