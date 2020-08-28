#!/bin/bash
# Run it at its parent folder, and check result at ../perf. 
# USAGE - ./benchmark.sh 
#   [fairseq|fairseq+fastseq|transformers|transformers+fastseq]
#   <model>
#   <task>
#   <split> # train/val/test (text) or train/valid/test (binary)
#   <batch-sizes>
source utils.sh

./benchmark.sh transformers t5-base wmt_en_ro/raw val 64 --task translation_en_to_ro          # each bs takes 5 minutes
./benchmark.sh transformers+fastseq t5-base wmt_en_ro/raw val 64 --task translation_en_to_ro  # each bs takes 5 minutes
# Accuracy
grep "t5-base wmt_en_ro/raw val " perf | awk '{if($8!="NA"){c+=1;s+=$8}}END{print s/c}' | bash range.sh 27.82 27.84
# Speed on V100 16GB 250W
grep -E "transformers_v3.0.2 t5-base wmt_en_ro/raw val 64 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 7 7.5
# todo: 1, fastseq max bs is still 64. 2, no speed gain.
grep -E "transformers_v3.0.2\+fastseq_v.* t5-base wmt_en_ro/raw val 64 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 7 7.5

