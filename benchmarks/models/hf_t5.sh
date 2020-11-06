#!/bin/bash
# Run it at its parent folder, and check result at ../perf.
# USAGE - ./benchmark.sh
#   [fairseq|fairseq+fastseq|transformers|transformers+fastseq]
#   <model>
#   <task>
#   <split> # train/val/test (text) or train/valid/test (binary)
#   <batch-sizes>
source utils.sh

./benchmark.sh transformers t5-base wmt_en_ro/raw val 64 --task translation_en_to_ro --no_repeat_ngram_size 3         # each bs takes 5 minutes
./benchmark.sh transformers+fastseq t5-base wmt_en_ro/raw val 64/128 --task translation_en_to_ro --no_repeat_ngram_size 3  # each bs takes 5 minutes
# Accuracy
grep "t5-base wmt_en_ro/raw val " perf | awk '{if($8!="NA"){c+=1;s+=$8}}END{print s/c}' | bash range.sh 27.42 27.44
# Speed on V100 16GB 250W
grep -E "transformers_v3.0.2 t5-base wmt_en_ro/raw val 64 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 4.6 5.2
grep -E "transformers_v3.0.2\+fastseq_v.* t5-base wmt_en_ro/raw val 64 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 6.8 7.3
grep -E "transformers_v3.0.2\+fastseq_v.* t5-base wmt_en_ro/raw val 128 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 7.5 8.0

