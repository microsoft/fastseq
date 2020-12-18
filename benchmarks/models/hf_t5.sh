#!/bin/bash
# Run it at its parent folder, and check result at ../perf.
# USAGE - ./benchmark.sh
#   [fairseq|fairseq+fastseq|transformers|transformers+fastseq]
#   <model>
#   <task>
#   <split> # train/val/test (text) or train/valid/test (binary)
#   <batch-sizes>
source hf.sh

./benchmark.sh \
    transformers \
    t5-base \
    wmt_en_ro/raw \
    val \
    64 \
    --task translation_en_to_ro \
    --no_repeat_ngram_size 3
./benchmark.sh \
    transformers+fastseq \
    t5-base \
    wmt_en_ro/raw \
    val \
    64/128 \
    --task translation_en_to_ro \
    --no_repeat_ngram_size 3
# Accuracy
grep "t5-base wmt_en_ro/raw val " perf \
	| awk '{if($8!="NA"){c+=1;s+=$8}}END{print s/c}' \
	| ./range.sh 56.91 56.94
# Speed on V100 16GB 250W
grep -E "transformers_v3.0.2 t5-base wmt_en_ro/raw val 64 " perf \
	| awk '{s+=$13}END{if(NR==0) print -1; else print s/NR}' \
	| ./range.sh 5 100
grep -E "transformers_v3.0.2\+fastseq_v.* t5-base wmt_en_ro/raw val 64 " perf \
	| awk '{s+=$13}END{print s/NR}' \
	| ./range.sh 19.3 100
grep -E "transformers_v3.0.2\+fastseq_v.* t5-base wmt_en_ro/raw val 128 " perf \
	| awk '{s+=$13}END{print s/NR}' \
	| ./range.sh 23.8 100
