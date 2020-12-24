#!/bin/bash
# Run it at its parent folder, and check result at ../perf. 
# USAGE - ./benchmark.sh 
#   [fairseq|fairseq+fastseq|transformers|transformers+fastseq]
#   <model>
#   <task>
#   <split> # train/val/test (text) or train/valid/test (binary)
#   <batch-sizes>
source hf.sh

# MODEL - mbart
./benchmark.sh \
    transformers \
    facebook/mbart-large-en-ro \
    wmt_en_ro/raw \
    val \
    64 \
    --task translation
./benchmark.sh \
    transformers+fastseq \
    facebook/mbart-large-en-ro \
    wmt_en_ro/raw \
    val \
    64 \
    --task translation
# Accuracy
grep "facebook/mbart-large-en-ro wmt_en_ro/raw val " perf \
	| awk '{if($8!="NA"){c+=1;s+=$8}}END{print s/c}' \
	| ./range.sh 56.1 56.3
# Speed on V100 16GB 250W
grep -E "transformers_v3.0.2 facebook/mbart-large-en-ro wmt_en_ro/raw val 64 " perf \
	| awk '{s+=$13}END{if(NR==0) print -1; else print s/NR}' \
	| ./range.sh 6.0 100
grep -E "transformers_v3.0.2\+fastseq_v.* facebook/mbart-large-en-ro wmt_en_ro/raw val 64 " perf \
	| awk '{s+=$13}END{print s/NR}' \
	| ./range.sh 9 100
