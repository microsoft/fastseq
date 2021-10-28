#!/bin/bash
# Run it at its parent folder, and check result at ../perf.
# USAGE - ./benchmark.sh
#   [fairseq|fairseq+fastseq|transformers|transformers+fastseq]
#   <model>
#   <task>
#   <split> # train/val/test (text) or train/valid/test (binary)
#   <batch-sizes>
source hf.sh

# MODEL - distibart cnn
# TASK - cnn dm val full set
./benchmark.sh \
    transformers \
    hf.sshleifer.distilbart-cnn-12-6.tar.gz \
    cnn_dm/raw \
    val \
    64 \
    --task summarization
./benchmark.sh \
    transformers+fastseq \
    hf.sshleifer.distilbart-cnn-12-6.tar.gz \
    cnn_dm/raw \
    val \
    64/128 \
    --task summarization \
    --postprocess_workers 3

# Accuracy
grep "hf.sshleifer.distilbart-cnn-12-6.tar.gz cnn_dm/raw val " perf \
	| awk '{print $9}' \
	| awk -F'|' '{if($1!="NA"){c+=1;s+=$1}}END{print s/c}' \
	| ./range.sh 0.45 0.452
# Speed on V100 16GB 250W
grep -E "transformers_v4.12.0 hf.sshleifer.distilbart-cnn-12-6.tar.gz cnn_dm/raw val 64 " perf \
	| awk '{s+=$13}END{if(NR==0) print -1; else print s/NR}' \
	| ./range.sh 3 4
grep -E "transformers_v4.12.0\+fastseq_v.* hf.sshleifer.distilbart-cnn-12-6.tar.gz cnn_dm/raw val 64 " perf \
	| awk '{s+=$13}END{print s/NR}' \
	| ./range.sh 16.5 100
grep -E "transformers_v4.12.0\+fastseq_v.* hf.sshleifer.distilbart-cnn-12-6.tar.gz cnn_dm/raw val 128 " perf \
	| awk '{s+=$13}END{print s/NR}' \
	| ./range.sh 18.3 100
