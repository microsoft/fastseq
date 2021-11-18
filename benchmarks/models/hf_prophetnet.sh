#!/bin/bash
# Run it at its parent folder, and check result at ../perf.
# USAGE - ./benchmark.sh
#   [fairseq|fairseq+fastseq|transformers|transformers+fastseq]
#   <model>
#   <task>
#   <split> # train/val/test (text) or train/valid/test (binary)
#   <batch-sizes>
source hf.sh

# MODEL - prophetnet from transformer
# TASK - cnn dm val full set
# ./benchmark.sh \
#     transformers \
#     microsoft/prophetnet-large-uncased \
#     cnn_dm_bert/raw \
#     val \
#     256/320/512 \
#     --task translation \
#     --no_repeat_ngram_size 3
./benchmark.sh \
    transformers+fastseq \
    microsoft/prophetnet-large-uncased \
    cnn_dm_bert/raw \
    val \
    256/320/512 \
    --task translation \
    --no_repeat_ngram_size 3

# # Accuracy
# grep "facebook/bart-large-cnn cnn_dm/raw val " perf \
# 	| awk '{print $9}' \
# 	| awk -F'|' '{if($1!="NA"){c+=1;s+=$1}}END{print s/c}' \
# 	| ./range.sh 0.447 0.448
# # Speed on V100 16GB 250W
# grep -E "transformers_v4.12.0 facebook/bart-large-cnn cnn_dm/raw val 32 " perf \
# 	| awk '{s+=$13}END{if(NR==0) print -1; else print s/NR}' \
# 	| ./range.sh 2 3
# grep -E "transformers_v4.12.0\+fastseq_v.* facebook/bart-large-cnn cnn_dm/raw val 32 " perf \
# 	| awk '{s+=$13}END{print s/NR}' \
# 	| ./range.sh 7 100
# grep -E "transformers_v4.12.0\+fastseq_v.* facebook/bart-large-cnn cnn_dm/raw val 64 " perf \
# 	| awk '{s+=$13}END{print s/NR}' \
# 	| ./range.sh 11 100
# grep -E "transformers_v4.12.0\+fastseq_v.* facebook/bart-large-cnn cnn_dm/raw val 128 " perf \
# 	| awk '{s+=$13}END{print s/NR}' \
# 	| ./range.sh 12 100

