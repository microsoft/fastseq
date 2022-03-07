#!/bin/bash
# Run it at its parent folder, and check result at ../perf.
# USAGE - ./benchmark.sh
#   [fairseq|fairseq+fastseq|transformers|transformers+fastseq]
#   <model>
#   <task>
#   <split> # train/val/test (text) or train/valid/test (binary)
#   <batch-sizes>
source hf.sh
# MODEL - bart large cnn from transformer
# TASK - cnn dm val full set

./benchmark.sh \
    transformers+fastseq \
    gpt2 \
    cnn_dm/raw \
    val \
    64/128/256 \
    --task summarization \
    --no_repeat_ngram_size 3 \
    --max_tokenizer_length 512 \
    --max_gen_length 711 \
    --causal_lm

./benchmark.sh \
    transformers \
    gpt2 \
    cnn_dm/raw \
    val \
    64/128 \
    --task summarization \
    --no_repeat_ngram_size 3 \
    --max_tokenizer_length 512 \
    --max_gen_length 711 \
    --causal_lm

# Accuracy
grep "gpt2 cnn_dm/raw val " perf \
	| awk '{print $9}' \
	| awk -F'|' '{if($1!="NA"){c+=1;s+=$1}}END{print s/c}' \
	| ./range.sh 0.160 0.162
# Speed on V100 16GB 250W
grep -E "transformers_v4.12.0 gpt2 cnn_dm/raw val 64 " perf \
	| awk '{s+=$13}END{if(NR==0) print -1; else print s/NR}' \
	| ./range.sh 3.5 4.5
grep -E "transformers_v4.12.0\+fastseq_v.* gpt2 cnn_dm/raw val 64 " perf \
	| awk '{s+=$13}END{print s/NR}' \
	| ./range.sh 16 100
grep -E "transformers_v4.12.0\+fastseq_v.* gpt2 cnn_dm/raw val 128 " perf \
	| awk '{s+=$13}END{print s/NR}' \
	| ./range.sh 20 100
grep -E "transformers_v4.12.0\+fastseq_v.* gpt2 cnn_dm/raw val 256 " perf \
	| awk '{s+=$13}END{print s/NR}' \
	| ./range.sh 21 100
    