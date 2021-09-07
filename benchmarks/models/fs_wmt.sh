#!/bin/bash
# Run it at its parent folder, and check result at ../perf.
# USAGE - ./benchmark.sh
#   [fairseq|fairseq+fastseq|transformers|transformers+fastseq]
#   <model>
#   <task>
#   <split> # train/val/test (text) or train/valid/test (binary)
#   <batch-sizes>
source utils.sh

# MODEL - wmt16
# ./benchmark.sh \
#     fairseq \
#     wmt16.en.de.32k \
#     wmt16_en_de_bpe32k/bin \
#     valid \
#     256
./benchmark.sh \
    fairseq+fastseq \
    wmt16.en.de.32k \
    wmt16_en_de_bpe32k/bin \
    valid \
    256 \
    --postprocess-workers 5
# Accuracy
grep " wmt16.en.de.32k wmt16_en_de_bpe32k/bin valid " perf \
	| awk '{if($8!="NA"){c+=1;s+=$8}}END{print s/c}' \
	| ./range.sh 0.05 0.07
# Speed on V100 16GB 250W
grep -E "fairseq_v0.9.0 wmt16.en.de.32k wmt16_en_de_bpe32k/bin valid 256 " perf \
	| awk '{s+=$13}END{if(NR==0) print -1; else print s/NR}' \
	| ./range.sh 93 100
grep -E "fairseq_v0.9.0\+fastseq_v.* wmt16.en.de.32k wmt16_en_de_bpe32k/bin valid 256 " perf \
	| awk '{s+=$13}END{print s/NR}' \
	| ./range.sh 350 1000
grep -E "fairseq_v0.9.0\+fastseq_v.* wmt16.en.de.32k wmt16_en_de_bpe32k/bin valid 512 " perf \
	| awk '{s+=$13}END{print s/NR}' \
	| ./range.sh 390 1000
grep -E "fairseq_v0.9.0\+fastseq_v.* wmt16.en.de.32k wmt16_en_de_bpe32k/bin valid 1024 " perf \
	| awk '{s+=$13}END{print s/NR}' \
	| ./range.sh 405 1000
