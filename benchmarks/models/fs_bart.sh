#!/bin/bash
# Run it at its parent folder, and check result at ../perf.
# USAGE -./benchmark.sh
#   [fairseq|fairseq+fastseq|transformers|transformers+fastseq]
#   <model>
#   <task>
#   <split> # train/val/test (text) or train/valid/test (binary)
#   <batch-sizes>
source utils.sh

# TASK - cnn dm val full
./benchmark.sh \
    fairseq \
    bart.large.cnn \
    cnn_dm/len-1024.bin \
    valid \
    32
./benchmark.sh \
    fairseq+fastseq \
    bart.large.cnn \
    cnn_dm/len-1024.bin \
    valid \
    32/64/128/256 \
    --max-tokens 131072
./benchmark.sh \
    fairseq+fastseq+el \
    bart.large.cnn \
    cnn_dm/len-1024.bin \
    valid \
    320

# Accuracy
grep "bart.large.cnn cnn_dm/len-1024.bin valid " perf \
	| awk '{if($8!="NA"){c+=1;s+=$8}}END{print s/c}' \
	| ./range.sh 17.9 18
# Speed on V100 16GB 250W
grep -E "fairseq_v0.10.2 bart.large.cnn cnn_dm/len-1024.bin valid 32 " perf \
	| awk '{s+=$13}END{if(NR==0) print -1; else print s/NR}' \
	| ./range.sh 2.1 2.7
grep -E "fairseq_v0.10.2\+fastseq_v.* bart.large.cnn cnn_dm/len-1024.bin valid 32 " perf \
	| awk '{s+=$13}END{print s/NR}' \
	| ./range.sh 7.8 100
grep -E "fairseq_v0.10.2\+fastseq_v.* bart.large.cnn cnn_dm/len-1024.bin valid 64 " perf \
	| awk '{s+=$13}END{print s/NR}' \
	| ./range.sh 13.0 100
grep -E "fairseq_v0.10.2\+fastseq_v.* bart.large.cnn cnn_dm/len-1024.bin valid 128 " perf \
	| awk '{s+=$13}END{print s/NR}' \
	| ./range.sh 18.1 100
grep -E "fairseq_v0.10.2\+fastseq_v.* bart.large.cnn cnn_dm/len-1024.bin valid 256 " perf \
	| awk '{s+=$13}END{print s/NR}' \
	| ./range.sh 19 100
grep -E "fairseq_v0.10.2\+fastseq_v.* bart.large.cnn cnn_dm/len-1024.bin valid 320 " perf \
        | awk '{s+=$13}END{print s/NR}' \
        | ./range.sh 25 100
