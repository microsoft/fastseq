#!/bin/bash
# Run it at its parent folder, and check result at ../perf. 
# USAGE - ./benchmark.sh 
#   [fairseq|fairseq+fastseq|transformers|transformers+fastseq]
#   <model>
#   <task>
#   <split> # train/val/test (text) or train/valid/test (binary)
#   <batch-sizes>
source utils.sh

# below only works with python 1.6+, comment out for now
## MODEL - wmt16
#./benchmark.sh fairseq wmt16.en.de.32k wmt16_en_de_bpe32k/bin test 256  # take 20 minutes
#./benchmark.sh fairseq+fastseq wmt16.en.de.32k wmt16_en_de_bpe32k/bin test 256/512/1024 # take 20 minites
## Accuracy
#grep " wmt16.en.de.32k wmt16_en_de_bpe32k/bin test " perf | awk '{if($8!="NA"){c+=1;s+=$8}}END{print s/c}' | bash range.sh 0.179 0.181 
## Speed on V100 16GB 250W
#grep -E "fairseq_v0.9.0 wmt16.en.de.32k wmt16_en_de_bpe32k/bin test 256 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 49 51
#grep -E "fairseq_v0.9.0\+fastseq_v.* wmt16.en.de.32k wmt16_en_de_bpe32k/bin test 1024 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 75 100
#
