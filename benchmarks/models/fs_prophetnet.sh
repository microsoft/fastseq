#!/bin/bash
# Run it at its parent folder, and check result at ../perf.
# USAGE -./benchmark.sh
#   [fairseq|fairseq+fastseq|transformers|transformers+fastseq]
#   <model>
#   <task>
#   <split> # train/val/test (text) or train/valid/test (binary)
#   <batch-sizes>
source utils.sh

# ./benchmark.sh fairseq+fastseq prophetnet_large_160G_gigaword_model cnn_dm_bert/bin valid 32/64/128  # it will take 20 minutes per run

./benchmark.sh fairseq+fastseq prophetnet_large_160G_gigaword_model cnn_dm_bert.1k/len-1024.bin valid 32/64/128

# Accuracy
grep "prophetnet_large_160G_gigaword_model cnn_dm_bert.1k/len-1024.bin valid" perf | awk '{if($8!="NA"){c+=1;s+=$8}}END{print s/c}' | bash range.sh 2.2 2.4
# Speed on V100 16GB 250W
grep -E "fairseq_v0.9.0\+fastseq_v.* prophetnet_large_160G_gigaword_model cnn_dm_bert.1k/len-1024.bin valid 32 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 6.1 6.6
grep -E "fairseq_v0.9.0\+fastseq_v.* prophetnet_large_160G_gigaword_model cnn_dm_bert.1k/len-1024.bin valid 64 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 9.2 9.7
grep -E "fairseq_v0.9.0\+fastseq_v.* prophetnet_large_160G_gigaword_model cnn_dm_bert.1k/len-1024.bin valid 128 " perf | awk '{s+=$13}END{print s/NR}'| bash range.sh 11.2 11.7
