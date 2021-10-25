#!/bin/bash
# Run it at its parent folder, and check result at ../perf.
# USAGE -./benchmark.sh
#   [fairseq|fairseq+fastseq|transformers|transformers+fastseq]
#   <model>
#   <task>
#   <split> # train/val/test (text) or train/valid/test (binary)
#   <batch-sizes>
source utils.sh

# TODO: update when ProphetNet is compatible with fairseq 0.10.2
# Download ProphetNet repo as the baseline if it does not exist
# prophetnet_repo_path=$CACHE_DIR/ProphetNet
# git_clone_if_not_in_cache \
#     https://github.com/microsoft/ProphetNet.git \
#     $prophetnet_repo_path
#
# ./benchmark.sh \
#     fairseq \
#     prophetnet_large_160G_cnndm_model \
#     cnn_dm_bert/len-512.bin \
#     valid \
#     64 \
#     --user-dir $prophetnet_repo_path/src/prophetnet/
./benchmark.sh \
    fairseq+fastseq \
    prophetnet_large_160G_cnndm_model \
    cnn_dm_bert/len-512.bin \
    valid \
    64/128

# Accuracy
grep "prophetnet_large_160G_cnndm_model cnn_dm_bert/len-512.bin valid" perf \
	| awk '{if($8!="NA"){c+=1;s+=$8}}END{print s/c}' \
	| ./range.sh 19.1 19.2
# # Speed on V100 16GB 250W
# TODO: update when ProphetNet is compatible with fairseq 0.10.2
# grep -E "fairseq_v0.10.2 prophetnet_large_160G_cnndm_model cnn_dm_bert/len-512.bin valid 32 " perf \
# 	| awk '{s+=$13}END{if(NR==0) print -1; else print s/NR}' \
# 	| ./range.sh 2 3
# grep -E "fairseq_v0.10.2 prophetnet_large_160G_cnndm_model cnn_dm_bert/len-512.bin valid 64 " perf \
# 	| awk '{s+=$13}END{if(NR==0) print -1; else print s/NR}' \
# 	| ./range.sh 2 3
grep -E "fairseq_v0.10.2\+fastseq_v.* prophetnet_large_160G_cnndm_model cnn_dm_bert/len-512.bin valid 32 " perf \
	| awk '{s+=$13}END{if(NR==0) print -1; else print s/NR}' \
	| ./range.sh 5.7 6.5
grep -E "fairseq_v0.10.2\+fastseq_v.* prophetnet_large_160G_cnndm_model cnn_dm_bert/len-512.bin valid 64 " perf \
	| awk '{s+=$13}END{print s/NR}' \
	| ./range.sh 7.5 10
grep -E "fairseq_v0.10.2\+fastseq_v.* prophetnet_large_160G_cnndm_model cnn_dm_bert/len-512.bin valid 128 " perf \
	| awk '{s+=$13}END{print s/NR}'\
	| ./range.sh 10 15
