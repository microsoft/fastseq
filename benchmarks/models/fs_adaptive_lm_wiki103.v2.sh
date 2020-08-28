#!/bin/bash
# Run it at its parent folder, and check result at ../perf. 
# USAGE - ./benchmark.sh 
#   [fairseq|fairseq+fastseq|transformers|transformers+fastseq]
#   <model>
#   <task>
#   <split> # train/val/test (text) or train/valid/test (binary)
#   <batch-sizes>
source utils.sh

# clean cache if you want to start from a clean environment
#rm -rf ~/.cache/fastseq-cache
export LOOP=3   # repeat every generation X times

./benchmark.sh fairseq adaptive_lm_wiki103.v2 adaptive_lm_wiki103.v2/bin test 0 --task language_modeling # each loop takes 2 minutes
./benchmark.sh fairseq+fastseq adaptive_lm_wiki103.v2 adaptive_lm_wiki103.v2/bin test 0 --task language_modeling # each loop takes 2 minutes
# Accuracy
grep " adaptive_lm_wiki103.v2 adaptive_lm_wiki103.v2/bin test " perf | awk '{if($10!="NA"){c+=1;s+=$10}}END{print s/c}' | bash range.sh 2.93 2.94   # loss
grep " adaptive_lm_wiki103.v2 adaptive_lm_wiki103.v2/bin test " perf | awk '{if($11!="NA"){c+=1;s+=$11}}END{print s/c}' | bash range.sh 18.66 18.67 # perplexity
# Speed on V100 16GB 250W
grep -E "fairseq_v0.9.0 adaptive_lm_wiki103.v2 adaptive_lm_wiki103.v2/bin test " perf | awk '{s+=$14}END{print s/NR}' | bash range.sh 3100 10000
grep -E "fairseq_v0.9.0\+fastseq_v.* adaptive_lm_wiki103.v2 adaptive_lm_wiki103.v2/bin test " perf | awk '{s+=$14}END{print s/NR}' | bash range.sh 3100 10000

