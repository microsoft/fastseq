#!/bin/bash
# USAGE - ./benchmark.sh 
#   [fairseq|fairseq+fastseq|transformers|transformers+fastseq]
#   <model>
#   <task>
#   <split> # train/val/test (text) or train/valid/test (binary)
#   <batch-sizes>
set -eE -o functrace
failure() {
  local lineno=$1
  local msg=$2
  echo "Failed at line $lineno: $msg"
}
trap 'failure ${LINENO} "$BASH_COMMAND"' ERR

# clean cache if you want to start from a clean environment
#rm -rf ~/.cache/fastseq-cache
export LOOP=3   # repeat every generation X times

# TASK - tiny cnn dm 128 samples, each bs takes 1 minute, for quick success or fail test, don't check its speed numbers
#./benchmark.sh transformers+fastseq facebook/bart-large-cnn cnn_dm.128/raw val 32 --task summarization
#./benchmark.sh transformers facebook/bart-large-cnn cnn_dm.128/raw val 32 --task summarization
#./benchmark.sh fairseq+fastseq bart.large.cnn cnn_dm.128/len-1024.bin valid 32/64/128
#./benchmark.sh fairseq bart.large.cnn cnn_dm.128/len-1024.bin valid 32

# TASK - cnn dm 1k samples
./benchmark.sh fairseq+fastseq bart.large.cnn cnn_dm.1k/len-1024.bin valid 32/64/128/256 # take 7 minutes
./benchmark.sh fairseq bart.large.cnn cnn_dm.1k/len-1024.bin valid 32/64     # take 11 minutes
./benchmark.sh transformers+fastseq facebook/bart-large-cnn cnn_dm.1k/raw val 32/64 --task summarization # take 6 minutes
./benchmark.sh transformers facebook/bart-large-cnn cnn_dm.1k/raw val 32/64 --task summarization # take 6 minutes
./benchmark.sh transformers+fastseq sshleifer/distilbart-cnn-12-6 cnn_dm.1k/raw val 32/64/128/256/512 --task summarization   # each bs takes 5 minutes
./benchmark.sh transformers sshleifer/distilbart-cnn-12-6 cnn_dm.1k/raw val 32/64/128/256 --task summarization   # each bs takes 5 minutes
grep "bart.large.cnn cnn_dm.1k/len-1024.bin valid " perf | awk '{if($8!="NA"){c+=1;s+=$8}}END{print s/c}' | bash range.sh 10.4 10.6
grep "facebook/bart-large-cnn cnn_dm.1k/raw val " perf | awk '{if($9!="NA"){c+=1;s+=$9}}END{print s/c}' | bash range.sh 34.7 35
grep "sshleifer/distilbart-cnn-12-6 cnn_dm.1k/raw val " perf | awk '{if($9!="NA"){c+=1;s+=$9}}END{print s/c}' | bash range.sh 35 35.3

# TASK - cnn dm 5k samples
./benchmark.sh fairseq+fastseq bart.large.cnn cnn_dm.5k/len-1024.bin valid 32/64/128/256 # take 30 minutes
./benchmark.sh fairseq bart.large.cnn cnn_dm.5k/len-1024.bin valid 32/64         # take 30 minutes
grep "bart.large.cnn cnn_dm.5k/len-1024.bin valid " perf | awk '{if($8!="NA"){c+=1;s+=$8}}END{print s/c}' | bash range.sh 15.6 15.8

# TASK - wmt16
./benchmark.sh fairseq+fastseq wmt16.en.de.32k wmt16_en_de_bpe32k/bin test 32/64/128/256/512/1024  # take 20 minites
./benchmark.sh fairseq wmt16.en.de.32k wmt16_en_de_bpe32k/bin test 32/64/128/256/512       # take 20 minutes

# TASK - wmt_en_ro
./benchmark.sh transformers+fastseq t5-base wmt_en_ro/raw val 32/64/128/256/512/1024 --task translation_en_to_ro             # each bs takes 5 minutes
./benchmark.sh transformers t5-base wmt_en_ro/raw val 32/64/128/256/512/1024 --task translation_en_to_ro             # each bs takes 5 minutes
./benchmark.sh transformers+fastseq facebook/mbart-large-en-ro wmt_en_ro/raw val 32/64/128/256/512/1024 --task translation    # each bs takes 5 minutes
./benchmark.sh transformers facebook/mbart-large-en-ro wmt_en_ro/raw val 32/64/128/256/512/1024 --task translation    # each bs takes 5 minutes
grep "facebook/mbart-large-en-ro wmt_en_ro/raw val " perf | awk '{if($8!="NA"){c+=1;s+=$8}}END{print s/c}' | bash range.sh 27.6 28.0
grep "t5-base wmt_en_ro/raw val " perf | awk '{if($8!="NA"){c+=1;s+=$8}}END{print s/c}' | bash range.sh 27.6 28.0

# TASK - cnn dm
./benchmark.sh fairseq+fastseq bart.large.cnn cnn_dm/len-1024.bin valid 32/64/128/256  # take 1.5 hours, more valid batch-sizes to run
./benchmark.sh fairseq bart.large.cnn cnn_dm/len-1024.bin valid 32/64          # take 2 hours
./benchmark.sh transformers+fastseq facebook/bart-large-cnn cnn_dm/raw val 32/64 --task summarization    # take 1.5 hours
grep "facebook/bart-large-cnn cnn_dm/raw val " perf | awk '{if($9!="NA"){c+=1;s+=$9}}END{print s/c}' | bash range.sh 44.6 45
./benchmark.sh transformers+fastseq sshleifer/distilbart-cnn-12-6 cnn_dm/raw val 32/64/128 --task summarization  # take 2.5 hours
grep "sshleifer/distilbart-cnn-12-6 cnn_dm/raw val " perf | awk '{if($9!="NA"){c+=1;s+=$9}}END{print s/c}' | bash range.sh 45 45.2
grep "bart.large.cnn cnn_dm/len-1024.bin valid " perf | awk '{if($8!="NA"){c+=1;s+=$8}}END{print s/c}' | bash range.sh 17.8 18

# Speed on cnn dm*
grep -E "fairseq_v0.9.0\+fastseq_v.* bart.large.cnn cnn_dm(.[15]k)?/len-1024.bin valid 32 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 7 100
grep -E "fairseq_v0.9.0\+fastseq_v.* bart.large.cnn cnn_dm(.[15]k)?/len-1024.bin valid 64 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 10 100
grep -E "fairseq_v0.9.0\+fastseq_v.* bart.large.cnn cnn_dm(.[15]k)?/len-1024.bin valid 128 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 10.5 100
grep -E "fairseq_v0.9.0 bart.large.cnn cnn_dm(.[15]k)?/len-1024.bin valid 32 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 2 4
grep -E "transformers_v3.0.2\+fastseq_v.* facebook/bart-large-cnn cnn_dm(.[15]k)?/raw val 32 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 5 100
grep -E "transformers_v3.0.2\+fastseq_v.* facebook/bart-large-cnn cnn_dm(.[15]k)?/raw val 64 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 6 100
grep -E "transformers_v3.0.2 facebook/bart-large-cnn cnn_dm(.[15]k)?/raw val 32 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 3 4
grep -E "transformers_v3.0.2\+fastseq_v.* sshleifer/distilbart-cnn-12-6 cnn_dm(.[15]k)?/raw val 32 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 5.7 100
grep -E "transformers_v3.0.2\+fastseq_v.* sshleifer/distilbart-cnn-12-6 cnn_dm(.[15]k)?/raw val 64 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 6.1 100
grep -E "transformers_v3.0.2 sshleifer/distilbart-cnn-12-6 cnn_dm(.[15]k)?/raw val 32 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 3.8 4.5
grep -E "transformers_v3.0.2 sshleifer/distilbart-cnn-12-6 cnn_dm(.[15]k)?/raw val 64 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 3.8 4.5

# Speed on wmt16
grep -E "fairseq_v0.9.0\+fastseq_v.* wmt16.en.de.32k wmt16_en_de_bpe32k/bin test 32 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 100 100
grep -E "fairseq_v0.9.0\+fastseq_v.* wmt16.en.de.32k wmt16_en_de_bpe32k/bin test 64 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 100 100
grep -E "fairseq_v0.9.0 wmt16.en.de.32k wmt16_en_de_bpe32k/bin test 32 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 100 100
grep -E "fairseq_v0.9.0 wmt16.en.de.32k wmt16_en_de_bpe32k/bin test 64 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 100 100

# Speed on wmt_en_ro
grep -E "transformers_v3.0.2\+fastseq_v.* t5-base wmt_en_ro/raw val 32 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 100 100
grep -E "transformers_v3.0.2\+fastseq_v.* t5-base wmt_en_ro/raw val 64 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 100 100
grep -E "transformers_v3.0.2 t5-base wmt_en_ro/raw val 32 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 100 100
grep -E "transformers_v3.0.2 t5-base wmt_en_ro/raw val 64 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 100 100
