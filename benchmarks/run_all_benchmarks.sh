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

## FAST TESTS - tiny cnn dm 128 samples, each bs takes 1 minute, for quick success or fail test, don't check its speed numbers
#./benchmark.sh transformers+fastseq facebook/bart-large-cnn cnn_dm.128/raw val 32 --task summarization
#./benchmark.sh transformers facebook/bart-large-cnn cnn_dm.128/raw val 32 --task summarization
#./benchmark.sh fairseq+fastseq bart.large.cnn cnn_dm.128/len-1024.bin valid 32/64/128
#./benchmark.sh fairseq bart.large.cnn cnn_dm.128/len-1024.bin valid 32
#./benchmark.sh transformers+fastseq sshleifer/distilbart-cnn-12-6 cnn_dm.128/raw val 32/64/128 --task summarization
#./benchmark.sh transformers sshleifer/distilbart-cnn-12-6 cnn_dm.128/raw val 32/64/128 --task summarization

# MODEL - bart large cnn from fairseq
./benchmark.sh fairseq bart.large.cnn cnn_dm/len-1024.bin valid 32/64          # each loop 2 hours
./benchmark.sh fairseq+fastseq bart.large.cnn cnn_dm/len-1024.bin valid 32/64/128/256  # each loop 1.5 hours
# Accuracy
grep "bart.large.cnn cnn_dm/len-1024.bin valid " perf | awk '{if($8!="NA"){c+=1;s+=$8}}END{print s/c}' | bash range.sh 17.9 18
# Speed on V100 16GB 250W
grep -E "fairseq_v0.9.0 bart.large.cnn cnn_dm(.[0-9]*k)?/len-1024.bin valid 32 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 2.2 2.4
grep -E "fairseq_v0.9.0\+fastseq_v.* bart.large.cnn cnn_dm/len-1024.bin valid 32 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 6 100
grep -E "fairseq_v0.9.0\+fastseq_v.* bart.large.cnn cnn_dm/len-1024.bin valid 64 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 8.7 100
grep -E "fairseq_v0.9.0\+fastseq_v.* bart.large.cnn cnn_dm/len-1024.bin valid 128 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 10.8 100

# MODEL - bart large cnn from transformer
./benchmark.sh transformers facebook/bart-large-cnn cnn_dm/raw val 32/64 --task summarization    # each loop 2 hours
./benchmark.sh transformers+fastseq facebook/bart-large-cnn cnn_dm/raw val 32/64/128/256 --task summarization    # each loop 2 hours
# Accuracy
grep "facebook/bart-large-cnn cnn_dm/raw val " perf | awk '{if($9!="NA"){c+=1;s+=$9}}END{print s/c}' | bash range.sh 44.78 44.82
# Speed on V100 16GB 250W
grep -E "transformers_v3.0.2 facebook/bart-large-cnn cnn_dm/raw val 32 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 2.2 2.4
grep -E "transformers_v3.0.2\+fastseq_v.* facebook/bart-large-cnn cnn_dm/raw val 32 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 3.9 100
grep -E "transformers_v3.0.2\+fastseq_v.* facebook/bart-large-cnn cnn_dm/raw val 64 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 4.5 100

# MODEL - distibart cnn
./benchmark.sh transformers sshleifer/distilbart-cnn-12-6 cnn_dm/raw val 64/128/256 --task summarization  # take 2.5 hours
./benchmark.sh transformers+fastseq sshleifer/distilbart-cnn-12-6 cnn_dm/raw val 64/128/256/512 --task summarization  # take 2.5 hours
# Accuracy
grep "sshleifer/distilbart-cnn-12-6 cnn_dm/raw val " perf | awk '{if($9!="NA"){c+=1;s+=$9}}END{print s/c}' | bash range.sh 45 45.2
# Speed on V100 16GB 250W
grep -E "transformers_v3.0.2 sshleifer/distilbart-cnn-12-6 cnn_dm/raw val 64 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 3.0 3.2
grep -E "transformers_v3.0.2\+fastseq_v.* sshleifer/distilbart-cnn-12-6 cnn_dm/raw val 64 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 6.1 100

# MODEL - wmt16
./benchmark.sh fairseq wmt16.en.de.32k wmt16_en_de_bpe32k/bin test 256/512       # take 20 minutes
# broken now
#./benchmark.sh fairseq+fastseq wmt16.en.de.32k wmt16_en_de_bpe32k/bin test 256/512/1024/2048  # take 20 minites
# Accuracy
grep " wmt16.en.de.32k wmt16_en_de_bpe32k/bin test " perf | awk '{if($8!="NA"){c+=1;s+=$8}}END{print s/c}' | bash range.sh 0.179 0.181 
# Speed on V100 16GB 250W
grep -E "fairseq_v0.9.0 wmt16.en.de.32k wmt16_en_de_bpe32k/bin test 256 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 49 51
#grep -E "fairseq_v0.9.0\+fastseq_v.* wmt16.en.de.32k wmt16_en_de_bpe32k/bin test 1024 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 69 100

## MODEL - t5
./benchmark.sh transformers t5-base wmt_en_ro/raw val 64/128/256/512/1024 --task translation_en_to_ro          # each bs takes 5 minutes
# broken now
#./benchmark.sh transformers+fastseq t5-base wmt_en_ro/raw val 64/128/256/512/1024 --task translation_en_to_ro  # each bs takes 5 minutes
# Accuracy
grep "t5-base wmt_en_ro/raw val " perf | awk '{if($9!="NA"){c+=1;s+=$9}}END{print s/c}' | bash range.sh 26.9 26.92
# Speed on V100 16GB 250W
grep -E "transformers_v3.0.2 t5-base wmt_en_ro/raw val 64 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 3.5 4
#grep -E "transformers_v3.0.2\+fastseq_v.* t5-base wmt_en_ro/raw val 64 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 100 100

# MODEL - mbart
./benchmark.sh transformers facebook/mbart-large-en-ro wmt_en_ro/raw val 64/128 --task translation    # each bs takes 5 minutes
./benchmark.sh transformers+fastseq facebook/mbart-large-en-ro wmt_en_ro/raw val 64/128/256/512 --task translation    # each bs takes 5 minutes
# Accuracy
# broken now, fastseq result is different, bleu is higher 27.80 vs 27.89
#grep "facebook/mbart-large-en-ro wmt_en_ro/raw val " perf | awk '{if($8!="NA"){c+=1;s+=$8}}END{print s/c}' | bash range.sh 27.79 27.81
# Speed on V100 16GB 250W
grep -E "transformers_v3.0.2 facebook/mbart-large-en-ro wmt_en_ro/raw val 32 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 5.8 6.2
grep -E "transformers_v3.0.2\+fastseq_v.* facebook/mbart-large-en-ro wmt_en_ro/raw val 32 " perf | awk '{s+=$13}END{print s/NR}' | bash range.sh 6.0 100

