#!/bin/bash
#rm -rf ~/.cache/fastseq-cache

# USAGE - ./benchmark.sh 
#   [fastseq|fairseq|transformer]
#   <model>
#   <task>
#   <split> # train/val/test (text) or train/valid/test (binary)
#   <batch-sizes>

# TASK - tiny cnn dm 128 samples, each bs takes 1 minute
./benchmark.sh transformer facebook/bart-large-cnn cnn_dm.128/raw val 32 --task summarization
./benchmark.sh fastseq bart.large.cnn cnn_dm.128/len-1024.bin valid 32/64/128
./benchmark.sh fairseq bart.large.cnn cnn_dm.128/len-1024.bin valid 32

# TASK - cnn dm 1k samples
./benchmark.sh fastseq bart.large.cnn cnn_dm.1k/len-1024.bin valid 32/64/128 # take 7 minutes
./benchmark.sh fairseq bart.large.cnn cnn_dm.1k/len-1024.bin valid 32/64     # take 11 minutes
./benchmark.sh transformer facebook/bart-large-cnn cnn_dm.1k/raw val 32/64 --task summarization # take 6 minutes
./benchmark.sh transformer sshleifer/distilbart-cnn-12-6 cnn_dm.1k/raw val 32/64 --task summarization   # each bs takes 5 minutes

# TASK - cnn dm 5k samples
./benchmark.sh fastseq bart.large.cnn cnn_dm.5k/len-1024.bin valid 32/64/128/256 # take 30 minutes
./benchmark.sh fairseq bart.large.cnn cnn_dm.5k/len-1024.bin valid 32/64         # take 30 minutes

# TASK - cnn dm
./benchmark.sh fastseq bart.large.cnn cnn_dm/len-1024.bin valid 32/64/128/256  # take 1.5 hours, more valid batch-sizes to run
tail -4 perf | head -3 | awk '{s+=$8}END{print s}' | bash range.sh 17.88 17.95
tail -4 perf | head -1 | awk '{print $14}' | bash range.sh 460 10000
tail -3 perf | head -1 | awk '{print $14}' | bash range.sh 650 10000
tail -2 perf | head -1 | awk '{print $14}' | bash range.sh 740 10000
./benchmark.sh fairseq bart.large.cnn cnn_dm/len-1024.bin valid 32/64          # take 2 hours
tail -2 perf | head -1 | awk '{s+=$8}END{print s}' | bash range.sh 17.88 17.95
tail -2 perf | head -1 | awk '{print $14}' | bash range.sh 190 10000
./benchmark.sh transformer facebook/bart-large-cnn cnn_dm/raw val 32/64 --task summarization    # take 1.5 hours
tail -2 perf | head -1 | awk '{s+=$9}END{print s}' | bash range.sh 44.6 45
tail -2 perf | head -1 | awk '{print $13}' | bash range.sh 2 100
./benchmark.sh transformer sshleifer/distilbart-cnn-12-6 cnn_dm/raw val 32/64/128 --task summarization  # take 2.5 hours
tail -3 perf | head -2 | awk '{s+=$9}END{print s}' | bash range.sh 45 45.2
tail -3 perf | head -1 | awk '{print $13}' | bash range.sh 2.8 100
tail -2 perf | head -1 | awk '{print $13}' | bash range.sh 2.9 100

# TASK - wmt16
./benchmark.sh fastseq wmt16.en.de.32k wmt16_en_de_bpe32k/bin test 32/64/128/256/512/1024  # take 20 minites
./benchmark.sh fairseq wmt16.en.de.32k wmt16_en_de_bpe32k/bin test 32/64/128/256/512       # take 20 minutes

# TASK - wmt_en_ro
./benchmark.sh transformer t5-base wmt_en_ro/raw val 32/64 --task translation_en_to_ro             # each bs takes 5 minutes
./benchmark.sh transformer facebook/mbart-large-en-ro wmt_en_ro/raw val 32/64 --task translation    # each bs takes 5 minutes
tail -1 perf | awk '{print $8}' | bash range.sh 27.6 28.0
tail -1 perf | awk '{print $13}' | bash range.sh 6 100
tail -2 perf | head -1 | awk '{print $8}' | bash range.sh 27.6 28.0
tail -2 perf | head -1 | awk '{print $13}' | bash range.sh 4 100
