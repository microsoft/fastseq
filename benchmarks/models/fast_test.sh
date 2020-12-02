#!/bin/bash
# Run it at its parent folder, and check result at ../perf.
# USAGE -./benchmark.sh
#   [fairseq|fairseq+fastseq|transformers|transformers+fastseq]
#   <model>
#   <task>
#   <split> # train/val/test (text) or train/valid/test (binary)
#   <batch-sizes>
source utils.sh

# FAST TESTS - tiny cnn dm 128 samples, each bs takes 1 minute, for quick success or fail test, don't check its speed numbers
./benchmark.sh transformers+fastseq facebook/bart-large-cnn cnn_dm.128/raw val 32 --task summarization
./benchmark.sh transformers facebook/bart-large-cnn cnn_dm.128/raw val 32 --task summarization
./benchmark.sh fairseq+fastseq bart.large.cnn cnn_dm.128/len-1024.bin valid 32/64/128
./benchmark.sh fairseq bart.large.cnn cnn_dm.128/len-1024.bin valid 32
./benchmark.sh transformers+fastseq hf.sshleifer.distilbart-cnn-12-6.tar.gz cnn_dm.128/raw val 64/128 --task summarization
./benchmark.sh transformers hf.sshleifer.distilbart-cnn-12-6.tar.gz cnn_dm.128/raw val 64 --task summarization

