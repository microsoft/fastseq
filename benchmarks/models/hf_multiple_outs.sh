#!/bin/bash
# Run it at its parent folder, and check result at ../perf.
# USAGE - ./benchmark.sh
#   [fairseq|fairseq+fastseq|transformers|transformers+fastseq]
#   <model>
#   <task>
#   <split> # train/val/test (text) or train/valid/test (binary)
#   <batch-sizes>
source hf.sh
# MODEL - bart large cnn from transformer
# TASK - cnn dm val full set

./benchmark.sh \
    transformers+fastseq \
    gpt2 \
    cnn_dm/raw \
    val \
    64 \
    --task summarization \
    --no_repeat_ngram_size 3 \
    --max_tokenizer_length 512 \
    --max_gen_length 711 \
    --causal_lm \
    --beam 4

./benchmark.sh \
    transformers+fastseq \
    gpt2 \
    cnn_dm/raw \
    val \
    64 \
    --task summarization \
    --no_repeat_ngram_size 3 \
    --max_tokenizer_length 512 \
    --max_gen_length 711 \
    --causal_lm \
    --beam 4 \
    --num_return_sequences 2

./benchmark.sh \
    transformers+fastseq \
    gpt2 \
    cnn_dm/raw \
    val \
    64 \
    --task summarization \
    --no_repeat_ngram_size 3 \
    --max_tokenizer_length 512 \
    --max_gen_length 711 \
    --causal_lm \
    --beam 4 \
    --num_return_sequences 4

    