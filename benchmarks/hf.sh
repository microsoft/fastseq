#!/bin/bash
source utils.sh
if [[ $SKIP_BASELINE -eq 0 ]]; then
    export BASELINE_REPO=$CACHE_DIR/transformers_v3.0.2
    #https://github.com/huggingface/transformers.git \
    git_clone_if_not_in_cache \
	https://github.com/JiushengChen/transformers.git \
        $BASELINE_REPO \
        v3.0.2-ngram
fi
