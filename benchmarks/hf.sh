#!/bin/bash
source utils.sh
if [[ $SKIP_BASELINE -eq 0 ]]; then
    export BASELINE_REPO=$CACHE_DIR/transformers_v4.11.3
    #https://github.com/huggingface/transformers.git \
    git_clone_if_not_in_cache \
	https://github.com/JiushengChen/transformers.git \
        $BASELINE_REPO \
        v4.11.3-ngram
fi
