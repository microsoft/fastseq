#!/bin/bash
source utils.sh
if [[ $SKIP_BASELINE -eq 0 ]]; then
    export BASELINE_REPO=$CACHE_DIR/transformers_v4.12.0
    git_clone_if_not_in_cache \
	https://github.com/huggingface/transformers.git \
        $BASELINE_REPO \
        v4.12.0
fi
