export LOOP=3   # repeat every generation X times
export CACHE_DIR=~/.cache/fastseq-cache
mkdir -p $CACHE_DIR
export STDOUT_FILE=/tmp/fastseq.stdout
export STDERR_FILE=/tmp/fastseq.stderr
export SUMMARY_FILE=/tmp/fastseq.summary
export SCORE_FILE=/tmp/fastseq.score

set -eE -o functrace
failure() {
    local file=$1
    local lineno=$2
    local msg=$3
    echo "Failed at $file (line $lineno): $msg"
    echo
    cat $STDERR_FILE
}
me=`basename $0`
trap 'failure $me ${LINENO} "$BASH_COMMAND"' ERR

download_if_not_in_cache() {
    remote_path=$1
    local_path=$2
    if [ ! -f "$local_path" ]; then
        echo "Download from " $remote_path " to " $local_path
        wget -c -O $local_path "$remote_path"
        if [ $? -ne 0 ]; then
            echo "Failed to download '$remote_path'"
            exit -1
        fi
    else
        echo "Reuse $local_path"
    fi
}
