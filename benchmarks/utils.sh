export LOOP=${LOOP:-3}  # repeat every benchmark X times
export SKIP_BASELINE=${SKIP_BASELINE:-0}
export CACHE_DIR=${CACHE_DIR:-~/.cache/fastseq-cache}
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
    local ret=$4
    if [[ $file == benchmark* ]]; then
        cat $STDERR_FILE
        echo
    fi
    echo "`date` - Failed at $file (line $lineno): $msg"
    exit $ret
}
me=`basename $0`
trap 'failure $me ${LINENO} "$BASH_COMMAND" $?' ERR

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
        if [[ "$#" -eq 3 && $local_path == *.tar.gz ]]; then
	    wd=`pwd`
            cd `dirname $local_path`
            tar xzvf $local_path
	    cd $wd
        fi
    else
        echo "Reuse $local_path"
    fi
}

git_clone_if_not_in_cache() {
    git_url=$1
    local_path=$2
    if [ ! -d "$local_path" ]; then
        echo "Git clone " $git_url " to " $local_path
        git clone $git_url $local_path
        if [ $? -ne 0 ]; then
            echo "Failed to clone '$git_url'"
            exit -1
        fi
        if [[ "$#" -eq 3 ]]; then
	    wd=`pwd`
            cd $local_path
	    git checkout tags/$3
	    cd $wd
        fi
    else
        echo "Reuse $local_path"
    fi
}
