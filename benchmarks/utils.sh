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
