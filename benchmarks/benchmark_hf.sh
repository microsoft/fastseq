### start beam search ###
source utils.sh
framework="$1"; shift
model="$1"; shift
task="$1"; shift
split="$1"; shift
bss="$1"; shift
perff="$1"; shift

data_dir=$CACHE_DIR/tasks/$task
mkdir -p $data_dir
file_list=($split.source $split.target)
for f in "${file_list[@]}"; do
    local_path=$data_dir/$f
    download_if_not_in_cache https://fastseq.blob.core.windows.net/data/tasks/$task/$f $local_path
done

util=""
if [[ $framework == transformers ]]; then
    if [[ $SKIP_BASELINE -gt 0 ]]; then
        exit 0
    fi
    ver=`pip show transformers | awk  '{if($1=="Version:")print $2}'`
    framework_versioned="transformers_v$ver"
    util="fastseq-generate-for-transformers --without_fastseq_opt"
elif [[ "$framework" == "transformers+fastseq" ]]; then
    ver1=`pip show transformers | awk  '{if($1=="Version:")print $2}'`
    ver2=`pip show fastseq | awk  '{if($1=="Version:")print $2}'`
    framework_versioned="transformers_v$ver1+fastseq_v$ver2"
    util="fastseq-generate-for-transformers"
fi

model_dir=$CACHE_DIR/models
mkdir -p $model_dir
model_id=$model
if [[ $model == *.tar.gz ]]; then
    local_path=$model_dir/$model
    download_if_not_in_cache https://fastseq.blob.core.windows.net/data/models/$model $local_path uncompress
    model_id=${local_path::-7}
fi
IFS='/' read -ra bs_list <<< "$bss"
for i in `seq $LOOP`; do
for bs in "${bs_list[@]}"; do
    echo "`date` Processing Loop=$i/$LOOP Util=$framework_versioned Model=$model Task=$task Split=$split BS=$bs"
    rm -rf $SUMMARY_FILE $SCORE_FILE
    start=`date +%s`
    $util $model_id $data_dir/$split.source $SUMMARY_FILE --reference_path $data_dir/$split.target --device cuda --bs $bs --fp16 --score_path $SCORE_FILE $* > $STDOUT_FILE 2> $STDERR_FILE
    ret=$?
    end=`date +%s`
    runtime=$(($end-$start))
    if [[ $ret -eq 0 && -f "$SCORE_FILE" && -s "$SCORE_FILE" ]]; then
        samples=`wc -l $SUMMARY_FILE | awk '{print $1}'`
        throughput1=`awk -va=$samples -vb=$runtime 'BEGIN{printf "%.1f",a/b}'`
        tokens=NA
        throughput2=NA
        bleu=NA
        rouge1=NA
        rouge2=NA
        rougel=NA
        nf=`awk -F'[\s:,{}]' '{print NF}' $SCORE_FILE`
        if [ $nf -eq 4 ]; then
            bleu=`sed 's/.*"bleu": \([.0-9]*\).*/\1/' $SCORE_FILE | awk '{printf "%.2f",$1}'`
        else
            rouge1=`sed 's/.*"rouge1": \([.0-9]*\).*/\1/' $SCORE_FILE | awk '{printf "%.3f",$1}'`
            rouge2=`sed 's/.*"rouge2": \([.0-9]*\).*/\1/' $SCORE_FILE | awk '{printf "%.3f",$1}'`
            rougel=`sed 's/.*"rougeL": \([.0-9]*\).*/\1/' $SCORE_FILE | awk '{printf "%.3f",$1}'`
        fi
        echo "$framework_versioned $model $task $split $bs $samples $tokens $bleu $rouge1|$rouge2|$rougel NA NA $runtime $throughput1 $throughput2" >> $perff
    else
        echo "$framework_versioned $model $task $split $bs NA NA NA NA NA NA $runtime NA NA" >> $perff
        cat $STDERR_FILE
        echo "Return code: " $ret
        exit -1 # force to fail
    fi
done
done
