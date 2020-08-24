### start beam search ###
source utils.sh
util="$1"; shift
model="$1"; shift
task="$1"; shift
split="$1"; shift
bss="$1"; shift
perff="$1"; shift

cach_dir=~/.cache/fastseq-cache
mkdir -p $cach_dir
data_dir=$cach_dir/$task
mkdir -p $data_dir
file_list=($split.source $split.target)
for f in "${file_list[@]}"; do
    local_path=$data_dir/$f
    download_if_not_in_cache https://fastseq.blob.core.windows.net/data/tasks/$task/$f $local_path
done

extra_param=""
if [[ $util == transformers ]]; then
    ver=`pip show transformers | awk  '{if($1=="Version:")print $2}'`
    util_display="transformers_v$ver"
    extra_param="--without_fastseq_opt"
elif [[ "$util" == "transformers+fastseq" ]]; then
    ver1=`pip show transformers | awk  '{if($1=="Version:")print $2}'`
    ver2=`pip show fastseq | awk  '{if($1=="Version:")print $2}'`
    util_display="transformers_v$ver1+fastseq_v$ver2"
fi
stdout_file=/tmp/fastseq.stdout
stderr_file=/tmp/fastseq.stderr
summary_file=/tmp/fastseq.summary
score_file=/tmp/fastseq.score
IFS='/' read -ra bs_list <<< "$bss"
for i in `seq $LOOP`; do
for bs in "${bs_list[@]}"; do
    echo "Processing Loop=$i Util=$util_display Model=$model Task=$task Split=$split BS=$bs"
    rm -rf $summary_file $score_file
    start=`date +%s`
    fastseq-evaluate $model $data_dir/$split.source $summary_file --reference_path $data_dir/$split.target --device cuda --bs $bs --fp16 --score_path $score_file $extra_param $* > $stdout_file 2> $stderr_file
    ret=$?
    end=`date +%s`
    runtime=$(($end-$start))
    if [[ $ret -eq 0 && -f "$score_file" && -s "$score_file" ]]; then
        samples=`wc -l $summary_file | awk '{print $1}'`
        throughput1=`awk -va=$samples -vb=$runtime 'BEGIN{printf "%.1f",a/b}'`
        tokens=NA
        throughput2=NA
        bleu=NA
        rouge1=NA
        rouge2=NA
        rougel=NA
        nf=`awk -F'[\s:,{}]' '{print NF}' $score_file`
        if [ $nf -eq 4 ]; then
            bleu=`awk -F'[\s:,{}]' '{printf "%.2f",$3}' $score_file`
        else
            rouge1=`awk -F'[\s:,{}]' '{printf "%.2f",$3}' $score_file`
            rouge2=`awk -F'[\s:,{}]' '{printf "%.2f",$5}' $score_file`
            rougel=`awk -F'[\s:,{}]' '{printf "%.2f",$7}' $score_file`
        fi
        echo "$util_display $model $task $split $bs $samples $tokens $bleu $rouge1 $rouge2 $rougel $runtime $throughput1 $throughput2" >> $perff
    else
        echo "$util_display $model $task $split $bs NA NA NA NA NA NA $runtime NA NA" >> $perff
        if grep -Fq "RuntimeError: CUDA out of memory" $stderr_file; then
            :   # OOM is expected in some bs settings
        else
            cat $stderr_file
            echo "Return code: " $ret
            exit -1
        fi
    fi
done
done
