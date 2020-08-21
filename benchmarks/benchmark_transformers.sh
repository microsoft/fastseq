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
    extra_param="--fastseq_opt 0"
elif [[ "$util" == "transformers+fastseq" ]]; then
    ver1=`pip show transformers | awk  '{if($1=="Version:")print $2}'`
    ver2=`pip show fastseq | awk  '{if($1=="Version:")print $2}'`
    util_display="transformers_v$ver1+fastseq_v$ver2"
fi
IFS='/' read -ra bs_list <<< "$bss"
for bs in "${bs_list[@]}"; do
    echo "Processing BS=$bs"
    start=`date +%s`
    transformers-generate $model $data_dir/$split.source /tmp/out.summary --reference_path $data_dir/$split.target --device cuda --bs $bs --fp16 --score_path /tmp/out.score $extra_param $*
    ret=$?
    echo "Return code: " $ret
    end=`date +%s`
    runtime=$(($end-$start))
    if [ $ret -eq 0 ]; then
        samples=`wc -l /tmp/out.summary | awk '{print $1}'`
        throughput1=`awk -va=$samples -vb=$runtime 'BEGIN{printf "%.1f",a/b}'`
        tokens=NA
        throughput2=NA
        bleu=NA
        rouge1=NA
        rouge2=NA
        rougel=NA
        nf=`awk -F'[\s:,{}]' '{print NF}' /tmp/out.score`
        if [ $nf -eq 4 ]; then
            bleu=`awk -F'[\s:,{}]' '{printf "%.2f",$3}' /tmp/out.score`
        else
            rouge1=`awk -F'[\s:,{}]' '{printf "%.2f",$3}' /tmp/out.score`
            rouge2=`awk -F'[\s:,{}]' '{printf "%.2f",$5}' /tmp/out.score`
            rougel=`awk -F'[\s:,{}]' '{printf "%.2f",$7}' /tmp/out.score`
        fi
        echo "$util_display $model $task $split $bs $samples $tokens $bleu $rouge1 $rouge2 $rougel $runtime $throughput1 $throughput2" >> $perff
    else
        echo "$util_display $model $task $split $bs NA NA NA NA NA NA $runtime NA NA" >> $perff
    fi
done
