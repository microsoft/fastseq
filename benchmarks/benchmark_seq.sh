### start beam search ###
source utils.sh
util="$1"; shift
model="$1"; shift
task="$1"; shift
split="$1"; shift
bss="$1"; shift
perff="$1"; shift

cach_dir=/tmp/fastseq-cache
mkdir -p $cach_dir
model_dir=$cach_dir/$model
mkdir -p $model_dir
model_path=$model_dir/model.pt
download_if_not_in_cache https://fastseq.blob.core.windows.net/data/models/$model/model.pt $model_path

data_dir=$cach_dir/$task
mkdir -p $data_dir
file_list=(dict.source.txt dict.target.txt $split.source-target.source.bin $split.source-target.target.bin $split.source-target.source.idx $split.source-target.target.idx)
for f in "${file_list[@]}"; do
    local_path=$data_dir/$f
    download_if_not_in_cache https://fastseq.blob.core.windows.net/data/tasks/$task/$f $local_path
done

if [[ $util == fairseq* ]]; then
    ver=`pip show fairseq | awk  '{if($1=="Version:")print $2}'`
    util_display="fairseq($ver)"
else
    ver=`pip show fastseq | awk  '{if($1=="Version:")print $2}'`
    util_display="fastseq($ver)"
fi

output_file=/tmp/out.pred
IFS='/' read -ra bs_list <<< "$bss"
for bs in "${bs_list[@]}"; do
    echo "Processing BS=$bs"
    start=`date +%s`
    if [[ $model == wmt* ]]; then
        $util \
            $data_dir \
            --path $model_path \
            --batch-size $bs \
            --beam 4 \
            --lenpen 0.6 \
            --remove-bpe \
            --gen-subset $split $* \
        > $output_file
    else
        $util \
            $data_dir \
            --path $model_path \
            --fp16 \
            --task translation \
            --batch-size $bs \
            --gen-subset $split \
            --truncate-source  \
            --bpe gpt2 \
            --beam 4 \
            --num-workers 4 \
            --min-len 55 \
            --max-len-b 140 \
            --no-repeat-ngram-size 3 \
            --lenpen 2.0 \
            `#--print-alignment` \
            `#--print-step	# KeyError: steps` \
            --skip-invalid-size-inputs-valid-test $* \
        > $output_file
    fi
    ret=$?
    end=`date +%s`
    runtime=$(($end-$start))
    if [ $ret -eq 0 ]; then
        tail=`tail -2 $output_file`
        samples=`echo $tail | awk '{print $3}'`
        tokens=`echo $tail | awk '{sub(/[(]/, ""); print $5}'`
        bleu4=`echo $tail | awk '{gsub(",", ""); print $20}'`
        bleu=`echo $tail | awk '{print $21}'`
        throughput1=`awk -va=$samples -vb=$runtime 'BEGIN{print a/b}'`
        throughput2=`awk -va=$tokens -vb=$runtime 'BEGIN{print a/b}'`
        echo "$util_display $model $task $split $bs $samples $tokens $bleu4 NA NA NA $runtime $throughput1 $throughput2" >> $perff
    else
        echo "$util_display $model $task $split $bs NA NA NA NA NA NA $runtime NA NA" >> $perff
    fi
done
