### start beam search ###
source utils.sh
util="$1"; shift
model="$1"; shift
task="$1"; shift
split="$1"; shift
bss="$1"; shift
perff="$1"; shift
IFS='/' read -ra bs_list <<< "$bss"

cach_dir=~/.cache/fastseq-cache
mkdir -p $cach_dir

# model
model_dir=$cach_dir/$model
mkdir -p $model_dir
model_path=$model_dir/model.pt
download_if_not_in_cache https://fastseq.blob.core.windows.net/data/models/$model/model.pt $model_path

type=seq2seq
file_list=(dict.source.txt dict.target.txt $split.source-target.source.bin $split.source-target.target.bin $split.source-target.source.idx $split.source-target.target.idx)
if [[ $* == *language_modeling* ]]; then
    type=lm
    file_list=(dict.txt $split.bin $split.idx)
    bs_list=(NA)
fi

# data
data_dir=$cach_dir/$task
mkdir -p $data_dir
for f in "${file_list[@]}"; do
    local_path=$data_dir/$f
    download_if_not_in_cache https://fastseq.blob.core.windows.net/data/tasks/$task/$f $local_path
done

if [[ $util == fairseq ]]; then
    ver=`pip show fairseq | awk  '{if($1=="Version:")print $2}'`
    util_display="fairseq_v$ver"
    if [[ $type == lm ]]; then
        util=fairseq-eval-lm
    else
        util=fairseq-generate
    fi
elif [[ "$util" == "fairseq+fastseq" ]]; then
    ver1=`pip show fairseq | awk  '{if($1=="Version:")print $2}'`
    ver2=`pip show fastseq | awk  '{if($1=="Version:")print $2}'`
    util_display="fairseq_v$ver1+fastseq_v$ver2"
    if [[ $type == lm ]]; then
        util=fastseq-eval-lm-for-fairseq
    else
        util=fastseq-generate-for-fairseq
    fi
fi

stdout_file=/tmp/fastseq.stdout
stderr_file=/tmp/fastseq.stderr
mark1=" with beam="
mark2="| Evaluated "
for i in `seq $LOOP`; do
for bs in "${bs_list[@]}"; do
    echo "Processing Loop=$i/$LOOP Util=$util_display Model=$model Task=$task Split=$split BS=$bs"
    start=`date +%s`
    if [[ $type == lm ]]; then
        $util $data_dir \
            --path $model_path \
            --sample-break-mode complete \
            --max-tokens 3072 \
            --context-window 2560 \
            --softmax-batch 1024 \
            --fp16 \
            --gen-subset $split $* \
        > $stdout_file 2> $stderr_file
    elif [[ $type == seq2seq && $model == wmt* ]]; then
        $util \
            $data_dir \
            --path $model_path \
            --batch-size $bs \
            --beam 4 \
            --lenpen 0.6 \
            --remove-bpe \
            --gen-subset $split $* \
        > $stdout_file 2> $stderr_file
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
        > $stdout_file 2> $stderr_file
    fi
    ret=$?
    end=`date +%s`
    runtime=$(($end-$start))
    tail=`tail -2 $stdout_file`
    if [[ $ret -eq 0 &&  $tail == *$mark1* ]]; then
        samples=`echo $tail | sed 's/.*Translated \([0-9]*\) sentences.*/\1/'`
        tokens=`echo $tail | sed 's/.*Translated .* sentences (\([0-9]*\) tokens).*/\1/'`
        bleu4=`echo $tail | sed 's/.*BLEU4 = \([.0-9]*\).*/\1/' | awk '{printf "%.2f",$1}'` 
        bleu=`echo $tail | sed 's/.*BLEU4 = [.0-9]*, \([./0-9]*\) .*/\1/'`
        throughput1=`awk -va=$samples -vb=$runtime 'BEGIN{printf "%.1f",a/b}'`
        throughput2=`awk -va=$tokens -vb=$runtime 'BEGIN{printf "%.1f",a/b}'`
        echo "$util_display $model $task $split $bs $samples $tokens $bleu4 NA NA NA $runtime $throughput1 $throughput2" >> $perff
    elif [[ $ret -eq 0 &&  $tail == *$mark2* ]]; then
        samples=NA
        tokens=`echo $tail | sed 's/.*Evaluated \([0-9]*\) tokens.*/\1/'`
        loss=`echo $tail | sed 's/.*Loss.*: \([.0-9]*\),.*/\1/' | awk '{printf "%.2f",$1}'`
        perplexity=`echo $tail | sed 's/.*Perplexity.*: \([.0-9]*\).*/\1/' | awk '{printf "%.2f",$1}'`
        throughput2=`awk -va=$tokens -vb=$runtime 'BEGIN{printf "%.1f",a/b}'`
        echo "$util_display $model $task $split $bs $samples $tokens NA NA $loss $perplexity $runtime NA $throughput2" >> $perff
    else
        echo "$util_display $model $task $split $bs NA NA NA NA NA NA $runtime NA NA" >> $perff
        cat $stderr_file
        echo "Return code: " $ret
        exit -1 # force to fail
    fi
done
done
