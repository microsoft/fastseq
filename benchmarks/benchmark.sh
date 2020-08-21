util="$1"; shift
model="$1"; shift
task="$1"; shift
split="$1"; shift
bss="$1"; shift

shell=benchmark_seq.sh
if [ "$util" = "fairseq+fastseq" ]; then
    :
elif [ "$util" = "fairseq" ]; then
    :
elif [ "$util" = "transformers+fastseq" ]; then
    shell=benchmark_transformers.sh
elif [ "$util" = "transformers" ]; then
    shell=benchmark_transformers.sh
else
    echo "Unsupported util '$util'!"
    exit -1
fi
result_file=perf
echo "Util Model Task Split BatchSize Samples Tokens BLEU ROUGE1 ROUGE2 ROUGEL Runtime(seconds) Throughput(samples/s) Throughput(tokens/s)" >> $result_file
bash $shell $util $model $task $split $bss $result_file $*
echo "Check result at ./$result_file"
