framework="$1"; shift
model="$1"; shift
task="$1"; shift
split="$1"; shift
bss="$1"; shift

shell=benchmark_seq.sh
if [ "$framework" = "fairseq+fastseq" ]; then
    :
elif [ "$framework" = "fairseq" ]; then
    :
elif [ "$framework" = "transformers+fastseq" ]; then
    shell=benchmark_transformers.sh
elif [ "$framework" = "transformers" ]; then
    shell=benchmark_transformers.sh
else
    echo "Unsupported framework '$framework'!"
    exit -1
fi
result_file=perf
touch $result_file
echo "Util Model Task Split BatchSize Samples Tokens Bleu Rouge Loss Perplexity Runtime(seconds) Throughput(samples/s) Throughput(tokens/s)" >> $result_file
bash $shell $framework $model $task $split $bss $result_file $*
ret=$?
echo "Check result at ./$result_file"
exit $ret
