#!/bin/bash
unset n
read n
numInRange() {
   awk -v n="$1" -v low="$2" -v high="$3" 'BEGIN {if(n<low||n>high) print "Error"; else print "Success";}'
}

#echo "$n $1 $2"
ret=$(numInRange $n $1 $2)
if [[ $ret == Error* ]]; then
    echo "Test failed! $n is not in [$1, $2]."
    exit -1
else
    :
    #echo "Test passed! $n is in [$1, $2]."
fi
