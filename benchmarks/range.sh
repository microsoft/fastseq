#!/bin/bash
unset n
read n
numInRange() {
   awk -v n="$1" -v low="$2" -v high="$3" 'BEGIN {if(n<low||n>high) printf "Error: %f not in range [%f, %f]!\n", n, low, high}'
}

#echo "$n $1 $2"
numInRange $n $1 $2
