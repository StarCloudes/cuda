#!/bin/bash

sizes=(
  "5000 5000"
  "8192 8192"
  "16384 16384"
  "20000 20000"
  "8192 20000"
  "16384 8192"
)

# CUDA block size
blocksizes=(64 128 256 512)

outfile="benchmark_results.txt"
echo "Benchmark Results - $(date)" > $outfile
echo "----------------------------------------" >> $outfile

for size in "${sizes[@]}"; do
  n=$(echo $size | awk '{print $1}')
  m=$(echo $size | awk '{print $2}')
  echo "==== Matrix size: n=$n, m=$m ====" | tee -a $outfile
  for bs in "${blocksizes[@]}"; do
    echo "--- Testing blockSize=$bs ---" | tee -a $outfile
    ./exponentialIntegral.out -n $n -m $m -s $bs -t -g | tee -a $outfile
    echo "" >> $outfile
  done
  echo "" >> $outfile
done