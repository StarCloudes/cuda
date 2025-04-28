#!/usr/bin/env bash
set -e

#Matrix sizes to test (you can自由增删)
SIZES=(512 1024 2048 4096 8192 15360)
# Block configurations: "BX BY"
CONFIGS=(
  "8 8"
  "16 8"
  "8 16"
  "16 16"
  "32 8"
  "8 32"
  "32 16"
  "16 32"
  "32 32"
)

# Number of iterations for each test
P=1000

# Output CSV
OUT=results.csv
echo "Size,BlockX,BlockY,CPU_Time_ms,GPU_Time_ms,Speedup,Max_Diff" > $OUT

for N in "${SIZES[@]}"; do
  M=$N   # 这里用方阵，也可自行设不同 m
  for cfg in "${CONFIGS[@]}"; do
    BX=${cfg%% *}
    BY=${cfg##* }

    echo ">> Testing N=${N} M=${M} block=(${BX}×${BY})"

    # 1) Rebuild with this block size
    make clean
    make heat_sim BLOCK_X=${BX} BLOCK_Y=${BY}

    # 2) Run and capture
    ./heat_sim -n ${N} -m ${M} -p ${P} -t > temp_out.txt

    # 3) Parse metrics
    CPU=$(grep "Total CPU compute time" temp_out.txt | awk '{print $(NF-1)}')
    GPU=$(grep "Total GPU compute time" temp_out.txt | awk '{print $(NF-1)}')
    SPEEDUP=$(grep "Speedup" temp_out.txt | awk '{print $NF}' | tr -d 'x')
    MAXD=$(grep "Max matrix difference" temp_out.txt | awk '{print $NF}')

    # 4) Append
    echo "${N},${BX},${BY},${CPU},${GPU},${SPEEDUP},${MAXD}" >> $OUT
  done
done

echo "All tests done. See ${OUT}"